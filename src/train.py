"""데이터 생성 -> 모델 학습 -> 오프라인 평가.

시나리오 (AGV 20대, 1초 간격)
  A: 정상 2시간       -> 스케일러, LSTM-AE 학습(앞 80%) / 임계값 산정(뒤 20%)
  B: 장애 포함 3시간  -> Transformer 예측 모델 학습
  C: 장애 포함 2시간  -> 모든 평가 (학습에 쓰지 않은 별도 시드)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from engine import diagnose
from models import (AE_WINDOW, FC_HORIZON, FC_WINDOW, SLA_LATENCY_MS, LSTMAutoEncoder,
                    Scaler, TransformerForecaster, featurize, pinball_loss, trailing_mean)
from simulator import CAUSES, simulate

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 2)


# ---------------------------------------------------------------- 유틸
def windows(X, w):
    """(T, N, F) -> (T-w+1, N, w, F), 마지막 시점 기준 윈도."""
    return sliding_window_view(X, w, axis=0).transpose(0, 1, 3, 2)


def batched(fn, x, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(fn(torch.from_numpy(np.ascontiguousarray(x[i:i + bs]))).numpy())
    return np.concatenate(out)


def persist(mask, k):
    run, out = np.zeros(mask.shape[1], int), np.zeros_like(mask)
    for t in range(len(mask)):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run >= k
    return out


def segments(label, cause, min_len=5):
    segs = []
    for i in range(label.shape[1]):
        lab = np.concatenate([[0], label[:, i], [0]])
        d = np.diff(lab)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            if e - s + 1 >= min_len:
                c = np.bincount(cause[s:e + 1, i].astype(int), minlength=4)[1:].argmax() + 1
                segs.append((i, s, e, int(c)))
    return segs


def detection_report(alarm, label, cause, t0):
    """t0 이전은 윈도가 채워지지 않아 제외."""
    a, l = alarm[t0:], label[t0:]
    p, r, f, _ = precision_recall_fscore_support(l.ravel(), a.ravel(), average="binary", zero_division=0)
    segs = [s for s in segments(label, cause) if s[1] >= t0]
    hit, delays = 0, []
    for i, s, e, _ in segs:
        idx = np.where(alarm[s:e + 1, i])[0]
        if len(idx):
            hit += 1
            delays.append(int(idx[0]))
    fa = 0
    for i in range(alarm.shape[1]):
        rise = np.where(np.diff(alarm[:, i].astype(int)) == 1)[0] + 1
        for t in rise:
            if t >= t0 and label[max(0, t - 10):t + 10, i].sum() == 0:
                fa += 1
    hours = alarm.shape[1] * (len(alarm) - t0) / 3600
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
                incidents=len(segs), incident_recall=round(hit / max(1, len(segs)), 3),
                median_delay_s=float(np.median(delays)) if delays else None,
                false_alarms_per_agv_hour=round(fa / hours, 3))


# ---------------------------------------------------------------- 학습
def train_ae(Xa_s, epochs, log):
    T = len(Xa_s)
    cut = int(T * 0.8)
    W = windows(Xa_s, AE_WINDOW)                       # (T', N, w, F)
    tr = W[:cut - AE_WINDOW + 1:2].reshape(-1, AE_WINDOW, Xa_s.shape[-1])
    va = W[cut:].reshape(-1, AE_WINDOW, Xa_s.shape[-1])
    model = LSTMAutoEncoder()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    trt = torch.from_numpy(np.ascontiguousarray(tr))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(trt))
        tot = 0
        for b in range(0, len(trt), 256):
            x = trt[perm[b:b + 256]]
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        log(f"  AE epoch {ep + 1}/{epochs} recon MSE {tot / len(trt):.4f}")
    model.eval()
    va_scores = batched(model.score, va)
    thr = float(np.percentile(va_scores, 99.5))
    return model, thr


def fc_dataset(X_s, lat_s, stride):
    T = len(X_s)
    ts = np.arange(FC_WINDOW - 1, T - FC_HORIZON, stride)
    W = windows(X_s, FC_WINDOW)                        # index = t - (FC_WINDOW-1)
    x = W[ts - (FC_WINDOW - 1)]                         # (n_t, N, w, F)
    fut = np.stack([lat_s[ts + h] for h in range(1, FC_HORIZON + 1)], -1)   # (n_t, N, H)
    y = np.log(fut) - np.log(lat_s[ts])[..., None]
    return ts, x.reshape(-1, FC_WINDOW, X_s.shape[-1]), y.reshape(-1, FC_HORIZON).astype(np.float32)


def train_fc(Xb_s, lat_b, epochs, log):
    _, x, y = fc_dataset(Xb_s, lat_b, stride=2)
    n_val = int(len(x) * 0.15)
    xt, yt = torch.from_numpy(np.ascontiguousarray(x[:-n_val])), torch.from_numpy(y[:-n_val])
    xv, yv = torch.from_numpy(np.ascontiguousarray(x[-n_val:])), torch.from_numpy(y[-n_val:])
    model = TransformerForecaster()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 256):
            i = perm[b:b + 256]
            loss = pinball_loss(model(xt[i]), yt[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([pinball_loss(model(xv[b:b + 2048]), yv[b:b + 2048]).item()
                                for b in range(0, len(xv), 2048)]))
        log(f"  Forecaster epoch {ep + 1}/{epochs} val pinball {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    ridge = Ridge(alpha=1.0).fit(x[:-n_val, -20:].reshape(len(x) - n_val, -1), y[:-n_val])
    return model, ridge


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../artifacts")
    ap.add_argument("--agv", type=int, default=20)
    ap.add_argument("--ae-epochs", type=int, default=10)
    ap.add_argument("--fc-epochs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:6.1f}s] {m}", flush=True)

    log("시나리오 생성")
    A, _ = simulate(args.agv, 7200, seed=1, with_events=False)
    B, _ = simulate(args.agv, 10800, seed=2, with_events=True)
    C, ev_c = simulate(args.agv, 7200, seed=3, with_events=True)
    Xa, Xb, Xc = featurize(A), featurize(B), featurize(C)
    scaler = Scaler().fit(Xa[: int(len(Xa) * 0.8)])
    scaler.save(os.path.join(args.out, "scaler.json"))
    Xa_s, Xb_s, Xc_s = scaler.transform(Xa), scaler.transform(Xb), scaler.transform(Xc)
    lat_b, lat_c = trailing_mean(B["latency_ms"]), trailing_mean(C["latency_ms"])
    lab_c, cause_c = C["label"].astype(int), C["cause"].astype(int)
    metrics = {"data": {"agv": args.agv, "train_normal_s": 7200, "train_event_s": 10800,
                        "test_s": 7200, "test_events": len(ev_c),
                        "test_anomaly_ratio": round(float(lab_c.mean()), 4)}}

    # ---------------- 이상 탐지
    log("LSTM AutoEncoder 학습 (정상 데이터만)")
    ae, thr = train_ae(Xa_s, args.ae_epochs, log)
    torch.save(ae.state_dict(), os.path.join(args.out, "lstm_ae.pt"))

    t0 = AE_WINDOW - 1
    sc = np.full(lab_c.shape, -np.inf)
    sc[t0:] = batched(ae.score, windows(Xc_s, AE_WINDOW).reshape(-1, AE_WINDOW, Xc_s.shape[-1])) \
        .reshape(-1, args.agv)
    det = {"lstm_ae": detection_report(persist(sc > thr, 3), lab_c, cause_c, t0)}

    rule = (C["rssi"] < -75) | (C["sinr"] < 10) | (C["latency_ms"] > 50) | (C["loss_pct"] > 5)
    det["static_threshold"] = detection_report(persist(rule, 3), lab_c, cause_c, t0)

    cut = int(len(Xa_s) * 0.8)
    iso = IsolationForest(n_estimators=200, random_state=0).fit(
        Xa_s[:cut].reshape(-1, Xa_s.shape[-1])[np.random.choice(cut * args.agv, 50000, replace=False)])
    iso_thr = np.percentile(-iso.score_samples(Xa_s[cut:].reshape(-1, Xa_s.shape[-1])), 99.5)
    iso_sc = -iso.score_samples(Xc_s.reshape(-1, Xc_s.shape[-1])).reshape(lab_c.shape)
    det["isolation_forest"] = detection_report(persist(iso_sc > iso_thr, 3), lab_c, cause_c, t0)
    metrics["anomaly_detection"] = det
    for k, v in det.items():
        log(f"  {k:17s} {v}")

    # ---------------- 원인 진단 (탐지된 장애 대상)
    alarm = persist(sc > thr, 3)
    y_true, y_pred = [], []
    for i, s, e, c in segments(lab_c, cause_c):
        idx = np.where(alarm[s:e + 1, i])[0]
        if s < t0 or not len(idx):
            continue
        tau = s + idx[0]
        if tau < AE_WINDOW - 1:
            continue
        y_true.append(CAUSES[c])
        y_pred.append(diagnose(Xc[tau - AE_WINDOW + 1:tau + 1, i]))
    labels = list(CAUSES[1:]) + ["unknown"]
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    metrics["root_cause"] = {"n": len(y_true), "accuracy": round(acc, 3), "labels": labels,
                             "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    log(f"  원인 진단 정확도 {acc:.3f} (n={len(y_true)})")

    # ---------------- 지연 예측
    log("Transformer 예측 모델 학습")
    fc, ridge = train_fc(Xb_s, lat_b, args.fc_epochs, log)
    torch.save(fc.state_dict(), os.path.join(args.out, "forecaster.pt"))

    ts, xc, yc = fc_dataset(Xc_s, lat_c, stride=1)
    cur = lat_c[ts].reshape(-1)[:, None]
    true_ms = cur * np.exp(yc)
    q = batched(fc, xc)
    p50_ms, p90_ms = cur * np.exp(q[..., 0]), cur * np.exp(q[..., 1])
    ridge_ms = cur * np.exp(ridge.predict(xc[:, -20:].reshape(len(xc), -1)))
    change = np.abs(yc[:, -1]) > np.log(1.5)          # 10초 뒤 지연이 50% 이상 변하는 구간
    mae = lambda p, m=slice(None): round(float(np.abs(p[m] - true_ms[m]).mean()), 3)
    metrics["forecast"] = {
        "mae_ms_all": {"persistence": mae(np.repeat(cur, FC_HORIZON, 1)), "ridge": mae(ridge_ms),
                       "transformer_p50": mae(p50_ms)},
        "mae_ms_transitions": {"persistence": mae(np.repeat(cur, FC_HORIZON, 1), change),
                               "ridge": mae(ridge_ms, change), "transformer_p50": mae(p50_ms, change)},
        "transition_share": round(float(change.mean()), 4),
        "p90_coverage": round(float((true_ms <= p90_ms).mean()), 3),
    }
    log(f"  {metrics['forecast']}")

    # ---------------- SLA 위반 조기 경보
    T, N = lab_c.shape
    p90max = np.full((T, N), np.nan)
    p90max[ts] = p90_ms.max(1).reshape(len(ts), N)
    viol = lat_c > SLA_LATENCY_MS
    warn = (p90max > SLA_LATENCY_MS) & ~viol
    onsets = [(t, i) for i in range(N) for t in range(FC_WINDOW + 30, T)
              if viol[t, i] and not viol[t - 30:t, i].any()]

    def lead_times(sig):
        leads = []
        for t, i in onsets:
            w = np.where(sig[t - 15:t, i])[0]
            leads.append(15 - w[0] if len(w) else 0)
        return np.array(leads)

    lead_fc = lead_times(warn)
    lead_ae = lead_times(alarm)
    lead_any = np.maximum(lead_fc, lead_ae)
    false_warn = 0
    for i in range(N):
        rise = np.where(np.diff(warn[:, i].astype(int)) == 1)[0] + 1
        false_warn += sum(1 for t in rise if not viol[t:t + 20, i].any())
    metrics["early_warning"] = {
        "sla": f"5초 평균 지연 > {SLA_LATENCY_MS:.0f}ms", "violation_onsets": len(onsets),
        "warned_ahead_share": {"forecast": round(float((lead_fc > 0).mean()), 3),
                               "anomaly": round(float((lead_ae > 0).mean()), 3),
                               "combined": round(float((lead_any > 0).mean()), 3)},
        "median_lead_s_when_warned": {
            "forecast": float(np.median(lead_fc[lead_fc > 0])) if (lead_fc > 0).any() else 0,
            "combined": float(np.median(lead_any[lead_any > 0])) if (lead_any > 0).any() else 0},
        "false_warnings_per_agv_hour": round(false_warn / (N * (T - FC_WINDOW) / 3600), 3),
    }
    log(f"  {metrics['early_warning']}")

    json.dump({"ae_threshold": thr, "ae_persist_s": 3, "sla_latency_ms": SLA_LATENCY_MS},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    log("완료")


if __name__ == "__main__":
    main()
