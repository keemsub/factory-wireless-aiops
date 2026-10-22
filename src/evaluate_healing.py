"""자율 조치(Self-Healing) 효과 평가.

같은 장애 시나리오(학습에 쓰지 않은 시드)를 세 가지 방식으로 돌려 비교한다.
  none      : 조치 없음
  reactive  : 이상 탐지(LSTM-AE) 알람 시에만 조치
  proactive : 이상 탐지 + 지연 예측(Transformer p90) 경보로 선제 조치
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine import AIOpsEngine  # noqa: E402
from models import SLA_LATENCY_MS, SLA_LOSS_PCT, featurize, trailing_mean  # noqa: E402
from simulator import FactorySim  # noqa: E402


def run(mode, art, seed, T, N):
    sim = FactorySim(N, T, seed, with_events=True)
    eng = None if mode == "none" else AIOpsEngine(art, N, proactive=(mode == "proactive"))
    keys = ("latency_ms", "loss_pct", "label")
    rec = {k: np.zeros((T, N)) for k in keys}
    score = np.full((T, N), np.nan)
    decisions, step_ms = [], []
    for t in range(T):
        r = sim.step()
        for k in keys:
            rec[k][t] = r[k]
        if eng:
            t0 = time.perf_counter()
            out = eng.update(t, featurize(r), r["latency_ms"], r["link"], r["ap"])
            step_ms.append((time.perf_counter() - t0) * 1000)
            score[t] = out["score"]
            for d in out["decisions"]:
                d["done"] = sim.apply_action(d["agv"], d["action"])
                decisions.append(d)
    lat_s, loss_s = trailing_mean(rec["latency_ms"]), trailing_mean(rec["loss_pct"])
    viol = (lat_s > SLA_LATENCY_MS) | (loss_s > SLA_LOSS_PCT)
    return dict(viol=viol, lat_s=lat_s, score=score, decisions=decisions, label=rec["label"],
                step_ms=float(np.median(step_ms)) if step_ms else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="../artifacts")
    ap.add_argument("--docs", default="../docs")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=7200)
    ap.add_argument("--agv", type=int, default=20)
    args = ap.parse_args()

    res = {m: run(m, args.art, args.seed, args.seconds, args.agv) for m in ("none", "reactive", "proactive")}
    base = res["none"]
    summary = {}
    for m, r in res.items():
        v = int(r["viol"].sum())
        real = sum(1 for d in r["decisions"]
                   if base["label"][max(0, d["t"] - 10):d["t"] + 30, d["agv"]].any())
        summary[m] = {
            "sla_violation_agv_seconds": v,
            "reduction_vs_none": round(1 - v / int(base["viol"].sum()), 3),
            "actions": dict(Counter(d["done"] for d in r["decisions"])),
            "actions_by_trigger": dict(Counter(d["trigger"] for d in r["decisions"])),
            "actions_on_real_incident_share": round(real / len(r["decisions"]), 3) if r["decisions"] else None,
            "inference_ms_per_tick_median": r["step_ms"],
        }
        print(m, summary[m], flush=True)
    summary["sla"] = f"5초 평균 지연 > {SLA_LATENCY_MS:.0f}ms 또는 손실 > {SLA_LOSS_PCT:.0f}%"
    summary["scenario"] = {"seed": args.seed, "seconds": args.seconds, "agv": args.agv}
    json.dump(summary, open(os.path.join(args.art, "healing_metrics.json"), "w"), indent=2, ensure_ascii=False)

    # 예시 그림: 위반이 가장 많이 줄어든 AGV의 한 장애 구간
    gain = base["viol"].sum(0) - res["proactive"]["viol"].sum(0)
    i = int(gain.argmax())
    t_mid = int(np.where(base["viol"][:, i] & ~res["proactive"]["viol"][:, i])[0][0])
    lo, hi = max(0, t_mid - 120), min(args.seconds, t_mid + 240)
    tt = np.arange(lo, hi)
    os.makedirs(args.docs, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(tt, base["lat_s"][lo:hi, i], color="#c0392b", label="no action")
    ax[0].plot(tt, res["proactive"]["lat_s"][lo:hi, i], color="#2471a3", label="self-healing (proactive)")
    ax[0].axhline(SLA_LATENCY_MS, ls="--", color="gray", lw=1)
    ax[0].text(lo + 2, SLA_LATENCY_MS * 1.05, "SLA 50ms", color="gray", fontsize=8)
    lab = base["label"][lo:hi, i] > 0
    ax[0].fill_between(tt, 0, 1, where=lab, color="orange", alpha=0.15,
                       transform=ax[0].get_xaxis_transform(), label="injected fault (ground truth)")
    for d in res["proactive"]["decisions"]:
        if d["agv"] == i and lo <= d["t"] < hi:
            ax[0].axvline(d["t"], color="#1e8449", lw=1.2)
            ax[0].text(d["t"] + 2, ax[0].get_ylim()[1] * 0.85, f"{d['done']}\n({d['trigger']})",
                       color="#1e8449", fontsize=8)
    ax[0].set_ylabel("latency, 5s mean (ms)")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(tt, res["proactive"]["score"][lo:hi, i], color="#7d3c98")
    thr = json.load(open(os.path.join(args.art, "config.json")))["ae_threshold"]
    ax[1].axhline(thr, ls="--", color="gray", lw=1)
    ax[1].set_ylabel("anomaly score (LSTM-AE)")
    ax[1].set_xlabel("time (s)")
    ax[0].set_title(f"AGV-{i:02d}: no action vs. self-healing")
    fig.tight_layout()
    fig.savefig(os.path.join(args.docs, "healing_example.png"), dpi=130)
    print("saved figure for AGV", i, "around t", t_mid)


if __name__ == "__main__":
    main()
