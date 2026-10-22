"""온라인 AIOps 엔진: 이상 탐지 -> 장애 예측 -> 원인 진단(룰) -> 자율 조치(Self-Healing).

Kafka 서비스(detector_service.py)와 오프라인 평가(evaluate_healing.py)가 같은 엔진을 쓴다.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from models import (AE_WINDOW, FC_WINDOW, N_FEAT, SLA_LATENCY_MS, SMOOTH_S,
                    LSTMAutoEncoder, Scaler, TransformerForecaster)

# 진단 결과 -> 조치
POLICY = {
    "interference": "channel_switch",     # 간섭 없는 채널로 AP 채널 전환
    "ap_congestion": "load_balance",      # 여유 있는 인접 AP로 강제 로밍
    "ap_degradation": "roam",             # 인접 AP로 로밍, 없으면 P-5G 전환
    "unknown": "failover_p5g",            # 원인 불명 -> 이중화 링크로 우회
}


def diagnose(win, k_db=6.0, d_util_thr=0.15):
    """원인 진단 룰. win: (AE_WINDOW, F) 원 단위 피처 [rssi, sinr, log지연, log손실, AP사용률].

    최근 3초를 10~20초 전과 비교한다. 판단 순서가 핵심:
      1) SINR이 RSSI 변화로 설명되는 것보다 더 떨어졌다  -> 외부 간섭
      2) AP 사용률이 급등했거나 포화                      -> AP 혼잡
      3) 신호 세기 자체가 급락했거나 커버리지 밖           -> AP 출력 저하
    임계값은 학습용 시나리오(B)에서 정했고, 평가 시나리오(C)에는 손대지 않았다.
    """
    rssi, sinr, util = win[:, 0], win[:, 1], win[:, 4]
    now, ref = slice(-3, None), slice(-20, -10)
    d_rssi = rssi[now].mean() - np.median(rssi[ref])
    d_sinr = sinr[now].mean() - np.median(sinr[ref])
    d_util = util[now].mean() - np.median(util[ref])
    if d_sinr - d_rssi < -k_db:
        return "interference"
    if d_util > d_util_thr or util[now].mean() > 0.85:
        return "ap_congestion"
    if d_rssi < -k_db or rssi[now].mean() < -74:
        return "ap_degradation"
    if sinr[now].mean() < 12:
        return "interference"
    return "unknown"


class AIOpsEngine:
    def __init__(self, art_dir, n_agv, proactive=True, persist=3, cooldown_s=60, device="cpu"):
        cfg = json.load(open(os.path.join(art_dir, "config.json")))
        self.thr = cfg["ae_threshold"]
        self.scaler = Scaler.load(os.path.join(art_dir, "scaler.json"))
        self.ae = LSTMAutoEncoder()
        self.ae.load_state_dict(torch.load(os.path.join(art_dir, "lstm_ae.pt"), map_location=device))
        self.fc = TransformerForecaster()
        self.fc.load_state_dict(torch.load(os.path.join(art_dir, "forecaster.pt"), map_location=device))
        self.ae.eval(), self.fc.eval()
        self.N, self.proactive, self.persist, self.cooldown = n_agv, proactive, persist, cooldown_s
        self.buf = np.zeros((n_agv, FC_WINDOW, N_FEAT), np.float32)
        self.lat = np.zeros((n_agv, SMOOTH_S))
        self.count = 0
        self.exceed = np.zeros(n_agv, int)
        self.cool_until = np.zeros(n_agv)
        self.ap_cool = {}

    @torch.no_grad()
    def update(self, t, feats, latency_ms, link, ap):
        """feats: (N, F) featurize() 결과, latency_ms/link/ap: (N,)."""
        self.buf = np.roll(self.buf, -1, axis=1)
        self.buf[:, -1] = feats
        self.lat = np.roll(self.lat, -1, axis=1)
        self.lat[:, -1] = latency_ms
        self.count += 1
        n = min(self.count, SMOOTH_S)
        lat_s = self.lat[:, -n:].mean(1)

        score = np.full(self.N, np.nan)
        p90 = np.full(self.N, np.nan)
        if self.count >= AE_WINDOW:
            x = torch.from_numpy(self.scaler.transform(self.buf[:, -AE_WINDOW:]))
            score = self.ae.score(x).numpy()
            self.exceed = np.where(score > self.thr, self.exceed + 1, 0)
        if self.count >= FC_WINDOW:
            q = self.fc(torch.from_numpy(self.scaler.transform(self.buf))).numpy()
            p90 = lat_s * np.exp(q[:, :, 1]).max(1)        # 향후 10초 중 최악 90% 분위 지연
        alarm = self.exceed >= self.persist
        warn = (p90 > SLA_LATENCY_MS) & (lat_s < SLA_LATENCY_MS)

        decisions = []
        for i in range(self.N):
            if link[i] == "p5g" or t < self.cool_until[i]:
                continue
            if not (alarm[i] or (self.proactive and warn[i])):
                continue
            cause = diagnose(self.buf[i, -AE_WINDOW:])
            if not alarm[i] and cause == "unknown":
                continue                              # 예측만으로는 원인이 분명할 때만 선제 조치
            action = POLICY[cause]
            if action == "channel_switch":
                if self.ap_cool.get(int(ap[i]), -1) > t:
                    continue
                self.ap_cool[int(ap[i])] = t + 120
            self.cool_until[i] = t + self.cooldown
            decisions.append(dict(t=int(t), agv=i, cause=cause, action=action,
                                  trigger="anomaly" if alarm[i] else "forecast",
                                  score=float(score[i]), p90_ms=float(p90[i])))
        return dict(score=score, alarm=alarm, p90=p90, warn=warn, lat_s=lat_s, decisions=decisions)
