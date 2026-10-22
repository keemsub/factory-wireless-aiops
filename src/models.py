"""시계열 예측(Transformer)과 이상 탐지(LSTM AutoEncoder) 모델, 공용 전처리."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from simulator import FEATURES

N_FEAT = len(FEATURES)
AE_WINDOW = 30        # 이상 탐지 입력 길이 (초)
FC_WINDOW = 60        # 예측 입력 길이 (초)
FC_HORIZON = 10       # 예측 구간 (초)
QUANTILES = (0.5, 0.9)
SMOOTH_S = 5          # SLA 판정용 이동평균 (초)
SLA_LATENCY_MS = 50.0
SLA_LOSS_PCT = 5.0


def featurize(arrs):
    """(T, N) 배열 묶음 -> (T, N, F). 지연·손실은 로그 변환."""
    return np.stack([arrs["rssi"], arrs["sinr"], np.log1p(arrs["latency_ms"]),
                     np.log1p(arrs["loss_pct"]), arrs["ap_util"]], axis=-1).astype(np.float32)


def trailing_mean(x, w=SMOOTH_S):
    """시간축(0번 축) 기준 직전 w초 이동평균."""
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x], 0), 0)
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


class Scaler:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        self.mean = X.reshape(-1, X.shape[-1]).mean(0)
        self.std = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


class LSTMAutoEncoder(nn.Module):
    """정상 구간만으로 학습해, 복원 오차가 크면 이상으로 본다."""

    def __init__(self, n_feat=N_FEAT, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(h[-1])
        seed = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
        o, _ = self.dec(seed)
        return self.out(o)

    @torch.no_grad()
    def score(self, x, tail=10):
        """최근 tail초 복원 오차 평균 = 이상 점수."""
        err = (self(x) - x) ** 2
        return err[:, -tail:].mean(dim=(1, 2))


class TransformerForecaster(nn.Module):
    """최근 60초를 보고 향후 10초 지연(5초 이동평균)의 50%/90% 분위수를 예측.

    출력은 현재 지연 대비 로그 변화량이라, 0이면 '지금과 같음(persistence)'과 같다.
    """

    def __init__(self, n_feat=N_FEAT, window=FC_WINDOW, horizon=FC_HORIZON,
                 d_model=48, nhead=4, nlayers=2, n_q=len(QUANTILES)):
        super().__init__()
        self.horizon, self.n_q = horizon, n_q
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, horizon * n_q))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        q = self.head(h[:, -1]).view(-1, self.horizon, self.n_q)
        # 분위수 교차 방지: q90 = q50 + softplus(.)
        return torch.cat([q[..., :1], q[..., :1] + nn.functional.softplus(q[..., 1:])], -1)


def pinball_loss(pred, y, quantiles=QUANTILES):
    loss = 0
    for i, q in enumerate(quantiles):
        e = y - pred[..., i]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss
