"""공장 AGV 무선 텔레메트리 시뮬레이터.

- AGV가 200m x 100m 사각 트랙을 주행하며 8개 Wi-Fi AP 사이를 로밍한다.
- 1초마다 AGV별 RSSI, SINR, 지연, 패킷손실, AP 사용률을 만든다.
- 세 가지 장애를 무작위로 주입하고, 실제로 품질에 영향을 준 구간만 정답 라벨로 남긴다.
    interference   : 특정 구역·채널의 간섭 (용접기, 비인가 장비 등) -> SINR 저하
    ap_congestion  : AP 트래픽 폭주 -> 사용률·지연 증가
    ap_degradation : AP 출력 저하·고장 -> RSSI 저하
- 제어 명령(채널 전환, 부하 분산 로밍, P-5G 이중화 전환)을 받아 상태에 반영한다.
"""
from __future__ import annotations

import numpy as np

TRACK_W, TRACK_H = 200.0, 100.0
PERIM = 2 * (TRACK_W + TRACK_H)
AP_POS = np.array([[30, 10], [100, 10], [170, 10], [190, 50],
                   [170, 90], [100, 90], [30, 90], [10, 50]], dtype=float)
AP_CHANNELS = np.array([1, 6, 11, 1, 6, 11, 1, 6])
CHANNELS = (1, 6, 11)
N_AP = len(AP_POS)

NOISE_DBM, BG_INTERF_DBM, PTX_DBM = -95.0, -92.0, 20.0
ROAM_THR_DBM, ROAM_HYST_DB = -70.0, 5.0
ACTION_HOLD_S = 120          # 강제 로밍·백업 링크 유지 시간
CHANNEL_SWITCH_DELAY_S = 3   # 채널 전환 소요 시간

CAUSES = ("normal", "interference", "ap_congestion", "ap_degradation")
FEATURES = ("rssi", "sinr", "latency_ms", "loss_pct", "ap_util")


def track_xy(s):
    s = np.mod(s, PERIM)
    a, b, c = TRACK_W, TRACK_W + TRACK_H, 2 * TRACK_W + TRACK_H
    x = np.select([s < a, s < b, s < c], [s, np.full_like(s, TRACK_W), TRACK_W - (s - b)], 0.0)
    y = np.select([s < a, s < b, s < c], [np.zeros_like(s), s - a, np.full_like(s, TRACK_H)],
                  TRACK_H - (s - c))
    return x, y


def path_loss(d):
    """실내 공장 경로손실 (지수 3.0)."""
    return 40.0 + 30.0 * np.log10(np.maximum(d, 3.0))


def dbm_sum(*levels):
    return 10 * np.log10(sum(10 ** (np.asarray(l, dtype=float) / 10) for l in levels))


def latency_model(sinr, util):
    """기본 지연 + 큐잉 지연(사용률) + 재전송 지연(SINR)."""
    u = np.clip(util, 0, 0.97)
    return 4.0 + 4.5 * u / (1 - u) + 60.0 / (1 + np.exp((sinr - 8.0) / 2.5))


def loss_model(sinr, util):
    return 100.0 / (1 + np.exp((sinr - 4.0) / 1.5)) + 25.0 * np.clip((util - 0.88) / 0.09, 0, 1)


def _intensity(ev, t):
    dt = t - ev["start"]
    if dt < 0:
        return 0.0
    if dt < ev["ramp"]:
        return dt / ev["ramp"]
    dt -= ev["ramp"]
    if dt < ev["dur"]:
        return 1.0
    dt -= ev["dur"]
    return max(0.0, 1.0 - dt / 10.0)


class FactorySim:
    def __init__(self, n_agv=20, duration=3600, seed=0, with_events=True, event_gap_s=240):
        rng = np.random.default_rng(seed)
        self.N, self.T = n_agv, duration
        self.speed = rng.uniform(0.8, 1.5, n_agv)
        self.s0 = rng.uniform(0, PERIM, n_agv)

        # 잡음은 미리 생성해 두어, 제어 개입 여부와 무관하게 같은 시나리오가 재현되게 한다.
        e = rng.normal(0, 1, (duration, n_agv))
        shadow = np.zeros_like(e)
        shadow[0] = 3 * e[0]
        for t in range(1, duration):
            shadow[t] = 0.95 * shadow[t - 1] + np.sqrt(1 - 0.95 ** 2) * 3 * e[t]
        self.shadow = shadow
        self.link_offset = rng.normal(0, 2, (n_agv, N_AP))
        self.jitter = rng.lognormal(0, 0.25, (duration, n_agv))
        self.loss_noise = np.abs(rng.normal(0, 0.15, (duration, n_agv)))
        self.util_phase = rng.uniform(0, 2 * np.pi, N_AP)
        self.util_noise = rng.normal(0, 0.02, (duration, N_AP))
        self.p5g_noise = rng.normal(0, 1, (duration, n_agv, 3))
        self.events = self._schedule(rng, event_gap_s) if with_events else []

        self.ap_channel = AP_CHANNELS.copy()
        self.assoc = np.full(n_agv, -1)
        self.force_ap = np.full(n_agv, -1)
        self.force_until = np.zeros(n_agv)
        self.backup_until = np.zeros(n_agv)
        self.pending = []
        self.action_log = []
        self._rssi_all = None
        self.t = 0

    # ------------------------------------------------------------------ 시나리오
    def _schedule(self, rng, gap):
        events, t = [], 300.0
        while True:
            t += rng.exponential(gap)
            if t > self.T - 400:
                return events
            kind = str(rng.choice(["interference", "ap_congestion", "ap_degradation"]))
            ev = dict(kind=kind, start=int(t), dur=float(rng.uniform(60, 180)),
                      ramp=float(rng.uniform(3, 10) if kind == "ap_degradation" else rng.uniform(15, 40)))
            if kind == "interference":
                cx, cy = track_xy(np.array([rng.uniform(0, PERIM)]))
                ev.update(center=(float(cx[0]), float(cy[0])), radius=float(rng.uniform(40, 70)),
                          channel=int(rng.choice(CHANNELS)), peak=float(rng.uniform(-68, -58)))
            elif kind == "ap_congestion":
                ev.update(ap=int(rng.integers(N_AP)), peak=float(rng.uniform(0.45, 0.65)))
            else:
                ev.update(ap=int(rng.integers(N_AP)), peak=float(rng.uniform(12, 20)))
            events.append(ev)

    # ------------------------------------------------------------------ 제어
    def apply_action(self, agv, action):
        """제어 명령을 반영하고 실제로 수행된 조치 이름을 돌려준다."""
        t, cur = self.t, int(self.assoc[agv])
        if action == "channel_switch":
            self.pending.append((t + CHANNEL_SWITCH_DELAY_S, cur))
            done = "channel_switch"
        elif action in ("load_balance", "roam"):
            rs = self._rssi_all[agv].copy()
            rs[cur] = -np.inf
            alt = int(rs.argmax())
            if rs[alt] > -75:
                self.force_ap[agv], self.force_until[agv] = alt, t + ACTION_HOLD_S
                done = action
            else:
                self.backup_until[agv] = t + ACTION_HOLD_S
                done = "failover_p5g"
        elif action == "failover_p5g":
            self.backup_until[agv] = t + ACTION_HOLD_S
            done = "failover_p5g"
        else:
            raise ValueError(action)
        self.action_log.append((t, agv, done))
        return done

    def _apply_pending(self, t):
        keep = []
        for due, ap in self.pending:
            if due > t:
                keep.append((due, ap))
                continue
            # AP가 스펙트럼을 스캔해 간섭이 없는 채널로 옮긴다고 가정
            bad = {ev["channel"] for ev in self.events
                   if ev["kind"] == "interference" and _intensity(ev, t) > 0
                   and np.linalg.norm(AP_POS[ap] - ev["center"]) < ev["radius"] + 60}
            opts = [c for c in CHANNELS if c != self.ap_channel[ap] and c not in bad]
            opts = opts or [c for c in CHANNELS if c != self.ap_channel[ap]]
            self.ap_channel[ap] = opts[0]
        self.pending = keep

    def _associate(self, t, rssi_all):
        best = rssi_all.argmax(1)
        for i in range(self.N):
            if t < self.force_until[i]:
                self.assoc[i] = self.force_ap[i]
            elif self.assoc[i] < 0:
                self.assoc[i] = best[i]
            else:
                cur = self.assoc[i]
                if rssi_all[i, cur] < ROAM_THR_DBM and rssi_all[i, best[i]] > rssi_all[i, cur] + ROAM_HYST_DB:
                    self.assoc[i] = best[i]

    # ------------------------------------------------------------------ 1초 진행
    def step(self):
        t = self.t
        if t >= self.T:
            raise StopIteration
        self._apply_pending(t)
        x, y = track_xy(self.s0 + self.speed * t)
        pos = np.stack([x, y], 1)
        d = np.linalg.norm(pos[:, None, :] - AP_POS[None], axis=2)

        active = [(ev, k) for ev in self.events if (k := _intensity(ev, t)) > 0]
        ptx, boost = np.full(N_AP, PTX_DBM), np.zeros(N_AP)
        for ev, k in active:
            if ev["kind"] == "ap_degradation":
                ptx[ev["ap"]] -= k * ev["peak"]
            elif ev["kind"] == "ap_congestion":
                boost[ev["ap"]] += k * ev["peak"]

        base = -path_loss(d) + self.shadow[t][:, None] + self.link_offset
        rssi_all, rssi_all_cf = ptx[None] + base, PTX_DBM + base
        self._rssi_all = rssi_all
        self._associate(t, rssi_all)
        ap, idx = self.assoc.copy(), np.arange(self.N)
        on_bk = t < self.backup_until

        bg = 0.25 + 0.1 * np.sin(2 * np.pi * t / 1800 + self.util_phase) + self.util_noise[t]
        load = 0.05 * np.bincount(ap[~on_bk], minlength=N_AP)
        util_ap_cf = np.clip(bg + load, 0, 0.97)
        util_ap = np.clip(bg + load + boost, 0, 0.97)

        i_ev = np.full(self.N, -200.0)
        for ev, k in active:
            if ev["kind"] != "interference":
                continue
            hit = (np.linalg.norm(pos - np.array(ev["center"]), axis=1) < ev["radius"]) & \
                  (self.ap_channel[ap] == ev["channel"])
            i_ev[hit] = dbm_sum(i_ev[hit], -100 + k * (ev["peak"] + 100))

        nf = dbm_sum(NOISE_DBM, BG_INTERF_DBM)
        rssi, rssi_cf = rssi_all[idx, ap], rssi_all_cf[idx, ap]
        sinr = rssi - dbm_sum(nf, i_ev)
        sinr_no_int = rssi - nf
        util, util_cf = util_ap[ap], util_ap_cf[ap]
        latency = latency_model(sinr, util) * self.jitter[t]
        loss = np.clip(loss_model(sinr, util) + self.loss_noise[t], 0, 100)

        # 정답 라벨: 장애가 실제로 품질을 의미 있게 떨어뜨린 경우만 (우선순위 출력저하 > 간섭 > 혼잡)
        cause = np.zeros(self.N, dtype=int)
        cause[latency_model(sinr, util) / latency_model(sinr, util_cf) > 1.5] = 2
        cause[(sinr_no_int - sinr) > 6] = 1
        cause[(rssi_cf - rssi) > 6] = 3

        if on_bk.any():   # P-5G 이중화 링크로 우회 중
            n = self.p5g_noise[t][on_bk]
            rssi[on_bk] = -72 + 2 * n[:, 0]
            sinr[on_bk] = 22 + 1.5 * n[:, 1]
            latency[on_bk] = 9 * np.exp(0.15 * n[:, 2])
            loss[on_bk] = 0.05
            util[on_bk] = 0.3
            cause[on_bk] = 0

        self.t += 1
        return dict(t=t, x=x, y=y, ap=ap, link=np.where(on_bk, "p5g", "wifi"),
                    rssi=rssi, sinr=sinr, latency_ms=latency, loss_pct=loss, ap_util=util,
                    label=(cause > 0).astype(int), cause=cause)


def simulate(n_agv=20, duration=3600, seed=0, with_events=True):
    """제어 개입 없이 끝까지 돌린 결과를 (T, N) 배열 묶음으로 돌려준다."""
    sim = FactorySim(n_agv, duration, seed, with_events)
    keys = FEATURES + ("label", "cause")
    out = {k: np.zeros((duration, n_agv)) for k in keys}
    for t in range(duration):
        r = sim.step()
        for k in keys:
            out[k][t] = r[k]
    return out, sim.events


def to_records(r, ts):
    """Kafka로 보낼 JSON 레코드 목록."""
    recs = []
    for i in range(len(r["rssi"])):
        bk = r["link"][i] == "p5g"
        recs.append(dict(
            ts=ts, agv_id=f"AGV-{i:02d}", ap_id="P5G-1" if bk else f"AP-{int(r['ap'][i]):02d}",
            link=str(r["link"][i]), x=round(float(r["x"][i]), 1), y=round(float(r["y"][i]), 1),
            rssi=round(float(r["rssi"][i]), 2), sinr=round(float(r["sinr"][i]), 2),
            latency_ms=round(float(r["latency_ms"][i]), 2), loss_pct=round(float(r["loss_pct"][i]), 3),
            ap_util=round(float(r["ap_util"][i]), 3),
            label=int(r["label"][i]), cause=CAUSES[int(r["cause"][i])]))
    return recs
