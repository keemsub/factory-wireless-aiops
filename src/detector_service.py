"""AIOps 탐지·제어 서비스.

Kafka 'wireless.telemetry'를 구독해 1초 단위로 AGV 지표를 모으고,
이상 탐지 -> 지연 예측 -> 원인 진단 -> 조치 결정을 수행한 뒤
조치를 'wireless.actions'로 발행한다. 모든 지표는 Prometheus(:8000/metrics)로 노출한다.

--local 옵션을 주면 Kafka 없이 시뮬레이터를 같은 프로세스에서 돌린다 (Docker 없이 빠른 확인용).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from engine import AIOpsEngine
from models import featurize
from producer import ACTION_TOPIC, TELEMETRY_TOPIC, connect
from simulator import FEATURES, FactorySim, to_records

G = {f: Gauge(f"wireless_{f}", f"AGV 무선 지표: {f}", ["agv"]) for f in FEATURES}
G_SCORE = Gauge("aiops_anomaly_score", "LSTM-AE 이상 점수", ["agv"])
G_THR = Gauge("aiops_anomaly_threshold", "이상 점수 임계값")
G_ALARM = Gauge("aiops_alarm", "이상 알람 (1=발생)", ["agv"])
G_P90 = Gauge("aiops_forecast_p90_latency_ms", "향후 10초 최악 p90 지연 예측", ["agv"])
G_WARN = Gauge("aiops_sla_warning", "SLA 위반 예측 경보 (1=발생)", ["agv"])
G_TRUTH = Gauge("wireless_fault_ground_truth", "시뮬레이터가 주입한 실제 장애 (평가용)", ["agv", "cause"])
G_BACKUP = Gauge("wireless_on_p5g_backup", "P-5G 이중화 링크 사용 중", ["agv"])
C_ACTION = Counter("aiops_actions_total", "자율 조치 횟수", ["action", "cause", "trigger"])
H_INFER = Histogram("aiops_inference_seconds", "1초 틱 처리 시간 (AGV 전체)",
                    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25))


class TickProcessor:
    def __init__(self, art, n_agv, proactive):
        self.n = n_agv
        self.eng = AIOpsEngine(art, n_agv, proactive=proactive)
        G_THR.set(self.eng.thr)
        self.ids = [f"AGV-{i:02d}" for i in range(n_agv)]

    def process(self, recs):
        """recs: 같은 ts의 AGV 레코드 n개 -> 조치 목록."""
        recs = sorted(recs, key=lambda r: r["agv_id"])
        arr = {f: np.array([r[f] for r in recs], float) for f in FEATURES}
        link = np.array([r["link"] for r in recs])
        ap = np.array([int(r["ap_id"].split("-")[1]) if r["link"] == "wifi" else -1 for r in recs])
        with H_INFER.time():
            out = self.eng.update(recs[0]["ts"], featurize(arr), arr["latency_ms"], link, ap)
        for i, r in enumerate(recs):
            a = self.ids[i]
            for f in FEATURES:
                G[f].labels(a).set(r[f])
            if not np.isnan(out["score"][i]):
                G_SCORE.labels(a).set(out["score"][i])
            if not np.isnan(out["p90"][i]):
                G_P90.labels(a).set(out["p90"][i])
            G_ALARM.labels(a).set(int(out["alarm"][i]))
            G_WARN.labels(a).set(int(out["warn"][i]))
            G_BACKUP.labels(a).set(int(r["link"] == "p5g"))
            for c in ("interference", "ap_congestion", "ap_degradation"):
                G_TRUTH.labels(a, c).set(int(r["cause"] == c))
        acts = []
        for d in out["decisions"]:
            C_ACTION.labels(d["action"], d["cause"], d["trigger"]).inc()
            acts.append(dict(ts=d["t"], agv_id=self.ids[d["agv"]], action=d["action"],
                             cause=d["cause"], trigger=d["trigger"],
                             anomaly_score=round(d["score"], 4), p90_latency_ms=round(d["p90_ms"], 1)))
        return acts


def run_kafka(args, proc):
    from kafka import KafkaConsumer, KafkaProducer
    consumer = connect(lambda: KafkaConsumer(
        TELEMETRY_TOPIC, bootstrap_servers=args.bootstrap, group_id="aiops-detector",
        auto_offset_reset="latest", value_deserializer=lambda b: json.loads(b)))
    producer = connect(lambda: KafkaProducer(
        bootstrap_servers=args.bootstrap, key_serializer=str.encode,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode()))
    pending: dict[int, dict] = {}
    print("텔레메트리 구독 시작", flush=True)
    while True:
        for msgs in consumer.poll(timeout_ms=200).values():
            for m in msgs:
                pending.setdefault(m.value["ts"], {})[m.value["agv_id"]] = m.value
        # 한 틱의 AGV 레코드가 모두 모였거나, 더 최신 틱이 2개 이상 쌓였으면 처리
        for ts in sorted(pending):
            complete = len(pending[ts]) == proc.n
            stale = sum(1 for k in pending if k > ts) >= 2
            if not (complete or stale):
                break
            recs = list(pending.pop(ts).values())
            if len(recs) != proc.n:
                continue                       # 누락 틱은 건너뜀
            for a in proc.process(recs):
                producer.send(ACTION_TOPIC, key=a["agv_id"], value=a)
                print(f"[t={a['ts']}] {a['agv_id']} {a['cause']} -> {a['action']} ({a['trigger']})", flush=True)
        producer.flush()


def run_local(args, proc):
    sim = FactorySim(args.agv, int(args.hours * 3600), seed=args.seed, with_events=True)
    print(f"로컬 모드: Kafka 없이 실행, 장애 {len(sim.events)}건 예정", flush=True)
    period = 1.0 / args.speed
    nxt = time.time()
    while sim.t < sim.T:
        r = sim.step()
        for a in proc.process(to_records(r, ts=int(r["t"]))):
            done = sim.apply_action(int(a["agv_id"].split("-")[1]), a["action"])
            print(f"[t={a['ts']}] {a['agv_id']} {a['cause']} -> {done} ({a['trigger']})", flush=True)
        nxt += period
        time.sleep(max(0.0, nxt - time.time()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--art", default="../artifacts")
    ap.add_argument("--agv", type=int, default=20)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reactive", action="store_true", help="예측 기반 선제 조치 끄기")
    ap.add_argument("--local", action="store_true", help="Kafka 없이 시뮬레이터 내장 실행")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    start_http_server(args.port)
    proc = TickProcessor(args.art, args.agv, proactive=not args.reactive)
    (run_local if args.local else run_kafka)(args, proc)


if __name__ == "__main__":
    main()
