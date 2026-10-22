"""공장 무선 텔레메트리 발행기.

시뮬레이터를 실시간으로 돌려 AGV별 1초 지표를 Kafka 'wireless.telemetry'에 발행하고,
'wireless.actions'로 들어오는 제어 명령(채널 전환, 로밍, P-5G 전환)을 시뮬레이터에 반영한다.
"""
from __future__ import annotations

import argparse
import json
import time

from simulator import FactorySim, to_records

TELEMETRY_TOPIC = "wireless.telemetry"
ACTION_TOPIC = "wireless.actions"


def connect(factory, retries=60):
    from kafka.errors import KafkaError
    for _ in range(retries):
        try:
            return factory()
        except KafkaError:
            print("Kafka 대기 중...", flush=True)
            time.sleep(2)
    raise SystemExit("Kafka에 연결하지 못했습니다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--agv", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--speed", type=float, default=1.0, help="실제 1초당 시뮬레이션 초")
    args = ap.parse_args()
    from kafka import KafkaConsumer, KafkaProducer

    producer = connect(lambda: KafkaProducer(
        bootstrap_servers=args.bootstrap, linger_ms=5,
        key_serializer=str.encode, value_serializer=lambda v: json.dumps(v).encode()))
    consumer = connect(lambda: KafkaConsumer(
        ACTION_TOPIC, bootstrap_servers=args.bootstrap, group_id="factory-simulator",
        auto_offset_reset="latest", value_deserializer=lambda b: json.loads(b)))

    sim = FactorySim(args.agv, int(args.hours * 3600), seed=args.seed, with_events=True)
    print(f"시뮬레이션 시작: AGV {args.agv}대, 장애 {len(sim.events)}건 예정", flush=True)
    period = 1.0 / args.speed
    next_tick = time.time()
    while sim.t < sim.T:
        r = sim.step()
        for rec in to_records(r, ts=int(r["t"])):
            producer.send(TELEMETRY_TOPIC, key=rec["agv_id"], value=rec)
        producer.flush()

        for msgs in consumer.poll(timeout_ms=10).values():
            for m in msgs:
                a = m.value
                done = sim.apply_action(int(a["agv_id"].split("-")[1]), a["action"])
                print(f"[t={sim.t}] 조치 반영 {a['agv_id']}: {done} (원인 {a['cause']})", flush=True)

        next_tick += period
        time.sleep(max(0.0, next_tick - time.time()))


if __name__ == "__main__":
    main()
