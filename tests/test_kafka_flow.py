"""Kafka 메시지 흐름 테스트 (브로커 없이 메모리 내 가짜 Kafka로 실행).

producer -> wireless.telemetry -> detector -> wireless.actions -> producer(시뮬레이터 반영)
왕복이 실제로 이어지는지 확인한다.  실행: python tests/test_kafka_flow.py
"""
import collections
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TOPICS = collections.defaultdict(list)
LOCK = threading.Lock()


class _Msg:
    def __init__(self, key, value):
        self.key, self.value = key, value


class FakeProducer:
    def __init__(self, key_serializer=None, value_serializer=None, **_):
        self.ks, self.vs = key_serializer, value_serializer

    def send(self, topic, key=None, value=None):
        with LOCK:
            TOPICS[topic].append((self.ks(key), self.vs(value)))

    def flush(self):
        pass


class FakeConsumer:
    def __init__(self, topic, value_deserializer=None, **_):
        self.topic, self.vd, self.pos = topic, value_deserializer, 0

    def poll(self, timeout_ms=0):
        with LOCK:
            new = TOPICS[self.topic][self.pos:]
            self.pos += len(new)
        if not new:
            time.sleep(timeout_ms / 1000 / 4)
            return {}
        return {0: [_Msg(k, self.vd(v)) for k, v in new]}


fake = types.ModuleType("kafka")
fake.KafkaProducer, fake.KafkaConsumer = FakeProducer, FakeConsumer
errors = types.ModuleType("kafka.errors")
errors.KafkaError = Exception
sys.modules["kafka"], sys.modules["kafka.errors"] = fake, errors

import detector_service  # noqa: E402
import producer  # noqa: E402


def main():
    art = os.path.join(ROOT, "artifacts")
    proc = detector_service.TickProcessor(art, n_agv=20, proactive=True)
    args = types.SimpleNamespace(bootstrap="fake")
    threading.Thread(target=detector_service.run_kafka, args=(args, proc), daemon=True).start()

    applied = []
    orig = producer.FactorySim.apply_action

    def spy(self, agv, action):
        done = orig(self, agv, action)
        applied.append((self.t, agv, done))
        return done

    producer.FactorySim.apply_action = spy
    # 장애가 생기는 30분 구간을 빠르게 재생 (seed 7은 서비스 기본 시나리오)
    sys.argv = ["producer.py", "--hours", "0.5", "--speed", "400", "--seed", "7"]
    producer.main()
    time.sleep(2)

    n_tel = len(TOPICS[producer.TELEMETRY_TOPIC])
    n_act = len(TOPICS[producer.ACTION_TOPIC])
    print(f"telemetry 메시지 {n_tel}건, action 메시지 {n_act}건, 시뮬레이터 반영 {len(applied)}건")
    print("조치 예시:", applied[:5])
    assert n_tel == 20 * 1800, "모든 틱의 텔레메트리가 발행돼야 함"
    assert n_act > 0 and len(applied) > 0, "조치가 발행되고 시뮬레이터에 반영돼야 함"
    print("OK")


if __name__ == "__main__":
    main()
