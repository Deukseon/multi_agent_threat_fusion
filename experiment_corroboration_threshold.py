"""
[검증 실험] "독립 센서 3개 이상 -> CRITICAL" 규칙(오버라이드②)을 "2개 이상"으로
낮추면 어떤 문제가 생기는지 실측으로 확인하는 스크립트. (2026-08-18, README 3번 항목 참고)

방법: 실제 위협이 전혀 없는 "평범한 배경 소음" 관측을 대량으로 무작위 생성해서
(무작위 민간 트랙, 흔한 RF 신호, 순수 노이즈 열이상 시계열 등), 4개 전문
에이전트를 그대로 돌리고, coordinator의 오버라이드② 문턱을 3 vs 2로 바꿔가며
"실제로는 위협이 없는데 CRITICAL로 오판하는 비율"을 직접 센다.

[2026-08-18 추가] 지금까지는 `agent/specialists.py`가 기본으로 쓰는 `MockForecaster`
기준으로만 이 실험을 돌렸다. Mock은 24시간 주기 패턴을 못 따라가서 정상 패턴에도 잦은
WATCH/ANOMALY를 내는 걸로 이미 알려져 있어서(Phase 1 README), 여기서 나온 "오탐의
99.8%가 단일 오버라이드에서 나온다"는 수치가 실제 `ChronosForecaster`로도 재현되는지가
남은 검증 과제다. 환경변수 `USE_REAL_FORECASTER=1`을 켜면 `agent.specialists` 모듈의
`_ir_forecaster`를 실제 `ChronosForecaster`로 바꿔치기(monkeypatch)해서 재현할 수 있다
— `agent/specialists.py` 자체는 건드리지 않는다(운영 그래프는 계속 Mock을 씀).
[샌드박스 한계] huggingface.co가 막혀 있어 이 실제 모델 다운로드는 사용자 PC에서만
가능 — 최초 실행 시 478MB 다운로드가 걸린다.
"""
from __future__ import annotations

import os

import numpy as np

import agent.specialists as specialists_module
from agent.specialists import cv_agent, ir_anomaly_node, radar_agent, sigint_agent


def random_benign_observation(rng: np.random.Generator) -> dict:
    """실제 위협 요소를 넣지 않은, 평범한 배경 상황을 무작위로 만든다."""
    obs: dict = {}

    if rng.random() < 0.9:
        n = int(rng.integers(0, 3))
        tracks = []
        for _ in range(n):
            tracks.append(
                {
                    "track_id": f"trk-{int(rng.integers(1, 999))}",
                    "altitude_ft": int(rng.integers(500, 40000)),
                    "speed_mps": int(rng.integers(20, 300)),
                    # 실제로는 위협이 아니어도 트랜스폰더 미장착·오류 등으로
                    # UNKNOWN이 나오는 경우가 실전에서도 흔하다 (25% 가정)
                    "identity": "UNKNOWN" if rng.random() < 0.25 else "NEUTRAL",
                    "in_protected_zone": rng.random() < 0.08,
                }
            )
        obs["radar_tracks"] = tracks

    if rng.random() < 0.9:
        n = int(rng.integers(0, 3))
        classes = ["civilian-vehicle", "building", "unknown-object", "military-vehicle"]
        weights = [0.5, 0.3, 0.15, 0.05]  # military-vehicle 5%는 오탐지(오분류) 가정
        dets = []
        for _ in range(n):
            cls = rng.choice(classes, p=weights)
            dets.append({"class": cls, "confidence": float(rng.uniform(0.3, 0.95))})
        obs["cv_detections"] = dets

    if rng.random() < 0.9:
        n = int(rng.integers(0, 3))
        sigs = []
        for _ in range(n):
            # 흔한 배경 RF(방송, 통신, 민간 레이더 등) 세기 분포를 가정
            sigs.append({"freq_mhz": float(rng.uniform(100, 2000)), "strength_db": float(rng.uniform(-95, -25))})
        obs["sigint_signals"] = sigs

    if rng.random() < 0.9:
        t = np.arange(300)
        baseline = 15.0 + 3.0 * np.sin(2 * np.pi * t / 144.0)
        noise = rng.normal(0, 0.6, size=300)
        series = baseline + noise  # 이상 이벤트 주입 없음 -- 순수 정상 패턴
        step = int(rng.integers(20, 299))
        obs["thermal_context"] = series[:step].tolist()
        obs["thermal_actual"] = float(series[step])

    return obs


def collect_reports(obs: dict) -> list:
    reports = []
    payload = {"cycle_id": 0, "observation": obs}
    if "radar_tracks" in obs:
        reports.append(radar_agent(payload)["specialist_reports"][0])
    if "cv_detections" in obs:
        reports.append(cv_agent(payload)["specialist_reports"][0])
    if "sigint_signals" in obs:
        reports.append(sigint_agent(payload)["specialist_reports"][0])
    if "thermal_context" in obs:
        reports.append(ir_anomaly_node(payload)["specialist_reports"][0])
    return reports


def overridden(reports: list, min_corroborators: int) -> bool:
    single_agent = any(r.confidence >= 0.8 and r.threat_score >= 85 for r in reports)
    corroborating = sum(1 for r in reports if r.threat_score >= 50)
    return single_agent or (corroborating >= min_corroborators)


def main() -> None:
    if os.environ.get("USE_REAL_FORECASTER") == "1":
        from ir_anomaly_agent import ChronosForecaster

        print("[실제 ChronosForecaster 사용 — 최초 실행 시 amazon/chronos-2(478MB) 다운로드]")
        specialists_module._ir_forecaster = ChronosForecaster()
    else:
        print("[MockForecaster 사용 중 — 실제 ChronosForecaster로 재현하려면 "
              "환경변수 USE_REAL_FORECASTER=1로 실행 (huggingface.co 접속 필요)]")
    print()

    rng = np.random.default_rng(0)
    # 실제 ChronosForecaster는 CPU 추론이 느려서(호출당 최대 1~2초) 5000회를 그대로
    # 돌리면 시간이 많이 걸린다. 환경변수 N_TRIALS로 시행 횟수를 줄일 수 있다
    # (예: N_TRIALS=300). Mock 기준일 땐 5000회도 수초 안에 끝나므로 기본값을 유지.
    n_trials = int(os.environ.get("N_TRIALS", 5000))

    false_critical = {2: 0, 3: 0}
    corroborator_counts: list[int] = []

    for _ in range(n_trials):
        obs = random_benign_observation(rng)
        reports = collect_reports(obs)
        corroborator_counts.append(sum(1 for r in reports if r.threat_score >= 50))
        for n in (2, 3):
            if overridden(reports, n):
                false_critical[n] += 1

    print(f"시행 횟수: {n_trials} (실제 위협 없는 순수 배경 소음 시나리오)")
    print()
    print(f"오버라이드② 문턱 = 3개(현재 설계): 오탐 CRITICAL {false_critical[3]}건 "
          f"({false_critical[3] / n_trials * 100:.2f}%)")
    print(f"오버라이드② 문턱 = 2개(가정한 변경): 오탐 CRITICAL {false_critical[2]}건 "
          f"({false_critical[2] / n_trials * 100:.2f}%)")
    print()
    ratio = (false_critical[2] / max(false_critical[3], 1))
    print(f"배율: 문턱을 2개로 낮추면 오탐률이 약 {ratio:.1f}배 증가")
    print()

    # 30초 간격 지속 감시라고 가정했을 때 하루 오탐 횟수 환산
    cycles_per_day = 24 * 60 * 60 // 30
    for n in (2, 3):
        rate = false_critical[n] / n_trials
        print(f"문턱={n}개 기준, 30초 폴링으로 하루 감시 시 예상 오탐 CRITICAL: "
              f"약 {rate * cycles_per_day:.1f}건/일")


if __name__ == "__main__":
    main()
