"""
[검증 실험] "독립 센서 3개 이상 -> CRITICAL" 규칙(오버라이드②)을 "2개 이상"으로
낮추면 어떤 문제가 생기는지 실측으로 확인하는 스크립트. (2026-08-18, README 3번 항목 참고)

방법: 실제 위협이 전혀 없는 "평범한 배경 소음" 관측을 대량으로 무작위 생성해서
(무작위 민간 트랙, 흔한 RF 신호, 순수 노이즈 열이상 시계열 등), 4개 전문
에이전트를 그대로 돌리고, coordinator의 오버라이드② 문턱을 3 vs 2로 바꿔가며
"실제로는 위협이 없는데 CRITICAL로 오판하는 비율"을 직접 센다.
"""
from __future__ import annotations

import numpy as np

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
    rng = np.random.default_rng(0)
    n_trials = 5000

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
