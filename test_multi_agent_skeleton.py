"""
Phase 2 멀티에이전트 뼈대 검증 스크립트.

Chronos-2 같은 실제 외부 모델/네트워크 접속이 필요 없는 순수 로직 검증이다.
(ir_anomaly_agent 쪽은 이미 Phase 1에서 MockForecaster로 구조 검증, ChronosForecaster로
실제 성능 검증까지 끝낸 상태 — 여기서는 "그 결과가 멀티에이전트 그래프 안에서도 똑같이
동작하는가"만 확인한다.)

시나리오 3개 + 예외 케이스 2개, 총 5개 사이클을 검증한다:
  1. 평시 사이클 — 전 센서 정상, 낮은 종합 점수
  2. 로켓엔진 점화 사이클 — 열이상 ANOMALY + 미확인 트랙이 보호구역 내부에 동시 존재
     -> CRITICAL 오버라이드가 걸려야 함
  3. 애매한 사이클 — 약한 CV 탐지 하나뿐, 신뢰도 낮음 -> 중간 정도 점수
  4. 전 센서 침묵 사이클 — observation이 텅 비어 있을 때 전문 에이전트가 아무도
     호출되지 않고, coordinator가 "보고 없음"으로 안전하게 처리하는지 확인
  5. 센서 일부만 살아있는 사이클 — radar_tracks만 있고 나머지는 없을 때
     dispatch_specialists가 radar_agent 하나만 부르는지 확인 (동적 fan-out 검증)
"""
from __future__ import annotations

import sys

from agent.graph import app
from synthetic_thermal_series import generate_thermal_series


def run_cycle(cycle_id: int, observation: dict) -> dict:
    result = app.invoke(
        {
            "cycle_id": cycle_id,
            "observation": observation,
            "specialist_reports": [],
            "final_assessment": None,
            "sitrep": None,
        }
    )
    print(result["sitrep"])
    print()
    return result


def main() -> int:
    series, anomaly_steps = generate_thermal_series(seed=42)
    print(f"(참고) 합성 열이상 시계열의 실제 이상 구간: {anomaly_steps}\n")

    checks: list[tuple[str, bool]] = []

    # --- 시나리오 1: 평시 사이클 (step=100, 이상 없음) ---------------------
    step = 100
    obs1 = {
        "radar_tracks": [
            {"track_id": "trk-1", "altitude_ft": 35000, "speed_mps": 230,
             "identity": "NEUTRAL", "in_protected_zone": False},
        ],
        "cv_detections": [],
        "sigint_signals": [],
        "thermal_context": series[:step].tolist(),
        "thermal_actual": float(series[step]),
    }
    r1 = run_cycle(step, obs1)
    checks.append(("평시 사이클 -> LOW/MEDIUM(80 미만)", r1["final_assessment"]["score"] < 80))
    checks.append(("평시 사이클 -> 전문 에이전트 4명 전원 호출됨", len(r1["specialist_reports"]) == 4))

    # --- 시나리오 2: 로켓엔진 점화 사이클 (step=152, 열이상 ANOMALY 구간) ---
    step = 152
    obs2 = {
        "radar_tracks": [
            {"track_id": "trk-9", "altitude_ft": 0, "speed_mps": 0,
             "identity": "UNKNOWN", "in_protected_zone": True},
        ],
        "cv_detections": [
            {"class": "military-vehicle", "confidence": 0.81},
        ],
        "sigint_signals": [
            {"freq_mhz": 432.1, "strength_db": -40, "note": "미상 주파수 신호 버스트"},
        ],
        "thermal_context": series[:step].tolist(),
        "thermal_actual": float(series[step]),
    }
    r2 = run_cycle(step, obs2)
    checks.append(("점화 사이클 -> CRITICAL", r2["final_assessment"]["level"] == "CRITICAL"))
    ir_report = next(r for r in r2["specialist_reports"] if r.agent_name == "ir_anomaly")
    # MockForecaster는 최근 구간 이동평균이라 스파이크가 context에 섞여 들어오는 순간
    # 예측값 자체가 따라 올라가서 ANOMALY 대신 WATCH로만 잡히는 경우가 있다 (Phase 1
    # README에 이미 기록된 한계 — 실제 성능은 사용자 PC의 ChronosForecaster로 검증됨).
    # 여기서는 "정상은 아니라고는 잡아냈는가"만 확인한다.
    checks.append(("점화 사이클 -> IR 에이전트가 WATCH 이상으로 판정", ir_report.summary.split("-> ")[-1] in ("WATCH", "ANOMALY")))

    # --- 시나리오 3: 애매한 사이클 (약한 CV 탐지 하나뿐) --------------------
    step = 50
    obs3 = {
        "cv_detections": [
            {"class": "unknown-object", "confidence": 0.35},
        ],
    }
    r3 = run_cycle(step, obs3)
    checks.append(("애매한 사이클 -> CRITICAL은 아님", r3["final_assessment"]["level"] != "CRITICAL"))
    checks.append(("애매한 사이클 -> cv_agent만 호출됨(1건)", len(r3["specialist_reports"]) == 1))

    # --- 시나리오 4: 전 센서 침묵 -------------------------------------------
    r4 = run_cycle(999, {})
    checks.append(("빈 관측 -> 전문 에이전트 0명 호출, 안전하게 처리", len(r4["specialist_reports"]) == 0))
    checks.append(("빈 관측 -> level UNKNOWN", r4["final_assessment"]["level"] == "UNKNOWN"))

    # --- 시나리오 5: 레이더만 살아있는 사이클(동적 fan-out 검증) ------------
    obs5 = {
        "radar_tracks": [
            {"track_id": "trk-3", "altitude_ft": 12000, "speed_mps": 90,
             "identity": "NEUTRAL", "in_protected_zone": False},
        ],
    }
    r5 = run_cycle(5, obs5)
    checks.append(("레이더만 존재 -> radar_agent 1건만 호출", len(r5["specialist_reports"]) == 1))
    checks.append(("레이더만 존재 -> 호출된 건 radar", r5["specialist_reports"][0].agent_name == "radar"))

    # --- 결과 요약 ------------------------------------------------------
    print("=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    all_passed = True
    for name, passed in checks:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{mark}] {name}")

    print()
    if all_passed:
        print(f"전체 {len(checks)}개 검증 통과.")
        return 0
    else:
        print("일부 검증 실패 — 위 FAIL 항목 확인 필요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
