"""
전문 에이전트(specialist) 4종의 노드 함수.

지금 단계(Phase 2 뼈대 설계)의 목적은 "병렬 fan-out/join 구조 자체가 제대로
동작하는가"를 검증하는 것이다. 그래서 radar/cv/sigint 세 에이전트는 아직
sitrep_fusion_agent(이전 프로젝트)의 실제 API 연동 코드를 그대로 가져오지 않고,
같은 판단 로직을 단순화한 목(mock) 버전으로 둔다 — 뼈대가 검증되면 다음 단계에서
실제 연동으로 교체할 자리(플레이스홀더)라는 뜻.

반면 ir_anomaly 에이전트는 다르다. 이건 Phase 1에서 이미 별도로 프로토타입·검증까지
끝낸 진짜 로직(ir_anomaly_agent.detect_anomaly)을 그대로 갖다 쓴다. 즉 이 파일이
Phase 1 산출물과 Phase 2 뼈대를 실제로 이어붙이는 지점이다.
"""
from __future__ import annotations

import numpy as np

from ir_anomaly_agent import MockForecaster, detect_anomaly

from .state import GraphState, SpecialistInput, SpecialistReport

# IR 이상탐지용 예측기는 매 사이클 새로 만들지 않고 모듈 로드 시 한 번만 생성해서 재사용.
# (실제 운영에서는 MockForecaster 대신 ChronosForecaster로 교체 — README 참고)
_ir_forecaster = MockForecaster()


def radar_agent(state: SpecialistInput) -> dict:
    """레이더/항적 담당관. sitrep_fusion_agent의 geofence+identification 판단을
    단순화해서 재현: 미확인 식별 + 보호구역 내부 + 고속이면 점수가 올라간다."""
    tracks = state["observation"].get("radar_tracks", [])
    if not tracks:
        report = SpecialistReport("radar", 0.0, 0.3, "탐지된 트랙 없음")
        return {"specialist_reports": [report]}

    best_score = 0.0
    reasons: list[str] = []
    for tr in tracks:
        score = 0.0
        if tr.get("identity") == "UNKNOWN":
            score += 30
            reasons.append(f"{tr['track_id']} 미확인 식별")
        if tr.get("in_protected_zone"):
            score += 40
            reasons.append(f"{tr['track_id']} 보호구역 내부")
        if tr.get("speed_mps", 0) > 200:
            score += 10
            reasons.append(f"{tr['track_id']} 고속({tr['speed_mps']}m/s)")
        best_score = max(best_score, score)

    summary = "; ".join(reasons) if reasons else "정상 범위 트랙만 존재"
    report = SpecialistReport("radar", best_score, 0.7, summary, {"tracks": tracks})
    return {"specialist_reports": [report]}


def cv_agent(state: SpecialistInput) -> dict:
    """EO/IR 영상 담당관. sitrep_fusion_agent Phase 6의 YOLO26-OBB 탐지 결과를
    받는다고 가정 — 여기선 클래스·신뢰도만 보고 단순 점수화한다."""
    detections = state["observation"].get("cv_detections", [])
    if not detections:
        report = SpecialistReport("cv", 0.0, 0.3, "영상 내 특이 탐지 없음")
        return {"specialist_reports": [report]}

    military_classes = {"military-vehicle", "warship", "missile-launcher"}
    best_score = 0.0
    reasons: list[str] = []
    for det in detections:
        cls = det.get("class", "unknown-object")
        conf = det.get("confidence", 0.0)
        base = 60 if cls in military_classes else 20
        score = base * conf
        reasons.append(f"{cls}(신뢰도 {conf:.2f})")
        best_score = max(best_score, score)

    confidence = max((d.get("confidence", 0.0) for d in detections), default=0.3)
    report = SpecialistReport("cv", best_score, confidence, "; ".join(reasons), {"detections": detections})
    return {"specialist_reports": [report]}


def sigint_agent(state: SpecialistInput) -> dict:
    """SIGINT(신호정보) 담당관.

    [2026-08-18 수정] 원래는 "신호가 세면 이상"이라는 규칙이었는데, 실측 결과
    실제 위협이 전혀 없는 배경 소음(방송·통신 등 흔한 강신호)에서도 47.3% 확률로
    50점 이상을 오판한다는 게 드러났다 (experiment_corroboration_threshold.py).
    세기(strength_db)는 "신호가 있다는 것"만 알려줄 뿐 "위험하다는 것"은 알려주지
    않는다 — 강한 신호 자체는 방송탑·통신기지국·민간 레이더 등 어디에나 있다.

    그래서 판단 기준을 뒤집었다: 핵심 근거는 `note`(신호처리 단계에서 이미
    "이상 신호"로 플래그된 경우 — 예: 미상 주파수, 암호화 버스트, 재밍 패턴)이고,
    세기는 "이상으로 플래그된 신호가 얼마나 강한가"를 보정하는 보조 지표로만 쓴다.
    note가 없으면 아무리 세도 최대 25점까지만 올라가서(다른 에이전트와 corroboration
    문턱인 50점을 절대 단독으로 못 넘김), 오버라이드②가 "시끄러운 배경 신호"에
    흔들리지 않게 만들었다.
    """
    signals = state["observation"].get("sigint_signals", [])
    if not signals:
        report = SpecialistReport("sigint", 0.0, 0.3, "특이 신호 없음")
        return {"specialist_reports": [report]}

    best_score = 0.0
    best_confidence = 0.3
    reasons: list[str] = []
    unflagged_present = False
    for sig in signals:
        strength = sig.get("strength_db", -100)
        has_note = bool(sig.get("note"))
        # 세기 자체는 약한 보조 근거 -- 최대 25점 (단독으로는 corroboration 문턱인
        # 50점을 절대 못 넘기게 상한을 낮게 잡음)
        strength_component = min(25.0, max(0.0, (strength + 100) * 0.375))

        if has_note:
            # 이상 신호로 실제 플래그된 경우에만 핵심 점수(60점)를 부여, 세기는 가산치
            score = min(100.0, 60.0 + strength_component)
            confidence = 0.75
            reasons.append(f"{sig['freq_mhz']}MHz: {sig['note']} (세기 {strength:.0f}dB)")
        else:
            score = strength_component
            confidence = 0.4
            unflagged_present = True

        if score > best_score:
            best_score = score
            best_confidence = confidence

    if not reasons:
        summary = (
            "강한 신호는 있으나 이상 신호로 플래그되지 않음(배경 RF로 판단)"
            if unflagged_present
            else "특이 신호 없음"
        )
    else:
        summary = "; ".join(reasons)

    report = SpecialistReport("sigint", best_score, best_confidence, summary, {"signals": signals})
    return {"specialist_reports": [report]}


def ir_anomaly_node(state: SpecialistInput) -> dict:
    """위성 IR 조기경보 담당관. Phase 1에서 검증한 forecast-then-compare
    이상탐지 로직(ir_anomaly_agent.detect_anomaly)을 그대로 호출한다."""
    obs = state["observation"]
    if "thermal_context" not in obs:
        report = SpecialistReport("ir_anomaly", 0.0, 0.3, "열이상 관측 없음(이번 사이클 대상 아님)")
        return {"specialist_reports": [report]}

    context = np.asarray(obs["thermal_context"], dtype=float)
    actual = float(obs["thermal_actual"])
    result = detect_anomaly(_ir_forecaster, context, actual, step=state["cycle_id"])

    score_by_severity = {"NORMAL": 0.0, "WATCH": 55.0, "ANOMALY": 90.0}
    confidence_by_severity = {"NORMAL": 0.4, "WATCH": 0.6, "ANOMALY": 0.85}

    summary = (
        f"실측 {result.actual:.1f} vs 예측중앙값 {result.predicted_median:.1f} "
        f"(90%구간 [{result.lower_bound:.1f}, {result.upper_bound:.1f}]) -> {result.severity}"
    )
    report = SpecialistReport(
        "ir_anomaly",
        score_by_severity[result.severity],
        confidence_by_severity[result.severity],
        summary,
        {"severity": result.severity, "actual": result.actual, "predicted_median": result.predicted_median},
    )
    return {"specialist_reports": [report]}
