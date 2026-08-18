"""
상황실장(coordinator) 노드 — 4개 전문 에이전트의 보고를 종합해서
최종 위협 평가 + SITREP 초안을 만든다.

지금은 규칙 기반(가중평균 + CRITICAL 강제 오버라이드)으로 종합한다. 여기서 만드는
state["sitrep"]은 일부러 템플릿 텍스트로만 남겨둔다 — "여러 에이전트의 판단을 하나로
합치는 로직 자체"를 검증하는 게 이 노드의 목적이라, LLM 호출까지 여기 붙이면 무엇을
검증하는지 흐려지기 때문. Claude API 기반 자연어 브리핑은 이 노드 다음에 실행되는
별도 노드(`agent/brief.py`의 `generate_sitrep`)로 분리했다 — sitrep_fusion_agent의
generate_brief와 같은 패턴, "종합 판단"과 "그 판단을 사람이 읽기 좋게 다듬는 것"을
서로 다른 책임으로 나눈 것.
"""
from __future__ import annotations

from .state import GraphState, SpecialistReport

# sitrep_fusion_agent의 threat_scoring.py와 동일한 임계값 체계를 재사용
# (CRITICAL 80+ / HIGH 60+ / MEDIUM 30+ / LOW)
_LEVEL_THRESHOLDS = (
    (80, "CRITICAL"),
    (60, "HIGH"),
    (30, "MEDIUM"),
)


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "LOW"


def coordinator(state: GraphState) -> dict:
    reports: list[SpecialistReport] = state["specialist_reports"]

    if not reports:
        return {
            "final_assessment": {"level": "UNKNOWN", "score": 0.0, "critical_override": False},
            "sitrep": f"[사이클 {state['cycle_id']}] 이번 사이클에 제출된 전문 에이전트 보고가 없습니다.",
        }

    # 신뢰도 가중평균 — 자신 없는 보고(낮은 confidence)는 최종 점수에 덜 반영되게 함
    weight_total = sum(r.confidence for r in reports) or 1.0
    weighted_avg = sum(r.threat_score * r.confidence for r in reports) / weight_total
    max_score = max(r.threat_score for r in reports)

    # 오버라이드 규칙 ①: sitrep_fusion_agent의 CRITICAL 오버라이드와 같은 사상 —
    # 신뢰도 높은(>=0.8) 전문 에이전트 "단독"이 85점 이상을 보고하면, 다른 에이전트의
    # 낮은 점수에 희석되지 않고 CRITICAL을 강제한다.
    single_agent_override = any(r.confidence >= 0.8 and r.threat_score >= 85 for r in reports)

    # 오버라이드 규칙 ② (Phase 2에서 새로 발견한 케이스): 서로 "독립적인" 센서 3개
    # 이상이 각자 중간 수준(50점 이상) 위협을 동시에 보고하면, 단일 센서의 극단값 하나
    # 보다 오히려 더 신뢰할 수 있는 신호다. 실제 실험에서 이 케이스가 나왔다 — 열이상
    # WATCH + 레이더(미확인·보호구역 내부) + SIGINT 이상신호가 동시에 떴는데, 어느 하나도
    # "confidence>=0.8 and score>=85"를 단독으로 못 채워서 가중평균만으로는 HIGH(64.7점)에
    # 그쳤다. 서로 다른 종류의 센서가 독립적으로 같은 결론에 도달했다는 사실 자체가
    # 우연히 하나가 튄 것보다 훨씬 강한 증거이므로(다중 센서 융합의 핵심 가치), 별도
    # 규칙으로 보완했다.
    corroborating = sum(1 for r in reports if r.threat_score >= 50)
    corroboration_override = corroborating >= 3

    critical_override = single_agent_override or corroboration_override
    final_score = 95.0 if critical_override else max(weighted_avg, max_score * 0.7)
    level = _score_to_level(final_score)

    lines = [f"[사이클 {state['cycle_id']}] 종합 위협등급: {level} (점수 {final_score:.1f})"]
    if single_agent_override:
        lines.append("  ※ CRITICAL 오버라이드 규칙① 발동 (신뢰도 높은 단일 전문 에이전트의 고위협 보고)")
    if corroboration_override:
        lines.append(f"  ※ CRITICAL 오버라이드 규칙② 발동 (독립 센서 {corroborating}개가 동시에 중간 이상 위협 보고 — 교차 corroboration)")
    for r in sorted(reports, key=lambda r: -r.threat_score):
        lines.append(f"  - [{r.agent_name}] {r.summary} (점수 {r.threat_score:.0f}, 신뢰도 {r.confidence:.2f})")

    return {
        "final_assessment": {"level": level, "score": final_score, "critical_override": critical_override},
        "sitrep": "\n".join(lines),
    }
