"""
자연어 SITREP 생성 노드 — sitrep_fusion_agent의 generate_brief 패턴을 재사용.

coordinator는 "규칙 기반으로 여러 에이전트의 판단을 하나로 합치는 로직"만 검증하는
게 목적이라 일부러 템플릿 텍스트(state["sitrep"])만 만들고 끝냈다 (README "다음 할
일" 8번 참고). 이 모듈은 그 다음 단계 — coordinator의 규칙 기반 요약을 원재료 삼아
Claude API로 지휘관 브리핑처럼 읽히는 자연어 문장을 만든다.

설계 원칙 (sitrep_fusion_agent의 generate_brief와 동일):
  - LLM 호출은 반드시 실패할 수 있다고 가정한다(API 키 미설정, 네트워크 문제, 요금
    한도 등). 실패해도 그래프 전체가 죽으면 안 되므로, 실패 시 규칙 기반 원시 요약으로
    안전하게 폴백한다 — "LLM이 없으면 브리핑이 없다"가 아니라 "LLM이 없으면 조금 덜
    다듬어진 브리핑이 나온다"가 되도록.
  - [샌드박스 검증 범위] 이 세션엔 실제 ANTHROPIC_API_KEY가 없어서 실제 LLM 호출
    자체는 검증 못 했다 — 대신 "API 키가 없을 때 폴백 경로가 안전하게 동작하는가"는
    검증했다. 실제 자연어 브리핑 품질은 사용자 PC(.env에 키 설정됨)에서 확인 필요.
"""
from __future__ import annotations

import logging

from .state import GraphState

logger = logging.getLogger(__name__)


def _raw_summary(state: GraphState) -> str:
    reports = state["specialist_reports"]
    assessment = state["final_assessment"] or {"level": "UNKNOWN", "score": 0.0, "critical_override": False}
    cycle_id = state["cycle_id"]

    if not reports:
        return f"[사이클 {cycle_id}] 감시 구역 내 특이사항 없음 — 이번 사이클에 보고한 전문 에이전트가 없습니다."

    lines = [f"[사이클 {cycle_id}] 종합 등급: {assessment['level']} (점수 {assessment['score']:.1f})"]
    if assessment.get("critical_override"):
        lines.append("[CRITICAL 오버라이드 발동 — 단일 고신뢰 보고 또는 다중 센서 교차 corroboration]")
    for r in sorted(reports, key=lambda r: -r.threat_score):
        lines.append(f"- {r.agent_name} 담당관: {r.summary} (위협점수 {r.threat_score:.0f}, 신뢰도 {r.confidence:.2f})")
    return "\n".join(lines)


def generate_sitrep(state: GraphState) -> dict:
    """coordinator 이후 실행. 규칙 기반 요약을 Claude API에 넘겨 자연어 브리핑으로 다듬는다."""
    raw_summary = _raw_summary(state)

    # 전문 에이전트가 하나도 없는 사이클(전 센서 침묵)은 LLM까지 부를 필요 없이
    # 원시 요약 그대로가 이미 브리핑이다.
    if not state["specialist_reports"]:
        return {"natural_language_brief": raw_summary}

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=500)
        prompt = (
            "당신은 방공 관제 상황실 분석관입니다. 아래는 서로 독립적인 전문 에이전트"
            "(레이더/영상/신호정보/적외선 담당관)가 이번 사이클에 각자 제출한 위협 평가와, "
            "이를 종합한 최종 등급입니다. 지휘관에게 보고할 간결한 한국어 브리핑을 "
            "작성하세요. 종합 등급을 먼저 언급하고, 가장 근거가 되는 보고부터 우선순위대로 "
            "설명한 뒤, 권고 조치를 포함하세요.\n\n"
            f"{raw_summary}"
        )
        response = llm.invoke(prompt)
        return {"natural_language_brief": response.content}
    except Exception as e:
        # API 키 미설정·네트워크 실패 등으로 죽어도 파이프라인은 안전하게 계속 진행
        logger.error("LLM 브리핑 생성 실패, 규칙 기반 요약으로 폴백: %s", e)
        return {"natural_language_brief": f"[LLM 호출 실패 - 원시 데이터로 대체]\n{raw_summary}"}
