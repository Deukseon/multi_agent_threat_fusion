"""
Phase 2 멀티에이전트 뼈대 — 공유 상태(State) 정의.

설계 방향 (프로젝트_진행_스케줄.md 2026-08-14 확정 사항):
  - ReAct식 슈퍼바이저(매 턴 "다음에 누굴 부를지" 판단)가 아니라,
    매 사이클마다 관련된 모든 전문 에이전트를 동시에 병렬 실행(fan-out)한 뒤
    그 결과를 한 곳(coordinator)에서 종합하는 구조.
  - 위협 융합은 애초에 "이번에 뭘 볼지 골라야 하는 문제"가 아니라
    "매번 가진 센서를 전부 동시에 봐야 하는 문제"이기 때문.

비유: coordinator는 "상황실장", 각 전문 에이전트(radar/cv/sigint/ir_anomaly)는
"각 분야 담당관". 상황실장이 매 사이클마다 담당관 전원에게 동시에 보고를 요청하고
(fan-out), 전원의 보고가 다 들어오면(join) 그걸 종합해서 하나의 SITREP을 쓴다.
"한 명씩 순서대로 물어보는" 방식이 아니다.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict


@dataclass
class SpecialistReport:
    """전문 에이전트 1명이 매 사이클마다 제출하는 보고서 1건."""

    agent_name: str          # "radar" / "cv" / "sigint" / "ir_anomaly"
    threat_score: float      # 0~100, 이 전문 에이전트 "관점"에서 본 위협 점수
    confidence: float        # 0~1, 이 판단 자체에 대한 자신감 (증거가 약하면 낮게)
    summary: str              # 사람이 바로 읽을 수 있는 한 줄 근거
    raw: dict = field(default_factory=dict)  # 원본 데이터 (감사/디버깅용, SITREP엔 안 씀)


class GraphState(TypedDict):
    """그래프 전체가 공유하는 상태.

    specialist_reports는 Annotated[..., operator.add]로 선언되어 있다.
    이게 핵심 트릭: 여러 전문 에이전트가 "동시에" 각자 SpecialistReport를 하나씩
    돌려주는데, 기본값으로는 나중에 도착한 값이 먼저 것을 덮어써버린다(값 4개 중
    3개가 유실됨). operator.add로 지정하면 "덮어쓰지 말고 리스트에 이어붙여라"는
    뜻이 되어, 4명의 보고가 전부 살아남아 하나의 리스트에 쌓인다.
    (비유: 여러 사람이 같은 칠판에 답을 쓰면 마지막 사람 것만 남지만,
    각자 포스트잇에 써서 같은 상자에 넣으면 전원의 답이 다 보존되는 것과 같다.)
    """

    cycle_id: int
    observation: dict[str, Any]                              # 이번 사이클의 원본 관측 묶음
    specialist_reports: Annotated[list[SpecialistReport], operator.add]
    final_assessment: Optional[dict]
    sitrep: Optional[str]                                     # coordinator가 만드는 규칙 기반 요약(감사·폴백용)
    natural_language_brief: Optional[str]                     # generate_sitrep이 Claude API로 만드는 자연어 브리핑


class SpecialistInput(TypedDict):
    """Send()로 각 전문 에이전트 노드에 전달되는 입력.

    그래프 전체 상태(GraphState)를 통째로 넘기지 않고 필요한 키만 골라 넘긴다.
    실제 다중 에이전트 시스템에서도 "각 담당관에게 필요한 정보만 브리핑하고,
    상관없는 다른 담당관의 원본 데이터까지 다 보여주지 않는다"는 원칙과 같다.
    """

    cycle_id: int
    observation: dict[str, Any]
