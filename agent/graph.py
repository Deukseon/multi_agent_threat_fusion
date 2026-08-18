"""
Phase 2 멀티에이전트 뼈대 — 그래프 조립.

핵심은 dispatch_specialists()다. 이 함수는 START 직후에 실행되는 "조건부 엣지"인데,
일반적인 조건부 엣지처럼 다음 노드 "하나"의 이름을 리턴하는 게 아니라
Send 객체의 "리스트"를 리턴한다. LangGraph는 이 리스트를 보고 "이 노드들을
전부 동시에(병렬로) 실행하고, 다 끝나면 각자의 다음 엣지를 따라가라"고 처리한다.

이게 바로 fan-out(팬아웃, 한 지점에서 여러 갈래로 흩어짐)이다. 이후 4개 전문
에이전트가 전부 coordinator로 이어지는 엣지를 갖고 있기 때문에, LangGraph는
자동으로 "이 사이클에 실행된 전문 에이전트 전원이 끝날 때까지" coordinator 실행을
미룬다 — 이게 join(합류)이다. 우리가 직접 "4명 다 끝났나 확인하는 코드"를
짤 필요가 없다는 게 Send API를 쓰는 이유.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .coordinator import coordinator
from .specialists import cv_agent, ir_anomaly_node, radar_agent, sigint_agent
from .state import GraphState

_SPECIALIST_NODES = ("radar_agent", "cv_agent", "sigint_agent", "ir_anomaly_agent")


def dispatch_specialists(state: GraphState) -> list[Send]:
    """이번 사이클의 관측 데이터(observation)에 어떤 종류의 데이터가 들어있는지 보고,
    관련된 전문 에이전트에게만 Send를 보낸다.

    예: 이번 사이클에 열이상 관측 자체가 없었다면 ir_anomaly_agent를 아예 부르지
    않는다 — "매번 전부 부른다"는 원칙이되, 애초에 그 사이클에 해당 센서 데이터가
    없으면 부를 이유가 없다는 뜻 (센서가 고장났거나 이번 폴링에 수신 실패한 경우와
    동일하게 처리).
    """
    obs = state["observation"]
    cycle_id = state["cycle_id"]
    payload = {"cycle_id": cycle_id, "observation": obs}

    sends: list[Send] = []
    if "radar_tracks" in obs:
        sends.append(Send("radar_agent", payload))
    if "cv_detections" in obs:
        sends.append(Send("cv_agent", payload))
    if "sigint_signals" in obs:
        sends.append(Send("sigint_agent", payload))
    if "thermal_context" in obs:
        sends.append(Send("ir_anomaly_agent", payload))

    if not sends:
        # 부를 전문 에이전트가 하나도 없으면(전 센서 침묵) 그래프가 여기서 그냥
        # 조용히 끝나버린다 — coordinator로 가는 엣지는 전부 "전문 에이전트가 끝나면"
        # 조건이라, 애초에 아무도 안 불렸으면 그 엣지 자체가 발동할 일이 없기 때문.
        # 그래서 이 경우엔 coordinator를 곧장 호출해서 "보고 없음"으로 안전하게
        # 마무리되게 한다 (기본값 대신 명시적으로 처리 — silent failure 방지).
        return [Send("coordinator", {"cycle_id": cycle_id, "specialist_reports": []})]

    return sends


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("radar_agent", radar_agent)
    graph.add_node("cv_agent", cv_agent)
    graph.add_node("sigint_agent", sigint_agent)
    graph.add_node("ir_anomaly_agent", ir_anomaly_node)
    graph.add_node("coordinator", coordinator)

    # START -> (fan-out) -> 관련 전문 에이전트들 (전 센서 침묵 시엔 coordinator로 직행)
    graph.add_conditional_edges(START, dispatch_specialists, list(_SPECIALIST_NODES) + ["coordinator"])

    # 전문 에이전트들 -> (join) -> coordinator
    for node_name in _SPECIALIST_NODES:
        graph.add_edge(node_name, "coordinator")

    graph.add_edge("coordinator", END)

    return graph.compile()


app = build_graph()
