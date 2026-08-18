"""
피아식별(IFF) 간이 판정 — sitrep_fusion_agent의 fusion/identification.py를
그대로 가져옴. FRIEND/HOSTILE은 공개 데이터만으로 확정할 수 없으므로
NEUTRAL/UNKNOWN만 반환한다 (radar_agent가 기대하는 identity 값과 정확히 일치).
"""
from dataclasses import dataclass


@dataclass
class Identification:
    identity: str        # NEUTRAL / UNKNOWN (radar_agent가 UNKNOWN 여부만 확인함)
    basis: str


def classify_identity(label: str | None) -> Identification:
    if label and label != "UNKNOWN":
        return Identification(identity="NEUTRAL", basis=f"호출부호 확인됨({label}), 민간 교통으로 추정")
    return Identification(identity="UNKNOWN", basis="호출부호 미확인, IFF 질의응답 데이터 없음")
