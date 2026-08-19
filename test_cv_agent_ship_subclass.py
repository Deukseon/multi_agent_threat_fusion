"""
agent/specialists.py의 cv_agent() -- ⑫ 희소 군사 자산 세분류(선박 v1) 통합 검증 스크립트.

[샌드박스 검증 범위] cv_agent()는 순수 함수(observation dict -> SpecialistReport)라
langgraph/ultralytics/torch 없이도 손으로 만든 observation만으로 완전히 검증 가능하다
(test_observation.py와 같은 원칙). 여기서는 "cv_detections 안의 sub_class 필드에 따라
점수가 올바르게 갈리는가"만 확인하고, 실제 YOLO+세분류기 파이프라인이 sub_class를
정확히 채우는지는(FinetunedShipSubclassifier) 사용자 PC에서 실 이미지로만 검증 가능하다
(README "다음 할 일" 참고).

test_multi_agent_skeleton.py와 겹치지 않는 이유: 그 파일은 agent.graph.app(langgraph
컴파일된 그래프) 전체를 도는데, 이 세션엔 langgraph가 없어 import 자체가 안 된다. 이
파일은 cv_agent() 하나만 직접 호출해서 langgraph 없이도 새 스코어링 로직을 검증한다.
"""
from __future__ import annotations

import sys

from agent.specialists import cv_agent


def run(detections: list[dict]) -> tuple[float, str]:
    state = {"observation": {"cv_detections": detections}, "cycle_id": 0}
    result = cv_agent(state)
    report = result["specialist_reports"][0]
    return report.threat_score, report.summary


def main() -> int:
    checks: list[tuple[str, bool]] = []

    # 1) 군함형(warcraft) sub_class -> base=85 (기존 ship 기본값 60보다 높음)
    score, _ = run([{"class": "ship", "confidence": 0.9, "sub_class": "warcraft"}])
    checks.append(("군함형(warcraft, conf 0.9) -> 85*0.9=76.5", abs(score - 76.5) < 1e-6))

    # 2) 상선형(merchant_ship) sub_class -> base=15 (CV_LOW_CONCERN_CLASSES와 동급으로 낮춤)
    score, _ = run([{"class": "ship", "confidence": 0.9, "sub_class": "merchant_ship"}])
    checks.append(("상선형(merchant_ship, conf 0.9) -> 15*0.9=13.5", abs(score - 13.5) < 1e-6))

    # 3) sub_class가 None(세분류 비활성/실패) -> 기존 CV_HIGH_CONCERN_CLASSES 규칙(base=60)으로 폴백
    score, _ = run([{"class": "ship", "confidence": 0.9, "sub_class": None}])
    checks.append(("sub_class=None -> 기존 규칙 폴백(60*0.9=54.0)", abs(score - 54.0) < 1e-6))

    # 4) sub_class 키 자체가 없는 구버전 observation dict -> 동일하게 안전 폴백
    score, _ = run([{"class": "ship", "confidence": 0.9}])
    checks.append(("sub_class 키 자체 없음 -> 폴백(60*0.9=54.0)", abs(score - 54.0) < 1e-6))

    # 5) unknown_ship_type -> 군함형/민간형 어느 집합에도 없으므로 기존 규칙 폴백
    score, _ = run([{"class": "ship", "confidence": 0.8, "sub_class": "unknown_ship_type"}])
    checks.append(("unknown_ship_type -> 폴백(60*0.8=48.0)", abs(score - 48.0) < 1e-6))

    # 6) ship이 아닌 클래스는 sub_class 값이 우연히 섞여 있어도 무시 (실제 파이프라인에서는
    #    sub_class가 ship 탐지에만 채워지지만, 방어적으로도 안전한지 확인)
    score, _ = run([{"class": "plane", "confidence": 0.7, "sub_class": "warcraft"}])
    checks.append(("ship이 아닌 클래스는 sub_class 무시(60*0.7=42.0)", abs(score - 42.0) < 1e-6))

    # 7) 여러 탐지 중 군함형이 최고점을 내면 best_score가 그걸 반영
    score, _ = run([
        {"class": "ship", "confidence": 0.5, "sub_class": "merchant_ship"},
        {"class": "ship", "confidence": 0.6, "sub_class": "warcraft"},
    ])
    checks.append(("혼합 탐지 -> 최고점은 군함형 쪽(85*0.6=51.0)", abs(score - 51.0) < 1e-6))

    # 8) 탐지 자체가 없으면 기존과 동일하게 안전 처리(회귀 확인)
    empty_result = cv_agent({"observation": {}, "cycle_id": 0})
    empty_report = empty_result["specialist_reports"][0]
    checks.append(("탐지 없음 -> score 0.0, 크래시 없음(회귀)", empty_report.threat_score == 0.0))

    # --- 결과 요약 --------------------------------------------------------
    print("=" * 60)
    print("cv_agent() 선박 세분류(sub_class) 스코어링 검증 결과")
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
