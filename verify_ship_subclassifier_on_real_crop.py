"""
[PC 전용 검증 스크립트] "탐지→크롭→세분류" 파이프라인 중 세분류기(및 크롭) 부분만
YOLO 탐지와 분리해서 직접 검증한다.

배경: verify_ship_subtype_integration.py로 종단 간(end-to-end) 검증을 시도했는데,
YOLO(DOTA-v1.0으로 학습됨)가 HRSC2016 이미지(100000641.bmp, 자동차운반선)에서 conf
0.01까지 낮춰도 "ship"을 전혀 못 잡는 걸 확인했다(대신 신뢰도 0.015짜리 노이즈성
"plane" 오탐 2건만 나옴). 이건 오늘 새로 만든 통합 코드의 버그가 아니라, DOTA
스타일(넓은 항공/위성 타일)로 학습된 YOLO가 HRSC2016 스타일(배 위주로 크롭된 이미지)
에는 그대로 적용이 안 되는 "도메인 갭" 문제로, Phase 2.5(2026-08-18)에서 실제 DOTA
검증 이미지로도 신뢰도가 0.13~0.33으로 낮게 나왔던 것과 같은 계열의 이미 알려진
한계다. YOLO 탐지 자체는 이 스크립트의 검증 범위가 아니다.

그래서 이 스크립트는 "YOLO가 ship을 찾아준다면 그 다음(크롭→세분류)이 제대로
동작하는가"만 따로 떼어서 검증한다 -- HRSC2016 실제 어노테이션에서 직접 사각형
좌표를 읽어 crop_rotated_box()로 크롭하고(2026-08-18에 이미 실제 데이터로 시각
검증된 로직), 그 크롭을 detect_objects()의 내부 로직과 동일한 방식으로
FinetunedShipSubclassifier.classify()에 직접 넣어서 실제로 sub_class가 나오는지,
그리고 실제 학습된 모델(mock 폴백이 아니라)이 쓰이는지 확인한다.

사용법 (C:\\dev\\multi_agent_threat_fusion 에서, 가상환경 활성화 후):
    python verify_ship_subclassifier_on_real_crop.py
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_IMAGE_PATH = os.path.join(
    "C:\\", "dev", "ship_subclassification", "data", "raw", "HRSC2016",
    "FullDataSet", "AllImages", "100000641.bmp",
)
_ANNOTATION_PATH = os.path.join(
    "C:\\", "dev", "ship_subclassification", "data", "raw", "HRSC2016",
    "FullDataSet", "Annotations", "100000641.xml",
)


def parse_hrsc_objects(xml_path: str):
    """HRSC2016 어노테이션 XML에서 선박 인스턴스별 (Class_ID, cx, cy, w, h, ang_rad)을 뽑는다."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.iter("HRSC_Object"):
        class_id = obj.findtext("Class_ID")
        cx = float(obj.findtext("mbox_cx"))
        cy = float(obj.findtext("mbox_cy"))
        w = float(obj.findtext("mbox_w"))
        h = float(obj.findtext("mbox_h"))
        ang = float(obj.findtext("mbox_ang"))
        objects.append((class_id, cx, cy, w, h, ang))
    return objects


def main() -> int:
    import cv2

    from crop_utils import crop_rotated_box
    from data_sources.cv_detection import _get_ship_subclassifier

    if not os.path.exists(_IMAGE_PATH) or not os.path.exists(_ANNOTATION_PATH):
        print(f"[오류] 이미지 또는 어노테이션이 없습니다:\n  {_IMAGE_PATH}\n  {_ANNOTATION_PATH}")
        return 1

    image_bgr = cv2.imread(_IMAGE_PATH)
    if image_bgr is None:
        print(f"[오류] 이미지를 못 읽었습니다: {_IMAGE_PATH}")
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    objects = parse_hrsc_objects(_ANNOTATION_PATH)
    print(f"어노테이션에서 찾은 선박 인스턴스: {len(objects)}건")

    checks: list[tuple[str, bool]] = []
    subclassifier = _get_ship_subclassifier()
    print(f"세분류기: {type(subclassifier).__name__}")
    checks.append(("실제 학습된 모델이 로드됨(Mock 폴백 아님)",
                    type(subclassifier).__name__ == "FinetunedShipSubclassifier"))

    for class_id, cx, cy, w, h, ang in objects:
        corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(ang))).astype(np.float32)
        crop = crop_rotated_box(image_rgb, corners)
        result = subclassifier.classify(crop)
        print(f"\n  Class_ID={class_id} crop_shape={crop.shape} -> "
              f"sub_class={result.sub_class} (conf={result.confidence:.3f}, source={result.source})")

        checks.append((f"Class_ID={class_id} -- sub_class가 4개 카테고리 중 하나로 채워짐",
                        result.sub_class in {"aircraft_carrier", "warcraft", "merchant_ship", "submarine"}))
        checks.append((f"Class_ID={class_id} -- source가 finetuned_v1(실제 모델 사용)",
                        result.source == "finetuned_v1"))
        checks.append((f"Class_ID={class_id} -- confidence가 0~1 범위",
                        0.0 <= result.confidence <= 1.0))

    print("\n" + "=" * 60)
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
        print(f"전체 {len(checks)}개 검증 통과 -- 크롭->세분류 파이프라인은 실제 데이터로 정상 동작 확인.")
        return 0
    else:
        print("일부 검증 실패 -- 위 FAIL 항목 확인 필요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
