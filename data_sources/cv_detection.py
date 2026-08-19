"""
실제 컴퓨터비전 기반 객체탐지 (EO/IR 표적탐지) — sitrep_fusion_agent Phase 6의
data_sources/cv_detection.py를 그대로 가져옴. YOLO26-OBB(DOTA-v1.0 파인튜닝,
mAP50=0.628) 모델로 비행기·선박·차량 등을 실제로 탐지한다.

[2026-08-18] 모델 가중치 파일(21.5MB)은 device_commit_files의 20MB 제한을 넘어서
이 세션에서 직접 전송할 수 없다 — 사용자가 PowerShell에서 다음 명령으로 직접
복사해야 한다:
    Copy-Item "C:\\dev\\sitrep_fusion_agent\\models\\yolo26s_obb_dota_best.pt" `
              "C:\\dev\\multi_agent_threat_fusion\\models\\yolo26s_obb_dota_best.pt"
모델 파일이 없으면 이 모듈은 import는 되지만 detect_objects() 호출 시 에러가 난다
(cv_agent는 관측 데이터에 cv_detections 키가 없으면 애초에 안 불리므로, 모델을 아직
안 옮겼어도 다른 에이전트는 정상 동작한다).

핵심 개념 - 지리참조(georeferencing): CV 모델은 "이미지 안 픽셀 좌표"로 결과를 준다.
이미지가 실제로 커버하는 지리적 범위(GeoBounds)를 안다는 전제 하에 픽셀 좌표를
선형 보간해서 위경도로 변환한다.

[2026-08-19] ⑫ 희소 군사 자산 세분류(선박 v1, `ship_subclassification` 저장소에서
검증 완료 — HRSC2016 실데이터 기준 test accuracy 94%)를 여기 통합했다. YOLO는
"ship"까지만 구분하고 함종(항공모함/전투함/상선/잠수함)은 절대 못 낸다는 게
README "알려진 한계"였는데, `object_class == "ship"`인 탐지에 한해 크롭 →
세분류기 호출을 한 단계 더 거쳐서 CVDetection.sub_class를 채운다. 크롭·세분류
로직 자체(`crop_utils.py`, `ship_subclassifier.py`)는 별도 저장소에서 실제
데이터로 이미 검증했으므로 여기선 "그 결과를 이 파이프라인에 연결하는 배관"만
새로 짰다 — YOLO 모델과 마찬가지로 지연 로드해서, 세분류 가중치 파일이 아직
없어도(models/ship_subclassifier_v1.pt) 이 모듈 자체는 문제없이 import된다.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)

_FINETUNED_WEIGHTS = os.path.join(_PROJECT_ROOT, "models", "yolo26s_obb_dota_best.pt")
_SHIP_SUBCLASSIFIER_WEIGHTS = os.path.join(_PROJECT_ROOT, "models", "ship_subclassifier_v1.pt")

_model_cache = {}
_ship_subclassifier_cache = {}


@dataclass
class GeoBounds:
    """이미지가 실제로 커버하는 지리적 범위 (실제로는 위성/드론 메타데이터에서 옴)"""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class CVDetection:
    object_class: str        # DOTA-v1.0 15개 클래스 중 하나 (plane, ship, large-vehicle 등)
    confidence: float        # 0~1
    lat: float
    lon: float
    source: str = "YOLO26-OBB(DOTA)"
    # object_class == "ship"일 때만 채워짐(⑫ v1) — 그 외 클래스는 항상 None.
    # aircraft_carrier/warcraft/merchant_ship/submarine/unknown_ship_type 중 하나.
    sub_class: Optional[str] = None
    sub_class_confidence: Optional[float] = None


def _get_model(weights: str = _FINETUNED_WEIGHTS):
    if weights not in _model_cache:
        from ultralytics import YOLO
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def _get_ship_subclassifier(weights: str = _SHIP_SUBCLASSIFIER_WEIGHTS):
    """선박 세분류기를 지연 로드·캐싱해서 반환.

    가중치 파일(models/ship_subclassifier_v1.pt, ship_subclassification 저장소에서
    복사해와야 함)이 없으면 MockShipSubclassifier로 안전하게 폴백한다 — YOLO
    가중치가 없을 때 detect_objects()가 아예 에러를 내는 것과 다른 선택인데,
    이유는 ship 세분류는 "있으면 좋은 부가 정보"라 파이프라인 전체를 막을
    이유가 없어서다(cv_agent는 sub_class가 없어도 기존처럼 동작함, 아래
    agent/specialists.py 참고). 다만 Mock은 실제 판단 근거로 쓰면 안 되는
    가로세로 비율 추정값이므로, 폴백 시 경고 로그를 남긴다.
    """
    if weights not in _ship_subclassifier_cache:
        from ship_subclassifier import FinetunedShipSubclassifier, MockShipSubclassifier

        if os.path.exists(weights):
            _ship_subclassifier_cache[weights] = FinetunedShipSubclassifier(weights, device="cpu")
        else:
            print(
                f"[cv_detection] 경고: 선박 세분류 가중치({weights})가 없어 "
                "MockShipSubclassifier로 폴백함 — 실제 판단 근거로 쓰지 말 것."
            )
            _ship_subclassifier_cache[weights] = MockShipSubclassifier()
    return _ship_subclassifier_cache[weights]


def _pixel_to_latlon(px: float, py: float, img_w: int, img_h: int, bounds: GeoBounds) -> tuple[float, float]:
    frac_x = px / img_w
    frac_y = py / img_h
    lon = bounds.lon_min + frac_x * (bounds.lon_max - bounds.lon_min)
    lat = bounds.lat_max - frac_y * (bounds.lat_max - bounds.lat_min)
    return lat, lon


def detect_objects(image_path: str, bounds: GeoBounds, conf_threshold: float = 0.4,
                    weights: str = _FINETUNED_WEIGHTS, imgsz: int = 1024,
                    classify_ship_subtype: bool = True) -> list[CVDetection]:
    """이미지 한 장에서 객체를 탐지하고, 각 탐지 결과를 위경도로 지리참조해서 반환.

    [2026-08-18 수정] `imgsz`를 명시적으로 안 넘기면 ultralytics는 기본값 640으로
    추론한다. 그런데 이 모델은 `imgsz=1024`로 파인튜닝됐다(`args.yaml` 확인) — 학습과
    추론 해상도가 다르면 특히 작은 객체(항구·선박처럼 가늘고 긴 객체)의 재현율이
    크게 떨어질 수 있다. 실제로 사용자 PC에서 실 이미지로 검증하다가 이 불일치 때문에
    탐지가 전혀 안 나오는 걸 발견해서(빈 리스트 반환), 학습 해상도와 맞춰 기본값을
    1024로 명시했다.

    [2026-08-19 추가] `classify_ship_subtype=True`(기본값)면 `object_class == "ship"`인
    탐지에 한해 크롭 → 선박 세분류기를 한 번 더 거쳐 `sub_class`/`sub_class_confidence`를
    채운다. `False`로 넘기면 YOLO 탐지만 하고 세분류는 건너뛴다(속도 우선 배치 처리 등에
    유용). 세분류가 개별 탐지 단위로 실패해도(크롭이 이미지 경계 밖이라 빈 배열이 되는
    등) 그 탐지 하나만 sub_class=None으로 남기고 넘어가며, 전체 탐지 파이프라인을
    죽이지 않는다 — YOLO 탐지 자체가 이미 다중 객체를 다루는 코드라 같은 원칙을 유지."""
    model = _get_model(weights)
    results = model(image_path, conf=conf_threshold, imgsz=imgsz, verbose=False)

    ship_subclassifier = _get_ship_subclassifier() if classify_ship_subtype else None

    detections = []
    for r in results:
        img_h, img_w = r.orig_shape
        if r.obb is None or len(r.obb) == 0:
            continue

        image_rgb = None  # 필요할 때만(ship 탐지가 실제로 있을 때만) 변환 — 매 이미지 1회로 충분
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            cx, cy = r.obb.xywhr[i][:2].tolist()
            lat, lon = _pixel_to_latlon(cx, cy, img_w, img_h, bounds)
            object_class = r.names[cls_id]

            sub_class = None
            sub_class_confidence = None
            if object_class == "ship" and ship_subclassifier is not None:
                try:
                    import numpy as np

                    from crop_utils import crop_rotated_box

                    if image_rgb is None:
                        import cv2
                        image_rgb = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)

                    corners = r.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)  # (4, 2) 픽셀 좌표
                    crop = crop_rotated_box(image_rgb, corners)
                    result = ship_subclassifier.classify(crop)
                    sub_class = result.sub_class
                    sub_class_confidence = round(result.confidence, 3)
                except Exception as exc:  # noqa: BLE001 — 세분류 실패가 탐지 전체를 죽이면 안 됨
                    print(f"[cv_detection] 선박 세분류 실패(탐지는 유지, sub_class=None): {exc}")

            detections.append(CVDetection(
                object_class=object_class,
                confidence=round(conf, 3),
                lat=round(lat, 6),
                lon=round(lon, 6),
                sub_class=sub_class,
                sub_class_confidence=sub_class_confidence,
            ))
    return detections


if __name__ == "__main__":
    import time

    test_bounds = GeoBounds(lat_min=35.10, lat_max=35.12, lon_min=129.05, lon_max=129.08)
    test_image = os.path.join(_PROJECT_ROOT, "cv_test", "sample.jpg")

    print("=== 이미지에서 객체 탐지 ===")
    t0 = time.time()
    dets = detect_objects(test_image, test_bounds)
    print(f"탐지 {len(dets)}건, {time.time()-t0:.2f}초")
    for d in dets[:5]:
        print(f"  {d.object_class} (신뢰도 {d.confidence}) @ ({d.lat}, {d.lon})")
