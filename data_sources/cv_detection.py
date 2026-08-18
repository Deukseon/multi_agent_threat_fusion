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
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)

_FINETUNED_WEIGHTS = os.path.join(_PROJECT_ROOT, "models", "yolo26s_obb_dota_best.pt")

_model_cache = {}


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


def _get_model(weights: str = _FINETUNED_WEIGHTS):
    if weights not in _model_cache:
        from ultralytics import YOLO
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def _pixel_to_latlon(px: float, py: float, img_w: int, img_h: int, bounds: GeoBounds) -> tuple[float, float]:
    frac_x = px / img_w
    frac_y = py / img_h
    lon = bounds.lon_min + frac_x * (bounds.lon_max - bounds.lon_min)
    lat = bounds.lat_max - frac_y * (bounds.lat_max - bounds.lat_min)
    return lat, lon


def detect_objects(image_path: str, bounds: GeoBounds, conf_threshold: float = 0.4,
                    weights: str = _FINETUNED_WEIGHTS) -> list[CVDetection]:
    """이미지 한 장에서 객체를 탐지하고, 각 탐지 결과를 위경도로 지리참조해서 반환."""
    model = _get_model(weights)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        img_h, img_w = r.orig_shape
        if r.obb is None or len(r.obb) == 0:
            continue
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            cx, cy = r.obb.xywhr[i][:2].tolist()
            lat, lon = _pixel_to_latlon(cx, cy, img_w, img_h, bounds)
            detections.append(CVDetection(
                object_class=r.names[cls_id],
                confidence=round(conf, 3),
                lat=round(lat, 6),
                lon=round(lon, 6),
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
