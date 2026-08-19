"""
실제 센서 데이터 → 전문 에이전트가 기대하는 observation 딕셔너리로 변환하는 수집기.

설계 원칙 (Phase 1의 Forecaster 인터페이스 분리와 같은 사상): "원본 데이터를 우리
포맷으로 바꾸는 로직"과 "실제로 네트워크/모델을 호출하는 부분"을 함수 단위로
분리했다. 앞엣것(*_to_dict)은 순수 함수라 샌드박스에서도 네트워크 없이 완전히
검증 가능하고, 뒤엣것(collect_*)은 실제 API·모델을 불러야 해서 사용자 PC에서만
검증 가능하다. 이렇게 나눠두면 "변환 로직 자체가 맞는가"와 "실제 센서가 응답하는가"를
독립적으로 확인할 수 있다.
"""
from __future__ import annotations

from typing import Optional

from data_sources.flight_tracker import AirTrack, fetch_tracks
from fusion.geofence import check_nearest_zone
from fusion.identification import classify_identity


def radar_track_to_dict(track: AirTrack, protected_zones: list[dict]) -> Optional[dict]:
    """AirTrack(OpenSky 원본) 하나를 radar_agent가 기대하는 형태로 변환.

    위경도가 없는 트랙(OpenSky가 가끔 null로 주는 경우)은 geofence 판단이
    불가능하므로 None을 반환해서 건너뛴다.
    """
    if track.latitude is None or track.longitude is None:
        return None

    ident = classify_identity(track.callsign)
    geofence_result = check_nearest_zone(
        track.latitude, track.longitude, track.heading_deg, protected_zones
    )

    return {
        "track_id": track.icao24,
        "altitude_ft": round(track.altitude_m * 3.28084, 0) if track.altitude_m is not None else None,
        "speed_mps": round(track.velocity_ms, 1) if track.velocity_ms is not None else 0.0,
        "identity": ident.identity,
        "in_protected_zone": geofence_result.inside_zone,
        # 아래는 radar_agent 스코어링엔 안 쓰이지만, 감사/디버깅용으로 원본 보존
        "lat": track.latitude,
        "lon": track.longitude,
        "zone_name": geofence_result.zone_name,
        "zone_distance_km": geofence_result.distance_km,
        "approaching_zone": geofence_result.approaching,
    }


def collect_radar_observation(bbox: tuple, protected_zones: list[dict]) -> list[dict]:
    """실제 OpenSky API를 호출해서 radar_agent용 observation 리스트를 만든다.

    [샌드박스 검증 불가] opensky-network.org는 이 개발 환경에서 네트워크가
    막혀 있어 실제 응답을 받을 수 없다 (huggingface.co와 동일한 정책). 사용자
    PC(인터넷 O)에서 실행해야 실제 항적이 잡히는지 확인 가능. API가 실패해도
    `fetch_tracks()`가 빈 리스트를 반환하므로 이 함수도 안전하게 빈 리스트를
    반환한다 (크래시하지 않음).
    """
    tracks = fetch_tracks(bbox=bbox)
    result = []
    for track in tracks:
        converted = radar_track_to_dict(track, protected_zones)
        if converted is not None:
            result.append(converted)
    return result


# --- CV(EO/IR 영상) 쪽 -------------------------------------------------------
# cv_detection 모듈은 ultralytics(YOLO)에 의존하는데, 이 패키지는 무거워서
# (torch 등) 필요할 때만 지연 import한다 — radar만 쓰고 싶은 경우 ultralytics가
# 없어도 이 파일 자체는 문제없이 import되게 하기 위함.

# DOTA-v1.0 15개 클래스 중, 군사적으로 더 의미 있을 수 있는 플랫폼류(항공기·선박·
# 대형 차량)를 "주목 대상"으로, 나머지 민간 인프라·스포츠 시설류는 낮은 가중치로 분류.
# [한계, fusion/identification.py와 같은 원칙] DOTA 분류 자체엔 "군용/민간" 구분이
# 없다 — "plane"이 여객기인지 전투기인지 이 모델은 모른다. 위치(보호구역 근접도)·
# 다른 센서와의 corroboration으로 보완해야 하는 약한 신호로 취급해야 한다.
CV_HIGH_CONCERN_CLASSES = {"plane", "helicopter", "ship", "large-vehicle"}
CV_LOW_CONCERN_CLASSES = {"small-vehicle", "harbor", "storage-tank", "bridge", "roundabout"}
# 나머지(tennis-court, swimming-pool 등 스포츠/레저 시설)는 위협 판단에 안 씀


def cv_detection_to_dict(det) -> dict:
    """cv_detection.CVDetection 객체를 cv_agent가 기대하는 형태로 변환.

    [2026-08-19 추가] sub_class/sub_class_confidence — ⑫ 희소 군사 자산 세분류(선박
    v1)에서 온 필드. object_class != "ship"이거나 세분류가 비활성/실패한 경우 둘 다
    None이며, radar_track_to_dict의 altitude_ft 등과 같은 방식으로 값이 없어도 키
    자체는 항상 포함시켜서 dict 형태를 안정적으로 유지한다(cv_agent가 .get()으로
    안전하게 접근 가능). getattr로 접근하는 이유는 test_observation.py의
    FakeCVDetection처럼 이 필드가 없는 구버전 더미 객체로도 계속 테스트할 수 있게
    하기 위함(하위 호환)."""
    return {
        "class": det.object_class,
        "confidence": det.confidence,
        "lat": det.lat,
        "lon": det.lon,
        "sub_class": getattr(det, "sub_class", None),
        "sub_class_confidence": getattr(det, "sub_class_confidence", None),
    }


def collect_cv_observation(
    image_path: str, bounds, conf_threshold: float = 0.4, classify_ship_subtype: bool = True
) -> list[dict]:
    """실제 YOLO26-OBB 모델로 이미지를 탐지해서 cv_agent용 observation 리스트를 만든다.

    [샌드박스 검증 불가] 모델 가중치(21.5MB)가 이 세션엔 없고(20MB 전송 제한),
    실제 항공/위성 이미지도 없다. 사용자 PC에서 모델 파일을 옮기고 실제 이미지로
    실행해야 검증 가능하다 (README "다음 할 일" 참고).

    classify_ship_subtype: detect_objects()로 그대로 전달(⑫ v1). False로 두면
    선박 세분류를 건너뛰고 YOLO 탐지 결과만 반환 — sub_class 관련 필드는 전부 None.
    """
    from data_sources.cv_detection import detect_objects

    detections = detect_objects(
        image_path, bounds, conf_threshold=conf_threshold, classify_ship_subtype=classify_ship_subtype
    )
    return [cv_detection_to_dict(d) for d in detections]
