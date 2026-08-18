"""
agent/observation.py의 "순수 변환 함수"(radar_track_to_dict, cv_detection_to_dict)
검증 스크립트.

[샌드박스 검증 범위] 이 두 함수는 실제 API·모델 호출 없이 손으로 만든 객체만으로
완전히 검증 가능하다 (observation.py 모듈 docstring 참고). collect_radar_observation/
collect_cv_observation(실제 네트워크·모델 호출)은 여기서 검증하지 않는다 — 이건
사용자 PC에서 확인해야 하는 부분이다 (README "다음 할 일" 참고).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from agent.observation import cv_detection_to_dict, radar_track_to_dict
from config import PROTECTED_ZONES
from data_sources.flight_tracker import AirTrack


@dataclass
class FakeCVDetection:
    """cv_detection.CVDetection과 필드가 같은 가짜 객체 (ultralytics 없이 테스트하기 위함)."""
    object_class: str
    confidence: float
    lat: float
    lon: float
    source: str = "TEST"


def main() -> int:
    checks: list[tuple[str, bool]] = []

    # --- radar_track_to_dict ------------------------------------------------

    # 1) 위경도가 없는 트랙 -> None 반환 (OpenSky가 가끔 null을 줄 때 대비)
    track_no_pos = AirTrack(
        icao24="abc123", callsign="KAL123", longitude=None, latitude=None,
        altitude_m=1000.0, velocity_ms=200.0, heading_deg=90.0, vertical_rate_ms=0.0,
    )
    result_no_pos = radar_track_to_dict(track_no_pos, PROTECTED_ZONES)
    checks.append(("위경도 없는 트랙 -> None 반환", result_no_pos is None))

    # 2) 인천국제공항 관제권(37.4602, 126.4407) 내부, 호출부호 없음(UNKNOWN)
    track_in_zone = AirTrack(
        icao24="def456", callsign=None, longitude=126.44, latitude=37.46,
        altitude_m=500.0, velocity_ms=150.0, heading_deg=45.0, vertical_rate_ms=2.0,
    )
    d_in_zone = radar_track_to_dict(track_in_zone, PROTECTED_ZONES)
    checks.append(("보호구역 내부 트랙 -> in_protected_zone True", d_in_zone is not None and d_in_zone["in_protected_zone"] is True))
    checks.append(("호출부호 없는 트랙 -> identity UNKNOWN", d_in_zone["identity"] == "UNKNOWN"))
    checks.append(("고도(m->ft) 변환 정확성", d_in_zone["altitude_ft"] == round(500.0 * 3.28084, 0)))
    checks.append(("track_id는 icao24를 그대로 사용", d_in_zone["track_id"] == "def456"))

    # 3) 보호구역에서 멀리 떨어짐 + 호출부호 있음(NEUTRAL)
    track_far = AirTrack(
        icao24="ghi789", callsign="AAR456", longitude=127.5, latitude=36.25,
        altitude_m=10000.0, velocity_ms=230.0, heading_deg=180.0, vertical_rate_ms=0.0,
    )
    d_far = radar_track_to_dict(track_far, PROTECTED_ZONES)
    checks.append(("멀리 떨어진 트랙 -> in_protected_zone False", d_far["in_protected_zone"] is False))
    checks.append(("호출부호 있는 트랙 -> identity NEUTRAL", d_far["identity"] == "NEUTRAL"))

    # 4) 고도 정보가 없는 트랙 -> altitude_ft None (크래시 없이 안전 처리)
    track_no_alt = AirTrack(
        icao24="jkl012", callsign="XYZ", longitude=128.0, latitude=35.5,
        altitude_m=None, velocity_ms=None, heading_deg=None, vertical_rate_ms=None,
    )
    d_no_alt = radar_track_to_dict(track_no_alt, PROTECTED_ZONES)
    checks.append(("고도 정보 없음 -> altitude_ft None", d_no_alt["altitude_ft"] is None))
    checks.append(("속도 정보 없음 -> speed_mps 0.0으로 안전 처리", d_no_alt["speed_mps"] == 0.0))

    # --- cv_detection_to_dict -------------------------------------------------

    det = FakeCVDetection(object_class="large-vehicle", confidence=0.77, lat=35.11, lon=129.06)
    d_cv = cv_detection_to_dict(det)
    checks.append(("cv_detection_to_dict -> class 필드 매핑", d_cv["class"] == "large-vehicle"))
    checks.append(("cv_detection_to_dict -> confidence 필드 매핑", d_cv["confidence"] == 0.77))
    checks.append(("cv_detection_to_dict -> lat/lon 매핑", d_cv["lat"] == 35.11 and d_cv["lon"] == 129.06))

    # --- 결과 요약 --------------------------------------------------------
    print("=" * 60)
    print("observation.py 순수 변환 함수 검증 결과")
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
