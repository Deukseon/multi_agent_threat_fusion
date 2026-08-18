"""
보호구역(geofence) 기반 위협 판단 — sitrep_fusion_agent의 fusion/geofence.py를
그대로 가져옴 (radar_agent 실제 연동에서 재사용).
"""
import math
from dataclasses import dataclass
from typing import Optional

EARTH_RADIUS_KM = 6371.0


@dataclass
class GeofenceResult:
    zone_name: Optional[str]
    distance_km: Optional[float]
    inside_zone: bool
    approaching: bool


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return _haversine_km(lat1, lon1, lat2, lon2)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return _bearing_deg(lat1, lon1, lat2, lon2)


def check_nearest_zone(lat: float, lon: float, heading_deg: Optional[float],
                        zones: list[dict]) -> GeofenceResult:
    if not zones:
        return GeofenceResult(zone_name=None, distance_km=None, inside_zone=False, approaching=False)

    nearest = min(zones, key=lambda z: _haversine_km(lat, lon, z["lat"], z["lon"]))
    distance = _haversine_km(lat, lon, nearest["lat"], nearest["lon"])
    inside = distance <= nearest["radius_km"]

    approaching = False
    if heading_deg is not None:
        bearing_to_zone = _bearing_deg(lat, lon, nearest["lat"], nearest["lon"])
        diff = abs((heading_deg - bearing_to_zone + 180) % 360 - 180)
        approaching = diff <= 60

    return GeofenceResult(
        zone_name=nearest["name"],
        distance_km=round(distance, 1),
        inside_zone=inside,
        approaching=approaching,
    )
