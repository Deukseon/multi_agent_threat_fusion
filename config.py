"""
프로젝트 전역 설정값 모음.

sitrep_fusion_agent의 config.py와 같은 패턴 — 감시 구역·보호구역·API 키를
한 곳에서 관리한다. `.env` 자동 로드도 여기서 한 번만 처리한다.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- 감시 구역 설정 (sitrep_fusion_agent와 동일값 재사용) ---
MONITOR_BBOX = (34.5, 125.5, 38.0, 129.5)   # (lamin, lomin, lamax, lomax) - 인천·부산/김해 관제권 포함
CENTER_LAT = 36.25
CENTER_LON = 127.5

# --- 보호구역(geofence) 설정 ---
PROTECTED_ZONES = [
    {"name": "인천국제공항 관제권", "lat": 37.4602, "lon": 126.4407, "radius_km": 20},
    {"name": "부산/김해 관제권", "lat": 35.1795, "lon": 128.9382, "radius_km": 20},
]

# --- API 키 (.env에서 자동 로드됨) ---
OPENSKY_CLIENT_ID = os.getenv("OPENSKY_CLIENT_ID")
OPENSKY_CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
