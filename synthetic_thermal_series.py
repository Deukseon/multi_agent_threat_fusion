"""
발사장 열이상 감시용 합성 시계열 생성기.

[2026-08-18 명확화] 1스텝=10분 간격으로 "꾸준히" 관측값이 들어온다고 가정하는데, 이건
정지궤도 위성(예: NOAA GOES-R 시리즈의 Fire/Hot Spot Characterization 제품 — 실제로
10분 간격 갱신, 무료 공개. 한반도 권역은 기상청 천리안위성 2A호/GK-2A가 같은 방식으로
"산불탐지" 제품을 만듦)의 연속 감시 패턴을 모사한 것이다. NASA FIRMS(VIIRS/MODIS 등
저궤도 위성 기반)는 특정 지점을 하루 1~4회만 지나가서 이 "매 사이클 연속 관측" 가정과
안 맞는다는 걸 나중에 확인했다(README "알려진 한계" 참고) — 그래서 지금은 FIRMS로
바꾸지 않고 이 합성 데이터를 계속 쓰되, "정지궤도 위성이 있다면 이렇게 보일 것"이라는
전제를 명시적으로 남겨둔다. 실제 GOES/GK-2A 데이터 연동은 접근 방법·인증·요청 제한을
아직 조사 안 해서 백로그로만 남긴다.

구성:
  - 위성 재방문 주기 1스텝 = 10분이라고 가정, 총 300스텝(50시간) 시뮬레이션
  - 베이스라인: 낮/밤 태양열 영향을 흉내낸 완만한 주기 변동(24시간 주기) + 관측 잡음
  - 이상 이벤트 2개 주입: (1) 로켓엔진 점화 - 급격한 스파이크 후 서서히 감쇠,
    (2) 정비용 소규모 발열 - 스파이크보다 작아서 정상 변동과 헷갈릴 수 있는 애매한 사례
"""
import math

import numpy as np


def generate_thermal_series(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 300
    t = np.arange(n)

    # 베이스라인: 24시간 주기(144스텝, 10분 간격) 완만한 열 변동 + 저강도 잡음
    baseline = 15.0 + 3.0 * np.sin(2 * np.pi * t / 144.0)
    noise = rng.normal(0, 0.6, size=n)
    series = baseline + noise

    anomaly_steps = []

    # 이상 1: 로켓엔진 점화 (step 150 부근) - 짧고 강한 스파이크 후 지수 감쇠
    ignition_start = 150
    for i in range(ignition_start, min(ignition_start + 15, n)):
        elapsed = i - ignition_start
        spike = 40.0 * math.exp(-elapsed / 4.0)
        series[i] += spike
    anomaly_steps.extend(range(ignition_start, ignition_start + 8))  # 확실히 이상으로 잡혀야 하는 구간

    # 이상 2: 소규모 정비 발열 (step 230 부근) - 약한 스파이크, 애매한 경계 사례
    minor_start = 230
    for i in range(minor_start, min(minor_start + 6, n)):
        elapsed = i - minor_start
        spike = 6.0 * math.exp(-elapsed / 2.0)
        series[i] += spike
    anomaly_steps.extend(range(minor_start, minor_start + 3))

    return series, sorted(set(anomaly_steps))
