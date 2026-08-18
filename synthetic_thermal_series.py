"""
발사장 열이상 감시용 합성 시계열 생성기.

실제로는 NASA FIRMS 같은 공개 위성 열이상 API에서 특정 좌표(발사장)의 밝기온도(brightness
temperature) 시계열을 받아오게 되지만, 지금은 그 데이터 없이도 이상탐지 로직 자체를
검증할 수 있도록 비슷한 패턴의 합성 시계열을 만든다.

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
