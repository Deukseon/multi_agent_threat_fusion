"""
IR 이상탐지 에이전트 (다음 프로젝트 Phase 1 프로토타입)

위성이 특정 발사장 좌표를 주기적으로 관측한 "적외선 밝기(열이상)" 시계열을 받아서,
Chronos-2(시계열 파운데이션 모델)로 "정상적으로 예상되는 다음 값"을 zero-shot 예측하고,
실제 관측값이 예측 구간(quantile band)을 벗어나면 이상(로켓엔진 점화 등 열원 급상승)으로
판단하는 forecast-then-compare 방식 이상탐지 모듈.

핵심 설계: 예측기(forecaster)를 인터페이스로 분리해서, 실제 운영에서는 Chronos2Pipeline을
쓰고 테스트/개발 중에는 MockForecaster로 같은 로직을 검증할 수 있게 했다. 이렇게 하면
"모델을 어떤 걸 쓰느냐"와 "이상탐지 로직 자체가 맞는가"를 독립적으로 검증할 수 있다.

[2026-08-14] 샌드박스 환경에서는 huggingface.co 접속이 막혀 있어(네트워크 정책상 403)
실제 Chronos-2 가중치를 받을 수 없었다. 그래서 이 프로토타입은 MockForecaster로
로직만 검증했고, 실제 Chronos-2 동작 검증은 인터넷이 열려 있는 사용자 컴퓨터에서
ChronosForecaster로 교체해서 실행해야 한다 (아래 __main__ 참고, 한 줄만 바꾸면 됨).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# 1. 예측기 인터페이스 (Chronos-2 실제 구현과 Mock 구현이 이 프로토콜을 공유)
# ---------------------------------------------------------------------------

class Forecaster(Protocol):
    def predict_quantiles(
        self, context: np.ndarray, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        context: 과거 관측값 시계열 (1차원 배열)
        반환: (quantile_preds, mean_preds)
          quantile_preds shape = (prediction_length, len(quantile_levels))
          mean_preds shape = (prediction_length,)
        """
        ...


class ChronosForecaster:
    """실제 운영용 — Chronos-2 zero-shot 예측. 인터넷 되는 환경(사용자 PC)에서만 동작."""

    def __init__(self, model_name: str = "amazon/chronos-2", device: str = "cpu"):
        from chronos import Chronos2Pipeline  # 지연 import (mock 전용 실행 시 불필요)
        self.pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device)

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        import torch
        context_t = torch.tensor(context, dtype=torch.float32)
        quantiles, mean = self.pipeline.predict_quantiles(
            [context_t], prediction_length=prediction_length, quantile_levels=quantile_levels
        )
        # [버그 수정, 2026-08-14] Chronos2Pipeline.predict_quantiles의 실제 반환 shape은
        # (n_variates, prediction_length, len(quantile_levels)) — 처음엔 이 n_variates
        # 축을 빼먹고 (prediction_length, len(quantile_levels))라고 잘못 가정해서
        # "index 2 is out of bounds for axis 1 with size 1" 에러가 났었다 (실제 실행 시
        # 사용자 PC에서 재현됨, chronos.chronos2.pipeline 소스 코드로 원인 확인).
        # 우리는 단변량(univariate) 시계열 1개만 다루므로 n_variates=1을 squeeze로 제거.
        q = quantiles[0].squeeze(0).numpy()   # -> (prediction_length, len(quantile_levels))
        m = mean[0].squeeze(0).numpy()        # -> (prediction_length,)
        return q, m


class MockForecaster:
    """
    [샌드박스 검증용] Chronos-2 없이 이상탐지 로직만 검증하기 위한 대역.
    최근 구간의 이동평균 + 이동표준편차로 정규분포를 가정한 예측 구간을 만든다.
    실제 시계열 파운데이션 모델의 예측력은 전혀 반영 안 됨 — 오직
    "forecast-then-compare 파이프라인 배관(plumbing)이 맞는가"만 검증하는 용도.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        recent = context[-self.window:]
        mu = float(np.mean(recent))
        sigma = float(np.std(recent)) + 1e-6
        mean_preds = np.full(prediction_length, mu)
        quantile_preds = np.zeros((prediction_length, len(quantile_levels)))
        for i, q in enumerate(quantile_levels):
            z = _norm_ppf(q)
            quantile_preds[:, i] = mu + z * sigma
        return quantile_preds, mean_preds


def _norm_ppf(p: float) -> float:
    """표준정규분포 분위수 함수 (scipy 없이 근사, Acklam's algorithm 간소화 버전)"""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    # Beasley-Springer-Moro 근사
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---------------------------------------------------------------------------
# 2. 이상탐지 로직 (forecast-then-compare)
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    step: int
    actual: float
    predicted_median: float
    lower_bound: float
    upper_bound: float
    is_anomaly: bool
    severity: str  # "NORMAL" / "WATCH" / "ANOMALY"


def detect_anomaly(
    forecaster: Forecaster,
    context: np.ndarray,
    actual_next: float,
    step: int,
    quantile_levels: list[float] = None,
) -> AnomalyResult:
    """
    context(과거 관측값)로 다음 1스텝을 예측하고, 실제 관측값(actual_next)과 비교해서
    예측 구간을 벗어나면 이상으로 판정한다.

    등급 판정: 90% 구간(0.05~0.95) 밖이면 WATCH, 98% 구간(0.01~0.99) 밖이면 ANOMALY.
    """
    quantile_levels = quantile_levels or [0.01, 0.05, 0.5, 0.95, 0.99]
    quantiles, mean_preds = forecaster.predict_quantiles(context, prediction_length=1, quantile_levels=quantile_levels)

    idx = {q: i for i, q in enumerate(quantile_levels)}
    median = quantiles[0, idx[0.5]]
    lo_90, hi_90 = quantiles[0, idx[0.05]], quantiles[0, idx[0.95]]
    lo_98, hi_98 = quantiles[0, idx[0.01]], quantiles[0, idx[0.99]]

    if actual_next < lo_98 or actual_next > hi_98:
        severity, is_anomaly = "ANOMALY", True
    elif actual_next < lo_90 or actual_next > hi_90:
        severity, is_anomaly = "WATCH", True
    else:
        severity, is_anomaly = "NORMAL", False

    return AnomalyResult(
        step=step, actual=float(actual_next), predicted_median=float(median),
        lower_bound=float(lo_90), upper_bound=float(hi_90),
        is_anomaly=is_anomaly, severity=severity,
    )


def run_monitoring_loop(
    forecaster: Forecaster, series: np.ndarray, min_context: int = 20
) -> list[AnomalyResult]:
    """시계열 전체를 슬라이딩 윈도우로 훑으면서 매 시점마다 이상 여부를 판정."""
    results = []
    for step in range(min_context, len(series)):
        context = series[:step]
        actual = series[step]
        results.append(detect_anomaly(forecaster, context, actual, step))
    return results


if __name__ == "__main__":
    # [2026-08-14] 기본값을 실제 Chronos-2로 바꿈. 인터넷이 안 되는 환경(예: 샌드박스)에서
    # 로직만 다시 확인하고 싶으면 아래 주석의 MockForecaster()로 바꿔서 실행하면 됨.
    forecaster = ChronosForecaster()
    # forecaster = MockForecaster()

    from synthetic_thermal_series import generate_thermal_series
    series, true_anomaly_steps = generate_thermal_series(seed=42)

    results = run_monitoring_loop(forecaster, series)

    detected_steps = [r.step for r in results if r.is_anomaly]
    print(f"실제 이상 구간: {true_anomaly_steps}")
    print(f"탐지된 이상 구간: {detected_steps}")
    for r in results:
        if r.is_anomaly:
            print(f"  step={r.step:3d} actual={r.actual:6.2f} 예측중앙값={r.predicted_median:6.2f} "
                  f"90%구간=[{r.lower_bound:.2f},{r.upper_bound:.2f}] -> {r.severity}")
