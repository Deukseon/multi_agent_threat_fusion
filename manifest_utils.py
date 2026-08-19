"""
build_manifest.py가 만든 manifest.csv를 다루는 순수 로직 — csv/collections만 쓰고
torch를 전혀 안 건드린다. 그래서 GPU/torch 설치 없이도(샌드박스에서도) 완전히
검증 가능하다.

train_subclassifier.py(torch 필요, 사용자 PC 전용)가 이 모듈을 그대로 가져다 쓴다 —
"학습에 쓸 데이터를 정리하는 부분"과 "실제로 모델을 학습시키는 부분"을 나누는
이 프로젝트의 기존 설계 원칙(Forecaster 인터페이스 분리 등)과 같은 이유.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

# ship_subclassifier.HRSC2016_LAYER1_CATEGORIES는 런타임 예측용 안전값
# "unknown_ship_type"까지 포함한 5개짜리인 반면, 여기 LAYER1_LABELS는 실제로
# 모델이 배우는 클래스(정답이 있는 것)만 4개 — 학습 라벨 공간과 예측 출력
# 공간의 목적이 다르므로 일부러 분리해뒀다.
LAYER1_LABELS = ("aircraft_carrier", "warcraft", "merchant_ship", "submarine")
LABEL_TO_IDX = {label: i for i, label in enumerate(LAYER1_LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}


def load_manifest(path: Path) -> list[dict]:
    """manifest.csv를 읽어 행(dict) 리스트로 반환."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_split(rows: list[dict], split_name: str) -> list[dict]:
    """특정 split(train/val/test)에 속한 행만 골라낸다."""
    return [r for r in rows if r["split"] == split_name]


def class_counts(rows: list[dict]) -> dict[str, int]:
    """LAYER1_LABELS 순서 그대로, 각 클래스별 등장 횟수(없으면 0)를 반환."""
    counts = Counter(r["layer1_category"] for r in rows)
    return {label: counts.get(label, 0) for label in LAYER1_LABELS}


def compute_class_weights(rows: list[dict]) -> list[float]:
    """CrossEntropyLoss(weight=...)에 바로 넣을 수 있는 클래스별 가중치.

    inverse-frequency 방식: weight_c = total / (num_classes * count_c).
    표본이 적은 클래스(예: submarine)에 더 큰 가중치를 줘서, 다수 클래스만
    맞혀도 손실이 낮아지는 걸 막는다. count_c가 0인 클래스(그 split에 해당
    클래스 표본이 아예 없음)는 0으로 나누기를 피하려고 가중치를 1.0으로 둔다 —
    이 경우 애초에 그 클래스를 학습 못 하니 별도로 로그에 경고해야 한다
    (train_subclassifier.py의 class_counts 출력으로 확인 가능).
    """
    counts = Counter(r["layer1_category"] for r in rows)
    total = sum(counts.values())
    n_classes = len(LAYER1_LABELS)
    weights = []
    for label in LAYER1_LABELS:
        c = counts.get(label, 0)
        if c == 0 or total == 0:
            weights.append(1.0)
        else:
            weights.append(total / (n_classes * c))
    return weights
