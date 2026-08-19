"""
회전된 바운딩박스(OBB) 영역을 이미지에서 잘라내는 순수 유틸리티.

YOLO26-OBB(ultralytics)의 결과 `r.obb.xyxyxyxy[i]`는 회전된 사각형의 4개 꼭짓점
좌표(4쌍의 (x, y), 픽셀 단위)로 나온다. 배는 물 위에서 임의의 방향을 향해 떠 있으므로
"가로세로가 이미지 축에 맞춰진(axis-aligned)" 크롭을 그냥 하면 배 주변 배경(바다)이
너무 많이 섞이거나, 심한 경우 배의 일부가 잘려나간다. 이 모듈은 이미지 전체를 배의
방향에 맞게 회전시킨 뒤 크롭해서, 세분류기가 "배 자체"만 최대한 깨끗하게 보게 한다.

순수 함수(네트워크·모델 의존 없음)라 샌드박스에서도 합성 이미지로 완전히 검증 가능.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_rotated_box(image: np.ndarray, corners: np.ndarray, pad_ratio: float = 0.1) -> np.ndarray:
    """4개 꼭짓점으로 정의된 회전 사각형 영역을, 그 사각형의 방향에 맞춰 회전시킨 뒤
    가로세로가 맞춰진 직사각형으로 잘라 반환한다.

    Args:
        image: (H, W, 3) 배열 (원본 이미지 전체, ultralytics의 `r.orig_img`에 해당).
        corners: (4, 2) 배열, 회전 사각형의 4개 꼭짓점 (x, y) 픽셀 좌표.
            ultralytics `result.obb.xyxyxyxy[i]`를 그대로 넘기면 됨.
        pad_ratio: 크롭 영역을 살짝 넉넉하게 잡기 위한 여백 비율 (기본 10%).
            선박 형태 분류기는 배 전체의 실루엣(길쭉함 등)이 중요하므로, 너무 딱
            맞게 자르면 끝부분이 잘려나가 형태 정보가 손실될 수 있어 추가함.

    Returns:
        (h', w', 3) 크롭된 이미지 배열. corners가 이미지 경계를 벗어나면 경계 안으로
        clip해서 반환 (빈 배열이 나오는 경우는 호출부에서 h==0 or w==0으로 감지 가능).
    """
    if corners.shape != (4, 2):
        raise ValueError(f"corners는 (4, 2) 형태여야 함, 받은 형태: {corners.shape}")

    img_h, img_w = image.shape[:2]
    pts = corners.astype(np.float32)

    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)

    # OpenCV 버전에 따라 minAreaRect가 반환하는 angle의 기준(어느 변이 "너비"인지)이
    # 달라서, w가 h보다 작으면(즉 "너비"로 잡힌 게 실제로는 짧은 변이면) 90도 돌려서
    # 항상 긴 변이 가로(w)가 되도록 통일한다 — 선박은 보통 길쭉하므로 크롭 결과가
    # 항상 "가로로 긴" 형태가 되게 만드는 목적.
    if w < h:
        angle += 90
        w, h = h, w

    w_padded = w * (1 + pad_ratio)
    h_padded = h * (1 + pad_ratio)

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (img_w, img_h))

    x1 = int(round(cx - w_padded / 2))
    y1 = int(round(cy - h_padded / 2))
    x2 = int(round(cx + w_padded / 2))
    y2 = int(round(cy + h_padded / 2))

    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)

    return rotated[y1:y2, x1:x2]
