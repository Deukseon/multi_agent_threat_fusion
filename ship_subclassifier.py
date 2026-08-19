"""
2단계 선박 세분류기 (⑫ 희소 군사 자산 세분류 프로젝트, v1 — 선박만).

배경: multi_agent_threat_fusion의 cv_agent는 YOLO26-OBB(DOTA-v1.0)로 "ship"까지만
구분하고, 함종(항공모함/전투함/상선 등)은 절대 구분하지 못한다 — DOTA엔 그런 세부
라벨이 애초에 없기 때문(closed-set 한계, README "알려진 한계" 참고). 이 프로젝트는
"1단계 탐지(ship) → 2단계 세분류(함종)"를 별도로 검증하는 게 목적.

설계 원칙 (multi_agent_threat_fusion의 Forecaster 인터페이스 분리, ir_anomaly_agent.py와
동일한 사상): "실제 모델을 불러서 추론하는 부분"과 "그 결과를 우리 포맷으로 감싸는
부분"을 인터페이스(ABC)로 분리한다. Mock 구현은 네트워크·모델 파일 없이 샌드박스에서
파이프라인 배관(crop → classify → 결과 dict)이 맞는지 검증하는 용도이고, 실제 학습된
모델은 사용자 PC(GPU, 데이터셋 O)에서만 검증 가능하다.

v1 범위: HRSC2016의 Layer-1 카테고리(상위 함종 4개: 항공모함/전투함/상선/잠수함)만
다룬다. Layer-2(세부 함종 29개, 예: 니미츠급 항공모함, 알레이버크급 구축함 등)나
FGSCR-42(42개 세분류, 이미지당 분류 라벨) 확장은 v1 검증 후 다음 단계로 미룬다.

[2026-08-18 갱신] 아래 클래스 목록은 더 이상 문헌조사 기반 추측이 아니다 — Kaggle에서
실제 HRSC2016_dataset.zip을 받아 `FullDataSet/sysdata.xml`을 직접 열어 확인한 값이다.
원래 예상은 "L2 상위 카테고리 3~4개"였는데, 실제로는 3단계 트리 구조였다:
  - Layer 0: ship (전체 루트, Class_ID=100000001) — "선박인 건 확실한데 세부 함종은
    불확실"한 객체에 실제로 쓰인다(예: Img 100000008, 100000721 등의 Annotations에서
    직접 확인). 즉 이 값은 "미분류" 캐치올로도 등장하므로, 학습 라벨을 만들 때
    Class_ID=100000001인 객체는 Layer 1로 못 올리고 버리거나 별도 처리해야 한다.
  - Layer 1: 상위 함종 카테고리 4개 (아래 HRSC2016_LAYER1_CATEGORIES)
  - Layer 2: 세부 함종 26개 (아래 HRSC2016_LAYER2_TO_LAYER1, v1.5 확장 후보) — 최초
    문헌조사 메모에는 29개로 추정했었는데, sysdata.xml을 직접 파싱해서 센 실제 개수는
    26개다(Class_ID가 100000021, 100000023을 건너뛰는 등 연속적이지 않음).
개별 이미지 어노테이션(`Annotations/*.xml`)의 `Class_ID`는 Layer 2 또는 Layer 0(위 설명)을
가리키고, `HRS_Class_ID`로 부모(Layer 1)를 거슬러 올라갈 수 있다 — v1 학습 라벨은 이
매핑을 거쳐 Layer 1로 변환해서 만들 계획이다.

- HRSC2016은 Kaggle에서 바로 받을 수 있고(로그인만 필요), 계층 구조가 있어
  Layer 1만으로도 "군함형 vs 상선형" 구분이라는 실전 가치가 바로 나옴.
- FGSCR-42는 42개 클래스로 더 세밀하지만, (1) 실제 배포판이 논문 명시 9,320장 중
  약 7,700장만 내려받아진다는 커뮤니티 이슈가 있고(다운로드 불완전), (2) Baidu Pan
  (비밀번호 필요)으로만 배포돼 접근 장벽이 있다. v1에서는 굳이 이 위험을 감수하지
  않고 HRSC2016만으로 파이프라인을 먼저 검증한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# HRSC2016 Layer 1 카테고리 (상위 함종 분류) — sysdata.xml에서 실제 확인한 4개
# (Class_ID 100000002/100000003/100000004/100000027). sysdata.xml 안에서는
# submarine이 다른 3개보다 뒤쪽(Class_ID가 큼)에 나오지만 Class_Layer는 똑같이 1이다.
HRSC2016_LAYER1_CATEGORIES = (
    "aircraft_carrier",  # 航母 (Class_ID=100000002)
    "warcraft",          # 军舰 (Class_ID=100000003)
    "merchant_ship",      # 商船 (Class_ID=100000004)
    "submarine",          # 潜艇 (Class_ID=100000027)
    "unknown_ship_type",  # 분류 실패/신뢰도 낮음 — 우리 쪽에서 추가한 안전값, sysdata.xml엔 없음
)

# 위협 스코어링 관점에서 "군함형"으로 취급할 카테고리 — cv_agent 통합 시
# 참고용(아직 실제 통합은 안 함, v1은 세분류기 자체 검증까지만).
MILITARY_LOOKING_CATEGORIES = {"aircraft_carrier", "warcraft", "submarine"}
CIVILIAN_LOOKING_CATEGORIES = {"merchant_ship"}

# HRSC2016 Layer 2 카테고리(세부 함종, sysdata.xml에서 그대로 옮김) → 소속 Layer 1.
# v1은 이 세부 분류까지는 목표로 하지 않지만(Layer 1만), 어노테이션의 Class_ID가
# 대부분 이 Layer 2를 가리키므로 학습 라벨을 만들 때(Layer 2 → Layer 1 변환) 필요하고,
# v1.5(세부 함종 확장) 착수 시에도 그대로 재사용 가능하다.
# 일부 이름("OXo|--)", "lute")은 원본 데이터 자체가 이렇게 기록돼 있음 — 중국어 원문
# (예: "琵琶形军舰" = "비파 모양 군함")을 영문화하는 과정에서 생긴 표기로 추정, 오타
# 아님. 클래스별 실제 인스턴스 수는 아직 집계 안 함 — 학습 스크립트 작성 시 확인 필요
# (Layer 2는 26개나 되므로 클래스당 인스턴스가 매우 적을 가능성이 높음).
#
# [2026-08-18 검증] sysdata.xml을 Python으로 직접 파싱해서 Class_ID → HRS_Class_ID
# 매핑을 재검증한 결과, 최초 작성 당시(문헌/직관 기반) 아래 2개가 실제로는 틀려
# 있었다 — 지금은 sysdata.xml 원문 기준으로 수정된 값이다:
#   - "Tarawa-class amphibious assault ship": warcraft(X) → aircraft_carrier(O)
#     (sysdata.xml: HRS_Class_ID=100000002, 즉 항모 계열로 분류돼 있음. 강습상륙함이라
#     "군함"일 것 같은 직관과 달리, HRSC2016 자체 분류 체계에서는 항모 하위에 있음.)
#   - "Medical ship": merchant_ship(X) → warcraft(O)
#     (sysdata.xml: HRS_Class_ID=100000003. 병원선이라 "상선"일 것 같은 직관과 달리,
#     HRSC2016에서는 군함 계열로 분류돼 있음 — 실제로 해군 소속 병원선이 많아서로 추정.)
# 이 두 사례는 "직관으로 매핑을 짜면 안 되고 반드시 원본 스키마를 확인해야 한다"는
# 좋은 예시라 주석으로 남겨둔다.
HRSC2016_LAYER2_TO_LAYER1 = {
    "Nimitz class aircraft carrier": "aircraft_carrier",
    "Enterprise class aircraft carrier": "aircraft_carrier",
    "Kitty Hawk class aircraft carrier": "aircraft_carrier",
    "Admiral Kuznetsov aircraft carrier": "aircraft_carrier",
    "Ford-class aircraft carriers": "aircraft_carrier",
    "Midway-class aircraft carrier": "aircraft_carrier",
    "Invincible-class aircraft carrier": "aircraft_carrier",
    "Tarawa-class amphibious assault ship": "aircraft_carrier",
    "Arleigh Burke class destroyers": "warcraft",
    "WhidbeyIsland class landing craft": "warcraft",
    "Perry class frigate": "warcraft",
    "Sanantonio class amphibious transport dock": "warcraft",
    "Ticonderoga class cruiser": "warcraft",
    "Abukuma-class destroyer escort": "warcraft",
    "Austen class amphibious transport dock": "warcraft",
    "USS Blue Ridge (LCC-19)": "warcraft",
    "OXo|--)": "warcraft",
    "lute": "warcraft",
    "Medical ship": "warcraft",
    "Container ship": "merchant_ship",
    "Container ship(_|.--.--|_]=": "merchant_ship",
    "Car carrier([]==[])": "merchant_ship",
    "Car carrier(======|": "merchant_ship",
    "Hovercraft": "merchant_ship",
    "yacht": "merchant_ship",
    "Cruise ship": "merchant_ship",
    # sysdata.xml의 Layer 2 목록 중 submarine 밑에 딸린 세부 항목은 없었음(잠수함은
    # Layer 1에서 바로 끝남) — 나머지 Layer 2 항목은 이 프로젝트에서 실제로 등장
    # 빈도가 낮아 보여 학습 스크립트 작성 시 이미지 카운트와 함께 재검증 필요.
}


@dataclass
class ShipSubclassResult:
    sub_class: str          # 예측된 Layer 1 카테고리 (HRSC2016_LAYER1_CATEGORIES 중 하나)
    confidence: float        # 0~1
    source: str = "mock"     # "mock" 또는 실제 모델 식별자


class ShipSubclassifier(ABC):
    """1단계에서 잘라낸 선박 크롭 이미지를 받아 세부 함종을 예측하는 인터페이스."""

    @abstractmethod
    def classify(self, crop_image: np.ndarray) -> ShipSubclassResult:
        """crop_image: (H, W, 3) uint8 RGB 배열. 반환: ShipSubclassResult."""
        raise NotImplementedError


class MockShipSubclassifier(ShipSubclassifier):
    """네트워크·학습된 모델 없이 파이프라인 배관만 검증하기 위한 목 구현.

    실제 이미지 내용은 전혀 안 보고, 크롭 이미지의 가로/세로 비율(aspect ratio)만
    보고 결정론적으로 분류한다 — "완전 무작위"보다 "그럴듯한 규칙"을 흉내내서,
    이후 실제 모델로 교체했을 때 인터페이스가 똑같이 동작하는지 확인하기 쉽게 함.
    (실제 함선은 항공모함처럼 클수록 가로로 더 길쭉한 경향이 있다는 점을 아주
    단순화해서 흉내낸 것 — 실제 판단 근거로 쓰면 안 됨, 목적은 배관 검증뿐. 잠수함은
    Mock에서는 다루지 않음 — 실제 모델 학습 후 별도 검증 필요.)
    """

    def classify(self, crop_image: np.ndarray) -> ShipSubclassResult:
        h, w = crop_image.shape[:2]
        if h == 0 or w == 0:
            return ShipSubclassResult("unknown_ship_type", 0.0, source="mock")

        aspect = w / h
        if aspect > 6.0:
            return ShipSubclassResult("aircraft_carrier", 0.6, source="mock")
        elif aspect > 3.0:
            return ShipSubclassResult("warcraft", 0.55, source="mock")
        else:
            return ShipSubclassResult("merchant_ship", 0.5, source="mock")


class FinetunedShipSubclassifier(ShipSubclassifier):
    """HRSC2016(Layer 1, 4개 카테고리)으로 파인튜닝한 실제 이미지 분류 모델.

    [2026-08-19 갱신] train_subclassifier.py 작성에 맞춰 실제 추론 로직을 구현했다 —
    ImageNet 사전학습 MobileNetV3-Small(백본 freeze) + 4-클래스 Linear 헤드,
    train_subclassifier.py의 build_model()과 완전히 같은 아키텍처를 여기서도
    그대로 재구성해야 state_dict가 맞게 로드된다(같은 모양이 아니면
    load_state_dict가 조용히 실패하거나 shape mismatch 에러가 남).

    [사용자 PC 전용] 가중치 파일과 torch/torchvision 의존성이 필요해서 샌드박스에서
    로드·추론 검증이 불가능하다(huggingface.co/opensky-network.org와 같은 네트워크
    제약 패턴 — `pip install torch` 자체가 샌드박스에서 막혀 있음을 직접 확인함,
    2026-08-19). torch/torchvision import를 메서드 안에서만 하는 지연 로드 방식을
    유지해서, Mock만 쓰는 테스트(test_ship_subclassifier.py)는 torch 없이도 계속
    동작한다 — 이 파일 자체를 import할 때 에러가 나면 안 되기 때문.
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None  # 지연 로드
        self._transform = None

    def _load(self):
        if self._model is not None:
            return

        import torch  # 지연 import — Mock만 쓰는 테스트는 torch 없이도 동작
        import torch.nn as nn
        from torchvision import models, transforms

        from manifest_utils import LAYER1_LABELS

        # train_subclassifier.py의 build_model()과 반드시 동일한 구조여야 함
        # (weights=None으로 불러서 ImageNet 사전학습 다운로드를 건너뜀 — 어차피
        # state_dict를 통째로 덮어쓸 거라 사전학습 가중치는 필요 없고, 오프라인
        # 환경에서도 로드 실패 없이 동작하게 하기 위함).
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(LAYER1_LABELS))
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def classify(self, crop_image: np.ndarray) -> ShipSubclassResult:
        self._load()

        import torch
        from PIL import Image

        from manifest_utils import IDX_TO_LABEL

        h, w = crop_image.shape[:2]
        if h == 0 or w == 0:
            return ShipSubclassResult("unknown_ship_type", 0.0, source="finetuned_v1")

        pil_image = Image.fromarray(crop_image)  # crop_image는 RGB 순서 가정 (crop_utils.crop_rotated_box 출력과 동일)
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)  # 배치 차원 추가 + 모델과 같은 디바이스로

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

        return ShipSubclassResult(IDX_TO_LABEL[pred_idx], confidence, source="finetuned_v1")
