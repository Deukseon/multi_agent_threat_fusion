"""
[PC 전용 검증 스크립트 v2] collect_cv_observation()이 실제 이미지에서
sub_class(선박 세분류)를 정말로 채우는지 종단 간(end-to-end)으로 확인한다.

[2026-08-19 v2] v1은 conf_threshold=0.25 고정이라, HRSC2016 이미지에서 YOLO가
아예 아무것도 못 잡았을 때(탐지 0건) "임계값이 너무 높아서인지" vs "이 이미지
자체에서 뭘 찾아도 안 잡히는 근본적인 문제인지"를 구분할 수 없었음. v2는 여러
임계값을 자동으로 훑고(threshold sweep), "ship"이 아닌 다른 클래스라도 뭔가
잡히는지까지 같이 보여줘서 원인을 구분할 수 있게 함.

샌드박스에선 ultralytics/torch가 없어 이 스크립트 자체를 실행할 수 없었다 -- 로직은
cv_detection.py/observation.py 코드 리뷰로만 검증했고, 실제로 YOLO가 ship을 잡고
세분류기가 sub_class를 채우는지는 이 스크립트로 사용자 PC에서 확인해야 한다.

사용법 (C:\\dev\\multi_agent_threat_fusion 에서, 가상환경 활성화 후):
    python verify_ship_subtype_integration.py
    python verify_ship_subtype_integration.py "다른\\이미지\\경로.bmp"
"""
import os
import sys

_DEFAULT_IMAGE = os.path.join(
    "C:\\", "dev", "ship_subclassification", "data", "raw", "HRSC2016",
    "FullDataSet", "AllImages", "100000641.bmp",
)

_THRESHOLDS = [0.25, 0.10, 0.05, 0.01]


def sweep_raw_detections(image_path: str):
    """conf_threshold를 낮춰가며 YOLO 원본 출력을 직접 확인 -- cv_detection.py의
    detect_objects()를 거치지 않고 ultralytics 모델을 바로 호출해서, 세분류
    로직과 무관하게 "이 이미지에서 YOLO가 뭐라도 잡는가"만 순수하게 확인한다."""
    from data_sources.cv_detection import _FINETUNED_WEIGHTS, _get_model

    model = _get_model(_FINETUNED_WEIGHTS)
    print(f"모델 가중치: {_FINETUNED_WEIGHTS}")
    print(f"이미지: {image_path}")

    for conf in _THRESHOLDS:
        results = model(image_path, conf=conf, imgsz=1024, verbose=False)
        r = results[0]
        n = 0 if r.obb is None else len(r.obb)
        print(f"\n  conf_threshold={conf} -> 원본 탐지 {n}건 (클래스 무관, ship 포함 전체)")
        if n > 0:
            for i in range(min(n, 10)):
                cls_id = int(r.obb.cls[i].item())
                c = float(r.obb.conf[i].item())
                print(f"      {r.names[cls_id]} (conf={c:.3f})")


def main() -> int:
    from data_sources.cv_detection import GeoBounds, detect_objects

    image_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_IMAGE
    if not os.path.exists(image_path):
        print(f"[오류] 이미지가 없습니다: {image_path}")
        print("HRSC2016 데이터셋 경로가 다르면 인자로 실제 이미지 경로를 넘겨주세요.")
        return 1

    print("=" * 60)
    print("[진단] 임계값(conf_threshold)을 낮춰가며 YOLO 원본 탐지 결과 확인")
    print("=" * 60)
    sweep_raw_detections(image_path)

    print("\n" + "=" * 60)
    print("[본검증] classify_ship_subtype 파이프라인 (conf_threshold=0.10으로 낮춰서 재시도)")
    print("=" * 60)

    dummy_bounds = GeoBounds(lat_min=35.00, lat_max=35.02, lon_min=129.00, lon_max=129.02)
    checks: list[tuple[str, bool]] = []

    dets_on = detect_objects(image_path, dummy_bounds, conf_threshold=0.10, classify_ship_subtype=True)
    print(f"\n[1] classify_ship_subtype=True (conf=0.10) -- 탐지 {len(dets_on)}건")
    for d in dets_on:
        print(f"    {d.object_class} (conf={d.confidence}) sub_class={d.sub_class} "
              f"(sub_conf={d.sub_class_confidence}) source={d.source}")

    ship_dets = [d for d in dets_on if d.object_class == "ship"]
    checks.append(("YOLO가 이미지에서 'ship'을 최소 1건 탐지함(conf=0.10)", len(ship_dets) > 0))
    if ship_dets:
        checks.append(("탐지된 ship 중 최소 1건은 sub_class가 채워짐(None 아님)",
                        any(d.sub_class is not None for d in ship_dets)))
        checks.append(("sub_class가 4개 카테고리 중 하나(또는 unknown_ship_type)",
                        all(d.sub_class in {"aircraft_carrier", "warcraft", "merchant_ship",
                                             "submarine", "unknown_ship_type", None}
                            for d in ship_dets)))

    dets_off = detect_objects(image_path, dummy_bounds, conf_threshold=0.10, classify_ship_subtype=False)
    print(f"\n[2] classify_ship_subtype=False (conf=0.10) -- 탐지 {len(dets_off)}건")
    checks.append(("세분류 꺼도 탐지 건수는 동일", len(dets_off) == len(dets_on)))
    checks.append(("세분류 꺼면 sub_class는 전부 None", all(d.sub_class is None for d in dets_off)))

    weights_path = os.path.join("models", "ship_subclassifier_v1.pt")
    weights_exist = os.path.exists(weights_path)
    print(f"\n[3] 체크포인트 파일 존재 여부: {weights_exist} ({weights_path})")
    if ship_dets:
        sources = {d.source for d in ship_dets if d.sub_class is not None}
        print(f"    세분류 source 값: {sources}")

    print("\n" + "=" * 60)
    print("검증 결과 요약")
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
        print("일부 검증 실패 -- 위 [진단] 섹션에서 어느 임계값부터 뭐가 잡히는지 확인하세요.")
        print("모든 임계값(0.01까지)에서 0건이면, 이 이미지가 YOLO 학습 데이터(DOTA)와")
        print("스타일/스케일이 너무 달라서 못 잡는 도메인 갭 문제일 가능성이 높습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
