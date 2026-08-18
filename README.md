# multi_agent_threat_fusion

`sitrep_fusion_agent`(단일 파이프라인 LangGraph 프로젝트)의 다음 단계로, 멀티에이전트 구조 +
시계열 이상탐지(비지도 학습 경험 보강)를 목표로 하는 신규 프로젝트입니다.

## Phase 1 — IR 이상탐지 에이전트 프로토타입 ✅ 완료 (2026-08-14~15)

멀티에이전트 뼈대를 만들기 전에, 가장 새로운 기술 요소인 **시계열 이상탐지(Chronos-2 기반
zero-shot forecast-then-compare)**가 실제로 동작하는지 먼저 단독으로 검증했습니다.

- `synthetic_thermal_series.py`: 위성이 발사장을 주기적으로 관측한다고 가정한 합성 열이상
  시계열 생성기. 24시간 주기 베이스라인(태양열 영향 흉내) + 로켓엔진 점화 스파이크(강) +
  정비용 소규모 발열(약, 애매한 경계 사례) 2종의 이상 이벤트를 주입.
- `ir_anomaly_agent.py`: forecast-then-compare 이상탐지 로직. `Forecaster` 인터페이스를
  분리해서 실제 운영용(`ChronosForecaster`, Chronos-2 사용)과 검증용(`MockForecaster`,
  이동평균+표준편차 기반 통계 모델)을 교체 가능하게 설계.

**결론**: 샌드박스에선 huggingface.co 접속이 막혀 `MockForecaster`로 배관(구조)만 검증했고,
사용자 PC에서 실제 `ChronosForecaster`(Chronos-2)로 재검증까지 완료. 사건 단위 탐지율
2/2(100%), 탐지 지연 0스텝 — "zero-shot 시계열 파운데이션 모델로 이상탐지가 실제로 되는가"
라는 핵심 가설 검증 완료. (point-level recall이 낮은 건 1-step-ahead 예측 특성상 정상 —
조기경보 목적엔 오히려 이상적인 동작. 상세 수치는 프로젝트 진행기록 참고.)

## Phase 2 — 멀티에이전트 뼈대 설계 ✅ 완료 (2026-08-18)

### 목표와 설계 방향

Phase 1까지는 "이상탐지 로직 하나"만 검증했습니다. Phase 2의 목표는 이걸 **레이더/CV/SIGINT
전문 에이전트와 나란히, 병렬로 동시에 실행하고, 그 결과를 한 곳에서 종합하는 멀티에이전트
구조**로 만드는 것입니다.

설계 방향은 이전 세션에서 이미 확정된 상태였습니다 (2026-08-14):

> ReAct식 슈퍼바이저(매 턴 "다음에 누굴 부를지" 판단)가 아니라, **매 사이클마다 관련된 모든
> 전문 에이전트를 동시에 병렬 실행(fan-out)한 뒤 그 결과를 종합**하는 구조. 위협 융합은
> 애초에 "이번에 뭘 볼지 골라야 하는 문제"가 아니라 "매번 가진 센서를 전부 동시에 봐야 하는
> 문제"이기 때문.

**비유**: coordinator는 "상황실장", 각 전문 에이전트는 "각 분야 담당관". 상황실장이 매
사이클마다 담당관 전원에게 동시에 보고를 요청하고(fan-out), 전원의 보고가 다 들어오면
(join) 그걸 종합해서 하나의 SITREP을 쓴다 — "한 명씩 순서대로 물어보는" 방식이 아니다.

### 구조

![아키텍처 다이어그램](docs/architecture.png)

```
                 ┌─────────────┐
                 │    START    │
                 └──────┬──────┘
                        │ dispatch_specialists()
                        │ (이번 사이클 관측 데이터에 있는 센서만 골라서 Send)
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
  radar_agent       cv_agent       sigint_agent    ir_anomaly_agent
   (레이더/항적)     (EO/IR 영상)      (신호정보)     (Phase 1 로직 재사용)
        │               │               │               │
        └───────────────┴───────┬───────┴───────────────┘
                                 ▼
                           coordinator
                        (종합 위협평가 + SITREP)
                                 │
                                 ▼
                               END
```

- `agent/state.py` — 공유 상태(`GraphState`) 정의. 핵심은
  `specialist_reports: Annotated[list[SpecialistReport], operator.add]` —
  여러 에이전트가 동시에 결과를 내놔도 서로 덮어쓰지 않고 리스트에 누적되게 하는 리듀서.
- `agent/specialists.py` — 전문 에이전트 4종의 노드 함수. radar/cv/sigint는 아직 목(mock)
  로직(다음 단계에서 실제 API 연동으로 교체 예정), `ir_anomaly_agent`만 Phase 1에서 검증한
  진짜 로직(`ir_anomaly_agent.detect_anomaly`)을 그대로 가져다 씀.
- `agent/coordinator.py` — "상황실장" 노드. 신뢰도 가중평균 + CRITICAL 오버라이드 2종으로
  종합 판단.
- `agent/graph.py` — `Send` API로 START에서 관련 전문 에이전트에게만 동적으로 fan-out.
  전 센서가 침묵인 사이클은 coordinator로 직행해서 안전하게 "보고 없음" 처리.

### 왜 Send API인가 (일반 조건부 엣지와 차이)

일반적인 조건부 엣지는 "다음 노드 하나의 이름"을 리턴합니다. `Send`를 쓰면 **노드 이름 +
그 노드에 넘길 입력값의 리스트**를 리턴할 수 있어서, "이 4개 노드를 동시에, 각자 다른
입력으로 실행해라"가 가능해집니다. LangGraph는 이 리스트를 보고 병렬 실행 후, 전부 끝나면
공통 다음 노드(coordinator)로 자동으로 합류(join)시켜줍니다 — "몇 명이 끝났는지 직접 세는
코드"를 짤 필요가 없습니다.

### 코드 리뷰 예정

다음 세션에서 진행기록 메모 원칙대로, Phase 2 코드 리뷰(퀴즈 형식)를 진행할 예정입니다.

### 검증 결과 (`test_multi_agent_skeleton.py`, 전부 순수 로직 검증 — 네트워크 불필요)

5개 시나리오, 10개 검증 항목 전부 통과:

| 시나리오 | 확인한 것 | 결과 |
|---|---|---|
| 평시 사이클 | 4개 전문 에이전트 전원 호출, 낮은 점수 | LOW (7.0점) |
| 로켓엔진 점화 사이클 | 열이상 WATCH + 미확인·보호구역 내부 트랙 + SIGINT 이상신호 동시 발생 | **CRITICAL (95점, 오버라이드②)** |
| 애매한 사이클 | 약한 CV 탐지 1건뿐 | LOW (7.0점), cv_agent 1건만 호출 |
| 전 센서 침묵 | 아무 센서 데이터 없음 → 아무도 호출 안 됨 | UNKNOWN, "보고 없음" 안전 처리 |
| 레이더만 존재 | 관측 데이터에 있는 센서만 동적으로 fan-out | radar_agent 1건만 호출 확인 |

### 설계 중 새로 발견한 문제와 보완 (실제 실행 중 발견 — 기록 남김)

**1. 코너 케이스: "약한 신호 여러 개"가 "강한 신호 하나"보다 못 잡히는 문제**

로켓엔진 점화 사이클을 테스트하다가, sitrep_fusion_agent의 CRITICAL 오버라이드 규칙
(`confidence>=0.8 and score>=85`인 에이전트가 하나라도 있으면 강제 CRITICAL)을 그대로
가져왔더니 실제로는 **아무도 이 조건을 단독으로 못 채워서 HIGH(64.7점)에 머무는** 상황이
나왔습니다. SIGINT(90점, 신뢰도 0.6), 레이더(70점, 신뢰도 0.7), IR(55점, 신뢰도 0.6) 세
센서가 "따로따로는 확신 없지만 다 같은 방향"을 가리키고 있는데, 가중평균이 그 신호를
희석시켜버린 겁니다.

이건 실제 다중 센서 융합에서 중요한 지점입니다 — 서로 독립적인 센서 3종이 우연히 동시에
같은 결론에 도달했다는 사실 자체가, 하나의 센서가 어쩌다 튄 것보다 훨씬 신뢰할 수 있는
증거입니다. 그래서 **오버라이드 규칙②**(독립 센서 3개 이상이 각자 50점 이상을 동시에
보고하면 CRITICAL 강제)를 추가했습니다. `agent/coordinator.py`에 근거 주석 포함.

**2. 버그: 전 센서가 침묵하면 그래프가 coordinator까지 도달하지 못하고 조용히 끝나버림**

`Send` 기반 fan-out은 "보낼 대상이 있을 때"만 동작합니다. 이번 사이클에 센서 데이터가
하나도 없어서 `dispatch_specialists()`가 빈 리스트를 리턴하면, START에서 나가는 엣지
자체가 하나도 발동하지 않아 그래프가 **아무 것도 안 하고 끝나버리는** 걸 테스트 중 실제로
확인했습니다 (`final_assessment`가 `None`인 채로 리턴 → 이후 코드에서 크래시). 전 센서
장애·통신 두절 같은 실제로 일어날 수 있는 상황인데 조용히 무시되는 건 위험한 실패
방식이라, "보낼 대상이 없으면 coordinator로 직행해서 명시적으로 '보고 없음'을 남긴다"는
분기를 추가해서 해결했습니다 (`agent/graph.py`).

**3. 오버라이드②의 문턱(3개)을 2개로 낮추면 안 되는 이유 (2026-08-18, `experiment_corroboration_threshold.py`로 실측)**

"독립 센서 3개 이상 동시 보고 → CRITICAL" 규칙에서, 문턱을 2개로 낮추면 얼마나 위험해지는지
실제로 시뮬레이션했습니다. 실제 위협이 전혀 없는 "평범한 배경 소음" 상황(민간 트랙, 흔한
RF 신호, 정상 열이상 패턴)을 5,000번 무작위 생성해서 4개 에이전트를 그대로 돌리고, 오탐
CRITICAL 비율을 쟀습니다.

| 문턱 | 오탐 CRITICAL 비율 | 30초 폴링 기준 하루 예상 오탐 |
|---|---|---|
| 3개 (현재 설계) | 8.76% | 약 252건/일 |
| 2개 (가정) | 18.04% | 약 520건/일 |

2배 이상 늘어나는 이유를 에이전트별로 뜯어보니 근본 원인이 드러났습니다 — 배경 소음
상황에서 "50점 이상"을 잘못 보고하는 비율이 `radar` 4.6%, `cv` 0.8%인 반면 `sigint`는
**47.3%**, `ir_anomaly`(Mock)는 **28.3%**로 유독 높았습니다. `sigint_agent`가 "신호가
세면 이상"이라는 지나치게 단순한 규칙이라 흔한 방송·통신 신호도 절반 가까이 50점을
넘겨버리고, `MockForecaster`는 Phase 1에서 이미 알려진 대로 24시간 주기 패턴을 못
따라가서 정상 패턴에도 잦은 WATCH를 냅니다. 문턱을 2개로 낮추면 사실상 **"가장 시끄러운
두 센서(sigint, ir_anomaly)가 우연히 같이 튀었는가"를 감지하는 규칙**으로 전락합니다 —
독립 증거 교차 확인이라는 오버라이드②의 설계 취지가 무너지는 셈입니다. 3개 문턱도
8.76%면 이미 높은 편이라, `sigint_agent`의 스코어링 로직 자체를 다음 단계에서 손볼
필요가 있다고 판단했습니다.

**3-1. `sigint_agent` 스코어링 로직 수정 및 재검증 (2026-08-18, 같은 세션에서 바로 진행)**

"실제 센서로 바꾸느냐"와는 무관하게 지금 로직 자체가 틀렸다고 판단해서, 실제 API 연동을
기다리지 않고 바로 고쳤습니다. 핵심 변경: 판단 근거를 세기(`strength_db`) 중심에서
`note`(신호처리 단계에서 실제로 "이상 신호"라고 플래그된 경우) 중심으로 뒤집었습니다.
`note`가 없으면 아무리 신호가 세도 최대 25점까지만 올라가서 corroboration 문턱(50점)을
단독으로 못 넘게 만들었습니다 (`agent/specialists.py`). `test_multi_agent_skeleton.py`에
회귀 테스트(세지만 미플래그된 신호가 50점 미만인지)를 추가해서 총 12개 검증 전부 통과
확인했습니다.

같은 5,000회 실험을 다시 돌려서 효과를 확인했습니다:

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| `sigint`가 배경 소음에서 50점 이상 오판하는 비율 | 47.3% | **0.0%** |
| 오버라이드② 문턱=3개일 때 오탐 CRITICAL | 8.76% | 8.40% |
| 오버라이드② 문턱=2개일 때 오탐 CRITICAL | 18.04% | 9.18% |
| 문턱 3개→2개로 낮췄을 때 오탐 배율 | 2.1배 | **1.1배** |

sigint 자체의 문제는 완전히 해결됐고(47.3%→0%), 문턱을 2개로 낮췄을 때의 "추가" 위험도
2.1배→1.1배로 크게 줄었습니다 — corroboration 문턱이 이제 sigint의 잡음에 흔들리지
않는다는 뜻입니다.

그런데 3개 문턱 기준 전체 오탐률(8.76%→8.40%)은 거의 안 줄었습니다. 왜 그런지 한 단계
더 파봤더니 예상 밖의 사실이 나왔습니다 — **애초에 오탐 CRITICAL의 99.8%는 corroboration
(오버라이드②)이 아니라 단일 에이전트 오버라이드(오버라이드①)에서 나오고 있었습니다.**
`ir_anomaly`가 `MockForecaster`의 한계로 순수 정상 패턴에서도 이따금 완전한 ANOMALY(98%
구간 이탈, 90점·신뢰도 0.85)를 내는데, 이 하나만으로 "신뢰도 0.8 이상 + 85점 이상" 단일
오버라이드 조건을 충족해버립니다. 즉 지금까지 "코너 케이스 corroboration 문제"라고
불렀던 8%대 오탐의 실체는 대부분 **corroboration과 무관하게, `MockForecaster` 하나의
과민 반응**이었다는 뜻입니다. 이건 이미 알려진 한계(Mock의 주기성 미학습 문제)라서
새로 고칠 필요는 없고, 실제 운영에서 `ChronosForecaster`로 바꾸면 상당 부분 해결될
것으로 예상되지만 — Phase 1에서 측정한 12.9%는 WATCH+ANOMALY를 합친 수치라 "단독
오버라이드를 얼마나 유발하는 완전한 ANOMALY만의 비율"은 아직 따로 측정한 적이 없습니다.
사용자 PC에서 `ChronosForecaster`로 이 실험을 재현해서 실제 단일-오버라이드 오탐률을
확인하는 게 다음 검증 과제입니다 (아래 "다음 할 일" 참고).

**3-2. 실제 `ChronosForecaster`로 재현 — 가설 확인 (2026-08-18, 사용자 PC에서 실측)**

3-1에서 세운 가설("8%대 오탐의 99.8%는 corroboration이 아니라 `MockForecaster`의 과민
반응")을 실제 Chronos-2로 검증했습니다. `USE_REAL_FORECASTER=1` 환경변수로
`agent.specialists._ir_forecaster`를 실제 `ChronosForecaster`로 바꿔치기해서 같은
실험을 재현(사용자 PC, `amazon/chronos-2` 로컬 추론이라 5,000회는 오래 걸려서 300회로
축소):

| 지표 | Mock (n=5,000) | 실제 Chronos-2 (n=300) |
|---|---|---|
| 오버라이드② 문턱=3개일 때 오탐 CRITICAL | 8.40% | **2.33%** (7건) |
| 오버라이드② 문턱=2개일 때 오탐 CRITICAL | 9.18% | **2.67%** (8건) |
| 문턱 3개→2개 오탐 배율 | 1.1배 | 1.1배 |
| 30초 폴링 기준 하루 예상 오탐(문턱=3개) | 약 242건/일 | 약 67.2건/일 |

가설대로 실제 모델을 쓰니 오탐률이 **8.40% → 2.33%로 약 3.6배 줄었습니다** — 3-1에서
찾아낸 원인(Mock이 24시간 주기 패턴을 못 따라가서 정상 상황에도 완전한 ANOMALY를
잘못 내는 것)이 맞았다는 뜻입니다. 문턱을 2개로 낮췄을 때의 "추가" 배율(1.1배)은
Mock과 실제 모델에서 동일하게 나와서, 이 부분은 forecaster 종류와 무관하게 안정적인
결과로 보입니다.

다만 두 가지는 유보적으로 봐야 합니다. 첫째, 표본 크기가 300회(Mock은 5,000회)라
통계적으로 더 거칠고, 둘째 2.33%(하루 약 67건)도 절대적으로 낮은 수치는 아닙니다 —
Mock 대비 크게 개선됐을 뿐, "실전에 그대로 써도 될 만큼 낮다"고 단정하긴 이릅니다.
그래도 이번 실험은 **애초에 세운 가설(문제의 근원이 corroboration 로직이 아니라
forecaster 선택이었다)을 직접 실측으로 뒷받침했다는 점에서 의미가 있습니다.**

## Phase 2.5 — radar/cv 실제 연동 (2026-08-18)

### 무엇을 했나

목(mock) 로직이었던 `radar_agent`/`cv_agent`를 sitrep_fusion_agent에서 이미 검증된
실제 연동 코드로 교체했습니다. `sigint_agent`처럼 스코어링 로직만 손본 게 아니라,
**실제 데이터 소스(OpenSky API, YOLO26-OBB 모델) 자체를 이 프로젝트에 옮겨왔다**는
점이 다릅니다.

- `config.py`, `fusion/geofence.py`, `fusion/identification.py`,
  `data_sources/flight_tracker.py`, `data_sources/cv_detection.py` — sitrep_fusion_agent
  에서 그대로 포팅 (검증된 코드 재사용, 새로 작성 안 함).
- `agent/observation.py` — **신규 모듈**. Phase 1의 Forecaster 인터페이스 분리와 같은
  설계 원칙을 적용: "원본 데이터 → 우리 포맷 변환" 로직(`radar_track_to_dict`,
  `cv_detection_to_dict`, 순수 함수)과 "실제 API·모델을 호출하는" 로직
  (`collect_radar_observation`, `collect_cv_observation`)을 분리했습니다. 앞엣것은
  샌드박스에서 네트워크 없이 완전히 검증 가능하고, 뒤엣것은 사용자 PC에서만 검증
  가능합니다.

### 실행 중 발견한 문제: CV 클래스 taxonomy 불일치

포팅하다가 실제로 실행해보진 않았지만 코드를 뜯어보는 과정에서, 기존 `cv_agent`의
목 로직이 `"military-vehicle"`, `"warship"`, `"missile-launcher"`처럼 **임의로 지어낸
클래스명**으로 점수를 매기고 있었다는 걸 발견했습니다. 그런데 실제 YOLO26-OBB 모델은
DOTA-v1.0 데이터셋의 15개 클래스(plane, ship, storage-tank, large-vehicle 등)만
출력하고, 여기엔 "군용/민간" 구분이 아예 없습니다 — 즉 실제 모델을 그대로 연결했으면
`cv_agent`가 **항상 점수 0에 가깝게만 나오는** 조용한 실패가 발생할 뻔했습니다.

`agent/observation.py`에 실제 taxonomy 기준 `CV_HIGH_CONCERN_CLASSES`
(plane/helicopter/ship/large-vehicle — 위협 판단에 더 의미 있을 수 있는 플랫폼류)와
`CV_LOW_CONCERN_CLASSES`(small-vehicle/harbor/storage-tank/bridge/roundabout — 민간
인프라류)를 정의하고, `cv_agent`를 3단계 점수화(고위험 60점·저위험 15점·무관 5점,
전부 신뢰도 곱)로 다시 작성했습니다. `fusion/identification.py`와 같은 한계가
그대로 적용됩니다 — "plane"이 여객기인지 전투기인지는 이 모델이 모르므로, 위치(보호구역
근접도)와 다른 센서와의 corroboration으로 보완해야 하는 약한 신호로 취급합니다.

### 검증 결과

| 검증 파일 | 범위 | 결과 |
|---|---|---|
| `test_multi_agent_skeleton.py` | cv_agent 교체 후 기존 5개 시나리오 전부 재실행 (테스트 데이터도 `"military-vehicle"`→`"large-vehicle"` 실제 클래스명으로 갱신) | 12/12 통과 |
| `test_observation.py` (신규) | `radar_track_to_dict`/`cv_detection_to_dict` 순수 변환 함수 — 위경도 결측, 보호구역 내부/외부, 호출부호 유무, 고도·속도 결측 등 | 12/12 통과 |

**샌드박스에서 검증 못 한 부분** (huggingface.co와 동일한 정책으로
opensky-network.org도 차단됨, `curl` 테스트로 확인):

- `collect_radar_observation()` — 실제 OpenSky API 응답을 받아오는 부분
- `collect_cv_observation()` — 실제 YOLO26-OBB 모델로 이미지를 추론하는 부분 (모델
  가중치 파일 21.5MB도 device_commit_files 20MB 제한 때문에 이 세션에서 직접 전송 못 함)

이 두 가지는 사용자 PC(인터넷 O)에서만 확인 가능합니다.

**[2026-08-18 추가] 사용자 PC 검증 완료**: 모델 파일 복사, `.env` 설정, `pip install -r
requirements.txt` 이후 `collect_radar_observation()`을 실제로 호출해서 OpenSky API
연동까지 확인 완료. `models/yolo26s_obb_dota_best.pt`를 포함한 전체 변경사항을
`git add -f models/yolo26s_obb_dota_best.pt`(`.gitignore`의 `*.pt` 규칙 — 원래
Chronos-2 같은 대용량 캐시 방지용 — 때문에 이 파인튜닝 모델만 예외적으로 강제 추가)로
커밋·push 완료. radar/cv 실제 연동 작업(Phase 2.5) 전체 마무리.

**[2026-08-18 추가] `collect_cv_observation()` 실제 이미지 검증 — 버그 1건 발견·수정**:
사용자 PC에서 실제 파인튜닝 모델(`yolo26s_obb_dota_best.pt`)로 처음 테스트했을 때
빈 리스트만 나오는 문제가 있었다. 원인은 `detect_objects()`가 추론 해상도(`imgsz`)를
명시하지 않아 ultralytics 기본값(640)으로 돌고 있었던 것 — 이 모델은 `imgsz=1024`로
파인튜닝됐는데(학습 로그 `args.yaml` 확인) 학습·추론 해상도가 다르면 특히 항구·선박
같은 가늘고 긴 객체의 재현율이 크게 떨어진다. `data_sources/cv_detection.py`의
`detect_objects()`에 `imgsz: int = 1024` 기본값을 명시해서 해결.

수정 후 실제 DOTA 검증 이미지(`runs/.../val_batch1_pred.jpg`)로 재테스트한 결과, 모델이
실제 DOTA 클래스(`ship`, `swimming pool`)를 정상적으로 출력하고 지리참조 계산까지
전체 파이프라인이 크래시 없이 동작함을 확인했다. **다만 이 테스트 이미지는 조건이
불리했다** — 원본 위성 타일이 아니라 검증 과정에서 2장을 이어붙이고 레터박스(회색
여백)를 넣은 뒤 그 위에 예전 예측 박스·텍스트까지 겹쳐 그린 시각화 산출물이라,
탐지 신뢰도가 전부 0.13~0.33으로 낮게 나와서 운영 기본 임계값(`conf_threshold=0.4`)
에서는 여전히 빈 리스트가 나왔다. 즉 **파이프라인 자체(추론→지리참조→변환)의 정상
동작은 검증됐지만, "운영 기본 임계값에서도 실전처럼 잘 탐지되는가"는 이번 테스트
이미지의 한계 때문에 완전히 증명하지 못했다** — 원본 위성/드론 이미지라면 학습 시
측정한 mAP50=0.628 수준의 신뢰도가 나올 것으로 예상되지만, 깨끗한 원본 이미지로
재확인하는 건 다음 세션 과제로 남긴다.

## Phase 2.6 — LLM 기반 SITREP 생성 완료 (2026-08-18)

coordinator는 일부러 규칙 기반 템플릿 텍스트(`state["sitrep"]`)만 만들고 멈춰뒀습니다
— "여러 에이전트 판단을 하나로 합치는 로직 자체"를 검증하는 게 그 노드의 목적이라,
LLM 호출까지 섞으면 무엇을 검증하는지 흐려지기 때문입니다. 이 단계에서 coordinator
다음에 실행되는 별도 노드 `agent/brief.py`의 `generate_sitrep`을 추가해서, 규칙 기반
요약을 원재료 삼아 Claude API(`langchain-anthropic`, `claude-sonnet-5`)로 지휘관
브리핑처럼 읽히는 자연어 문장을 만들도록 확장했습니다. sitrep_fusion_agent의
`generate_brief` 노드와 같은 패턴 — "종합 판단"과 "그 판단을 사람이 읽기 좋게 다듬는
것"을 서로 다른 책임으로 분리했습니다.

```
... -> coordinator (규칙 기반 종합, state["sitrep"]) -> generate_sitrep (Claude API로
자연어 브리핑, state["natural_language_brief"]) -> END
```

**안전장치**: LLM 호출은 API 키 미설정·네트워크 문제·요금 한도 등으로 언제든 실패할
수 있다고 가정하고, 실패해도 그래프 전체가 죽지 않도록 규칙 기반 원시 요약으로
안전하게 폴백하게 설계했습니다(`generate_sitrep`의 `try/except`). 전문 에이전트가
하나도 호출되지 않은 사이클(전 센서 침묵)은 애초에 LLM을 부를 필요가 없으므로 호출
자체를 건너뛰고 원시 요약을 그대로 씁니다(불필요한 API 비용 방지).

**검증 결과**: `test_multi_agent_skeleton.py`에 검증 2건 추가, 총 14/14 통과.
**[샌드박스 검증 범위]** 이 세션엔 `ANTHROPIC_API_KEY`가 없어서 실제 LLM 호출
자체는 매번 `Anthropic authentication failed`로 실패했습니다 — 이건 예상된
결과이고, 오히려 "API 키가 없을 때 폴백 경로가 안전하게 동작하는가"(그래프가 안
죽고, 로그만 남기고, 원시 요약으로 대체되는가)를 검증한 셈입니다. **실제 자연어
브리핑 품질(문장이 실제로 지휘관 보고처럼 읽히는가)은 사용자 PC(.env에
`ANTHROPIC_API_KEY` 설정됨)에서 확인 필요** — 아래 "다음 할 일" 참고.

### 알려진 한계 (다음 단계에서 다룰 것)

- 상태 지속성(트랙 이력)·경로 예측·분석관 승인(interrupt)·감사 로그는 아직 이
  프로젝트에 없음 — sitrep_fusion_agent의 Phase 3~5 자산을 이 멀티에이전트 구조
  위에 어떻게 다시 얹을지는 다음 단계 논의 필요.
- **[2026-08-18 추가, 구조적 한계] `cv_agent`는 태생적으로 "군용/민간" 또는 "함종·기종"
  세부 식별을 못 함** — 코드 버그가 아니라 지도학습 탐지 모델의 근본 제약. YOLO는 학습
  라벨에 없는 클래스는 절대 출력할 수 없고(closed-set), DOTA-v1.0(공개 학술용 항공/위성
  데이터셋)엔 battleship·submarine·전투기 기종 같은 군사 세부 유형 라벨이 아예 없음 —
  이런 라벨은 기밀 정찰 영상 + 전문가 라벨링이 필요해서 공개 데이터셋으로 나올 수가
  없기 때문. (참고: 선박 세부분류는 HRSC2016·FGSCR-42 같은 공개 데이터셋이 존재해서
  "1단계 탐지 → 2단계 세부분류" 파이프라인으로 개선 여지가 있음 — 항공기 쪽은 공개
  세부분류 데이터가 훨씬 드묾. 지금은 백로그로만 남겨둠.) 이 한계는
  `fusion/identification.py`가 FRIEND/HOSTILE을 절대 안 내고 NEUTRAL/UNKNOWN까지만
  판정하는 것과 같은 설계 철학으로 대응 — 센서가 실제로 아는 것보다 더 정밀한 척하지
  않고, 대신 위치(보호구역 근접도)·다른 센서와의 corroboration으로 보완.
- **[2026-08-18 추가, 구조적 한계] `ir_anomaly_agent`는 여전히 합성 열이상 데이터로만
  검증됨 — NASA FIRMS(VIIRS/MODIS) 실데이터 연동을 검토했으나 보류.** 기술적으론
  [FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)로 연동 가능하지만
  (무료 `MAP_KEY`, 10분당 5,000건 한도), 저궤도 위성(VIIRS/MODIS)은 특정 지점을 하루
  1~4회 정도만 지나가는 반면 `ir_anomaly_agent`는 "매 사이클 꾸준히 관측값이 들어온다"는
  전제로 Chronos-2 롤링 시계열 예측을 하도록 설계돼 있어 — 짧은 이벤트(로켓엔진 점화
  등)는 위성이 마침 그 순간을 지나가지 않으면 통째로 놓칠 수 있음. 이건 저궤도 관측위성
  기반 조기경보의 실제 한계이지 설계 결함이 아니며(실전에서는 정지궤도 적외선 위성을
  따로 씀), 지금은 이 한계 자체를 명시하고 합성 데이터를 계속 쓰기로 결정 — 전면 재설계
  (이력 기반 구조로 재작성, Phase 3 자산 재적용과도 얽힘)는 스코프 밖으로 판단.
  **[추가 확인]** `synthetic_thermal_series.py`가 가정하는 "10분마다 연속 관측"은 사실
  FIRMS(저궤도)가 아니라 **정지궤도 위성의 관측 패턴을 모사한 것**임을 명확히 함 —
  실제로 [NOAA GOES-R 시리즈의 Fire/Hot Spot Characterization(FDC) 제품](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_FDCC)이
  정확히 10분 간격으로 갱신되는 무료 공개 데이터이고, 한반도 권역은 기상청 천리안위성
  2A호(GK-2A)가 같은 방식(정지궤도)으로 ["산불탐지(FF)"](https://nmsc.kma.go.kr/resources/common/pdf/%EC%99%B8GK2A_L2_ATBD_%EA%B5%AD%EB%AC%B8_%EC%82%B0%EB%B6%88%ED%83%90%EC%A7%80_FF.pdf)
  제품을 만들고 있음을 확인했음. 즉 "실데이터로 못 바꾼다"가 아니라 "저궤도(FIRMS)
  대신 정지궤도(GOES/GK-2A) 데이터를 써야 지금 설계와 맞는다"가 더 정확한 결론 — 다만
  GK-2A 쪽 접근 방법·인증·요청 제한은 아직 조사 안 함, 실데이터 전환은 여전히 백로그.

### 다음 할 일

1. ~~사용자 PC에서 `test_multi_agent_skeleton.py` 재실행해서 동일 결과 재확인~~ ✅ 완료 (2026-08-18, GitHub push까지 완료)
2. ~~`sigint_agent` 스코어링 로직 수정 (세기 중심 → note 중심)~~ ✅ 완료 (2026-08-18, 배경 소음 오판율 47.3%→0%)
3. ~~`docs/architecture.png` 배치~~ ✅ 완료 (2026-08-18) — 이전 세션에서 만들었던 이미지가
   대화 압축 과정에서 유실돼 있던 걸 발견, `docs/make_architecture_diagram.py`
   (matplotlib 기반, 재실행하면 최신 구조로 다시 그릴 수 있음)로 Phase 2.5/2.6까지
   반영한 구조로 새로 생성해 `docs/architecture.png`에 저장
4. ~~사용자 PC에서 `experiment_corroboration_threshold.py`를 `ChronosForecaster`
   기준으로 재현~~ ✅ 완료 (2026-08-18) — 오탐 CRITICAL이 Mock 8.40% → 실제 Chronos-2
   2.33%(n=300)로 약 3.6배 감소, 3-1에서 세운 가설(Mock 과민반응이 원인) 실측 확인.
   상세는 위 "3-2" 참고
5. ~~radar/cv 목(mock) 로직을 sitrep_fusion_agent의 실제 연동 코드로 교체~~
   ✅ 완료 (2026-08-18, Phase 2.5)
6. ~~사용자 PC에서 radar/cv 실제 연동 검증~~ ✅ 완료 (2026-08-18) — `collect_radar_observation()`은
   실제 OpenSky API로 항적 수신까지 확인. `collect_cv_observation()`은 실제 이미지로
   테스트하다가 `imgsz` 불일치 버그(추론 640 vs 학습 1024)를 발견·수정했고, 수정 후
   파이프라인(추론→지리참조→변환) 자체는 정상 동작 확인 — 다만 테스트 이미지가
   검증용 시각화 산출물(레터박스+겹쳐그린 텍스트)이라 신뢰도가 낮게 나와서, 운영
   기본 임계값(0.4)에서도 실전처럼 잘 잡히는지는 **원본 위성/드론 이미지로 재확인
   필요**(다음 세션 과제로 이월). `models/yolo26s_obb_dota_best.pt` 포함 GitHub push까지 완료
7. ~~NASA FIRMS 등 실제 공개 열이상 데이터로 합성 데이터를 대체할지 검토~~ ✅ 검토 완료,
   **보류로 결정** (2026-08-18) — [FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)
   조사 결과, 무료 `MAP_KEY` 발급 필요·10분당 5,000건 한도·`bbox`+최대 5일 쿼리로
   기술적으론 연동 가능하지만, **위성 재방문 주기(revisit gap) 문제**가 지금 설계와
   정면충돌함을 확인. VIIRS/MODIS 같은 저궤도 위성은 특정 지점을 하루 1~4회 정도만
   지나가는데, `ir_anomaly_agent`는 "매 사이클 꾸준히 관측값이 들어온다"는 전제로
   Chronos-2 롤링 시계열 예측을 하도록 설계돼 있어서 — 로켓엔진 점화처럼 짧은 이벤트는
   위성이 마침 지나가지 않으면 통째로 놓칠 수 있음. (이건 설계 결함이 아니라 저궤도
   관측위성 기반 조기경보의 실제 한계 — 그래서 실전 조기경보 체계는 정지궤도 적외선
   위성(SBIRS 등)을 따로 씀.) 전면 재설계(이력 기반 구조, Phase 3 자산 재적용과도 얽힘)는
   지금 스코프 밖으로 판단, 합성 데이터를 계속 쓰고 이 한계를 "왜 합성 데이터로
   검증했는가"의 근거로 아래 "알려진 한계"에 기록만 함
8. ~~coordinator에 LLM 기반 SITREP 생성 붙일지 결정~~ ✅ 완료 (2026-08-18, Phase 2.6 —
   `agent/brief.py`의 `generate_sitrep` 노드 추가, 14/14 검증 통과). **[2026-08-18 추가]**
   사용자에게 실제 발급받은 `ANTHROPIC_API_KEY`가 아직 없음을 확인 — PC에서 재실행해도
   지금은 폴백 경로(`[LLM 호출 실패...]`)로만 동작함. 키 발급 후 `.env`에 등록하고
   `python test_multi_agent_skeleton.py`를 재실행해서 `natural_language_brief`에 실제
   Claude가 쓴 자연어 브리핑이 담기는지 확인하는 게 남은 검증 과제
