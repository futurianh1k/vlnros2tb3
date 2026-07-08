# PRD 리뷰: 차세대 VLN 플랫폼

## 0. 목적

첨부된 PRD(`차세대 비전-언어 내비게이션(VLN) 플랫폼 구축을 위한 제품 요구사항 문서`)를 현재
저장소(`vln_platform/`)의 구현 상태와 대조하여, 어느 요구사항이 충족되었고 어느 부분이
비어 있는지, 그리고 PRD가 지정한 스펙과 실제 코드가 어디서 갈라졌는지를 정리한다.

## 1. 현재 코드가 커버하는 범위 요약

현재 구현은 **Phase 1을 건너뛰고 Phase 2(비디오 스트림 + VLM 기반 엔진)의 스캐폴딩만
부분적으로** 갖추고 있다. 모듈 대응은 다음과 같다.

| PRD 요구사항 | 대응 모듈 | 상태 |
| --- | --- | --- |
| 3.2 스트리밍 비디오 버퍼 (링 버퍼, 5~10fps) | `src/core/buffer.py: StreamBufferSystem` | 부분 구현 |
| 3.2 공간 씬 그래프 빌더 | `src/core/scene_graph.py: SpatialSceneGraphBuilder` | 부분 구현 |
| 3.2 VLM/LLM 추론 레이어 | `src/accelerators/quantization.py: QuantizedInferenceBackend` | **미구현 (키워드 매칭 목업)** |
| 3.2 행동 토큰 디코더 | `src/schemas/data_models.py: ActionOutput` + `agent.py: _decode_action` | 구현 (스키마 수준) |
| 4.1 200ms 지연 제약 | `agent.py: compute_next_action`의 `elapsed_ms` 체크 | 구현 |
| 4.1 INT4/FP8 양자화 인터페이스 | `accelerators/quantization.py: QuantizationConfig` | **인터페이스만 존재, 실제 적용 안 됨** |
| 4.2 시뮬레이터 추상화(Gym 스타일) | 없음 | **미구현** |
| 4.3 `UnifiedNavigationAgent` 템플릿 | `src/core/agent.py` | 구현되었으나 시그니처가 PRD 스펙과 다름 |
| 5. 세이프가드 가로채기 레이어 | `src/utils/safety.py: SafetyInterceptor` | 구현 |
| 5. LoRA sim-to-real 파인튜닝 | 없음 | 미구현 (Phase 3 대상, 예정대로 보류) |
| 3.1 Matterport3D/DUET (Phase 1 전체) | 없음 | **완전 미구현** |

## 2. 주요 갭 및 리스크

### 2.1 Phase 1(레거시 검증)이 통째로 생략됨

PRD 로드맵은 Phase 1(Matterport3D + DUET)을 **P0 필수**로, Phase 2보다 먼저 배치했다.
현재 코드는 Phase 1 산출물 없이 곧바로 Phase 2 스타일의 에이전트 골격을 구현했다.
이는 두 가지로 해석 가능하다.

- (a) 의도적으로 Phase 1을 스킵하고 Phase 2 아키텍처를 먼저 검증하는 전략적 결정이었다면,
  PRD의 우선순위 표와 배치되므로 이해관계자 합의가 필요하다.
- (b) 단순 누락이라면, Dual-Scale Graph Transformer / 토폴로지 맵핑 / SR·SPL·NE 평가
  로깅 등 Phase 1 산출물을 별도 트랙으로 채워 넣어야 한다.

**권고:** 실제 로봇 하드웨어(TurtleBot3) 대상 프로젝트임을 감안하면 Matterport3D 이산
그래프 기반 시뮬레이터 검증보다 Phase 2/3 파이프라인이 더 실효성이 높다. Phase 1을
"참고 벤치마크"로 축소하고 Phase 2를 주력으로 진행하는 쪽을 권장하되, 이 판단은
PRD 오너의 명시적 승인을 받아야 한다.

### 2.2 VLM/LLM 추론 레이어가 실제로 존재하지 않음

`QuantizedInferenceBackend.forward()`는 지시문 문자열에서 `"left"`, `"go"`, `"desk"` 같은
키워드를 매칭해 고정된 속도값을 반환하는 규칙 기반 목업이다. PRD 3.2가 요구하는
"LLaVA 또는 InternVL 계열 오픈소스 경량 파운데이션 모델 백본"은 전혀 로드/추론되지
않는다. 현재는 인터페이스 계약(입력: tokens/frames/odometry, 출력: action dict)을
정의한 자리표시자 수준이며, 실제 모델 통합이 프로젝트의 핵심 미해결 과제다.

### 2.3 양자화 설정이 실질적으로 죽어 있음 (기존 코드 리뷰에서 확인된 버그)

`agent_config.yaml`의 `quantization.precision/backend/enabled` 값은 `AgentConfig`가
파싱하지 않고, `QuantizationConfig`는 항상 기본값(`fp8`/`vllm`)으로 생성된다. PRD
4.1이 요구하는 "INT4/FP8 양자화 모델 구동 인터페이스"는 설정 파일 상으로만 존재하고
실제로 전환 가능한 스위치가 아니다. VLM 백본 통합과 함께 반드시 재작업이 필요하다.

### 2.4 시뮬레이터 추상화 레이어 부재

PRD 4.2는 "Matterport3D, Habitat-Sim, Isaac Sim 등 물리 하위 엔진의 규격 변화에
종속되지 않는 통일된 Gym 스타일 인터페이스"를 명시적으로 요구한다. 현재 저장소에는
시뮬레이터 관련 코드가 전혀 없다. `UnifiedNavigationAgent`가 `ingest_frame`/
`ingest_odometry`로 데이터를 받는 구조 자체는 이 추상화와 호환 가능하지만, 실제
`gym.Env` 서브클래스나 시뮬레이터 어댑터는 아직 없다.

### 2.5 `UnifiedNavigationAgent`가 PRD 4.3 템플릿과 시그니처가 다름

| 항목 | PRD 스펙 | 현재 구현 |
| --- | --- | --- |
| 생성자 | `__init__(self, config_path: str)` 후 `self.backbone`, `self.spatial_scene_graph = {}` | `config_path` + 4개 선택적 DI 파라미터(`buffer_system`, `scene_graph`, `inference_backend`, `safety_interceptor`) |
| 프레임/오도메트리 갱신 | `update_stream_buffer(video_frame, odometry_data)` 단일 메서드 | `ingest_frame(frame, metadata)` / `ingest_odometry(odometry)`로 분리 |
| `compute_next_action` 반환 | `dict` | `ActionOutput` (pydantic 모델) |
| 씬 그래프 저장 | `self.spatial_scene_graph = {}` (딕셔너리) | `SpatialSceneGraphBuilder` (networkx 그래프 래퍼) |

현재 구현이 의존성 주입과 스키마 검증을 도입한 것은 테스트 용이성과 타입 안정성
측면에서 PRD의 예시 골격보다 더 견고한 설계다. 다만 PRD를 "강제 규격"으로 명시한
문구(`4.3 강제합니다`)를 감안하면, 이 이탈이 팀 내부에서 의도적으로 승인된 것인지
문서화가 필요하다. 본 리뷰는 **현재 설계를 유지**하고 PRD 문서 쪽을 실제 구현에 맞게
갱신할 것을 권장한다(딕셔너리 대신 그래프 구조, dict 대신 검증된 스키마가 실무적으로
더 안전함).

### 2.6 평가 지표(SR/SPL/NE) 및 대시보드 부재

Phase 1 범위이긴 하나, Phase 2/3에서도 내비게이션 성공률·경로 효율·목표 도달 오차는
회귀 테스트의 핵심 지표가 된다. 현재는 어떤 형태의 메트릭 로깅도 없다.

### 2.7 실제 예외 처리 관련 기존 이슈

이전 코드 리뷰(별도 스레드)에서 지적된 `agent.py`의 광범위한 `except Exception` 및
버퍼 프레임 수 하드코딩(`quantization.py`의 매직 넘버 `5`)은 VLM 백본 통합 시 디버깅을
심각하게 방해할 수 있으므로, Phase 2 본작업 착수 전에 우선 수정 대상으로 포함한다.

## 3. 오픈 퀘스천 (PRD 오너 확인 필요)

1. Phase 1(Matterport3D/DUET)을 실제로 구현할 것인가, 아니면 벤치마크 참고 자료로만
   남기고 Phase 2/3에 집중할 것인가?
2. `UnifiedNavigationAgent`의 PRD 예시 시그니처를 그대로 강제할 것인가, 현재의
   DI 기반 구조를 표준으로 승격할 것인가?
3. VLM 백본은 어떤 모델(LLaVA-1.6, InternVL2, Uni-NaVid 원본 체크포인트 등)을
   1차 통합 대상으로 삼을 것인가? 온보드(TurtleBot3 탑재 컴퓨트) 자원 제약이 모델
   크기 선택에 어떤 상한을 부여하는가?
4. ROS2/TurtleBot3 연동(Phase 3)의 목표 시점은? 시뮬레이터 검증 없이 바로 실물
   하드웨어로 갈 것인지, 중간에 Gazebo/Isaac Sim 시뮬레이션 루프를 넣을 것인지.

이 문서와 함께 제출하는 `docs/implementation-plan.md`는 위 오픈 퀘스천에 대해
**Phase 1은 축소, Phase 2 우선 완성, Phase 3는 ROS2/TurtleBot3 시뮬레이션 우선**이라는
가정 하에 작성되었다. 가정이 다르면 계획을 재조정해야 한다.
