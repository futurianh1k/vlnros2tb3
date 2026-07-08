# 구현 계획: VLN 플랫폼 (Phase 2 우선 완성 기준)

> 전제: `docs/prd-review.md`의 오픈 퀘스천에 대해 "Phase 1은 축소, Phase 2를 주력으로
> 완성, Phase 3는 ROS2/TurtleBot3 연동을 시뮬레이션 우선으로 진행"한다고 가정한다.
> 이 가정이 PRD 오너와 다르면 우선순위 재조정이 필요하다.

## 0. 설계 원칙

- **기존 아키텍처를 계승한다.** `UnifiedNavigationAgent`의 DI 구조(`buffer_system`,
  `scene_graph`, `inference_backend`, `safety_interceptor`)는 유지하고, PRD 4.3의
  예시 스켈레톤(dict 반환, `self.backbone` 단일 속성)으로 되돌리지 않는다 — pydantic
  스키마 검증과 컴포넌트 교체 가능성을 잃기 때문이다.
- **안전 레이어는 항상 최종 출력단에 유지한다.** VLM 백본이 어떤 것으로 바뀌더라도
  `SafetyInterceptor`를 거치지 않고 `ActionOutput`이 나가는 경로가 생기지 않도록 한다.
- **목업 → 실제 구현으로의 교체가 인터페이스 변경 없이 가능해야 한다.** 즉
  `QuantizedInferenceBackend.forward(tokens, frames, odometry) -> dict`의 시그니처는
  실제 VLM 통합 시에도 유지한다.

## 1. 즉시 수정 (Phase 2 착수 전 선행 작업)

기존 코드 리뷰에서 확인된 버그를 먼저 해소한다. VLM 백본 통합 이후에는 디버깅 난이도가
급격히 올라가므로 지금 고친다.

| 작업 | 파일 | 내용 |
| --- | --- | --- |
| 예외 처리 세분화 | `src/core/agent.py` | `except Exception`을 `except (ValidationError, InferenceBackendError)` 등으로 좁히고, 예외 발생 시 로깅 후 emergency stop |
| 양자화 설정 연결 | `src/core/agent.py`, `src/schemas 또는 config` | `AgentConfig`에 `quantization_precision`/`quantization_backend`/`quantization_enabled` 필드 추가, yaml 파싱에 반영, `QuantizationConfig` 생성 시 전달 |
| 매직 넘버 제거 | `src/accelerators/quantization.py` | `frames.shape[0] >= 5` → `forward()`에 `min_context_frames` 파라미터로 받거나 `QuantizationConfig`에 노출 |
| `_load_config` 중복 제거 | `src/core/agent.py` | yaml/fallback 두 분기가 공유하는 `_build_agent_config(payload: dict) -> AgentConfig` 헬퍼로 통합 |

완료 기준: 기존 테스트(`test_agent.py`, `test_buffer.py`) 통과 + 새 유닛 테스트로
quantization 설정 반영을 검증.

## 2. Phase 2 본작업 — VLM 기반 엔드투엔드 엔진 완성

### 2.1 VLM/LLM 추론 백본 통합 (최우선, 최대 리스크)

- `QuantizedInferenceBackend`를 실제 모델 로딩이 가능한 어댑터로 확장한다.
  - 신규 모듈 `src/accelerators/backbones/`: `LlavaBackbone`, `InternVLBackbone` 등
    모델별 어댑터를 두고, `QuantizedInferenceBackend`는 이 어댑터들의 공통
    인터페이스(`load()`, `infer(tokens, frames, odometry) -> raw_action_dict`)를
    호출하는 파사드로 리팩터링.
  - 양자화는 어댑터 로드 시점에 `QuantizationConfig.precision`(`fp8`/`int4`)에 따라
    실제 가중치 로딩 방식을 분기(예: `bitsandbytes`, `AutoGPTQ`, 또는 vLLM의 네이티브
    양자화 로더)한다.
  - 현재의 키워드 매칭 목업은 `MockInferenceBackend`로 이름을 바꿔 테스트/CI 전용
    백엔드로 남긴다 (실제 GPU/모델 없이도 에이전트 파이프라인 테스트 가능해야 함).
- 산출물: 실제 오픈소스 체크포인트 1종(예: LLaVA-1.6-7B 또는 InternVL2-2B, 온보드
  자원 제약 확인 후 결정 — PRD 오픈 퀘스천 3 참고)으로 end-to-end 추론이 되는 PoC.

### 2.2 스트리밍 비디오 버퍼 — 인코더 임베딩 캐싱

- 현재 `StreamBufferSystem`은 원본 텐서만 저장한다. PRD 3.2는 "비디오 인코더 토큰
  임베딩을 시계열로 저장"을 요구한다.
- `BufferEntry`에 `embedding: Optional[np.ndarray]` 필드를 추가하고, 백본의 비전
  인코더 forward 이후 결과를 캐싱해 반복 인코딩을 피하는 경로를 추가한다(추론
  지연시간 200ms 제약을 지키기 위한 핵심 최적화이기도 함).
- 5~10fps 처리량 요구사항에 대해 벤치마크 스크립트(`scripts/benchmark_buffer.py`)로
  실측치를 문서화한다.

### 2.3 공간 씬 그래프 — VLM 파이프라인과 연동

- 현재 `SpatialSceneGraphBuilder.add_or_update_node`는 독립적으로 테스트 가능한
  상태지만, 에이전트 루프에서 실제로 호출되는 지점이 없다(VLM이 오브젝트를 감지한
  결과를 씬 그래프에 반영하는 연결부가 미구현).
- `UnifiedNavigationAgent.compute_next_action`에 VLM 백본의 감지 결과(오브젝트
  라벨/좌표)를 `SpatialNode`로 변환해 `scene_graph.add_or_update_node()`를 호출하는
  단계를 추가한다.
- 씬 그래프 상태를 다음 스텝의 프롬프트/컨텍스트에 다시 주입하는 피드백 루프 설계
  (예: "최근 관측된 오브젝트 N개를 텍스트 프롬프트에 포함").

### 2.4 지연시간 및 양자화 검증

- `compute_next_action`의 200ms 체크는 유지하되, 실제 VLM 통합 후 P50/P95 지연시간을
  측정하는 벤치마크를 추가하고 목표 미달 시 양자화/가속 옵션을 조정하는 절차 문서화.

### 2.5 테스트 커버리지 확충

- 현재 `scene_graph.py`, `quantization.py`(전체 forward 로직), `safety.py`는 단위
  테스트가 없다. 아래를 추가한다.
  - `test_scene_graph.py`: 신규 노드 추가, 근접 노드 병합, confidence 블렌딩 검증.
  - `test_safety.py`: `should_stop` 경계값, `enforce_safety` 정상/비상 경로.
  - `test_quantization_backend.py`: (Mock 백엔드 대상) 텍스트 키워드 → action 매핑,
    실제 백본 도입 후에는 어댑터 인터페이스 계약 테스트로 교체.

## 3. Phase 2.5 — 시뮬레이터 추상화 레이어 (PRD 4.2)

Phase 3(실물 하드웨어)로 넘어가기 전에, 실제 로봇 없이도 엔드투엔드 루프를 검증할
시뮬레이터 어댑터가 필요하다.

- `src/simulation/base.py`: `class VLNSimEnv(gym.Env)` 형태의 추상 인터페이스 정의
  (`reset()`, `step(action) -> (frame, odometry, reward, done, info)`).
- 1차 어댑터 대상: **Habitat-Sim** (Matterport3D 자산 재사용 가능, Phase 1 자산을
  최소 비용으로 재활용) 또는 **Isaac Sim**(Phase 3 하드웨어와 물리 엔진 궁합이 더
  좋음). 자원/일정에 따라 팀이 택일 — 본 계획은 Habitat-Sim을 1차로, Isaac Sim을
  Phase 3 직전 전환 대상으로 제안한다.
- `UnifiedNavigationAgent`는 시뮬레이터 종류를 몰라야 하므로, 시뮬레이터 → 에이전트
  입력 변환은 어댑터 레이어에서만 수행한다(에이전트 코드 변경 없음).

## 4. Phase 3 — ROS2/TurtleBot3 연동 (착수 시점 이후 상세화)

이 저장소 이름(`vlnros2tb3`)이 이미 TurtleBot3 + ROS2를 지향하므로, Phase 3는 다음
순서로 진행한다.

1. `src/ros_bridge/` 신규 패키지: ROS2 노드가 카메라 토픽/오도메트리 토픽을 구독해
   `UnifiedNavigationAgent.ingest_frame`/`ingest_odometry`를 호출하고, `ActionOutput`을
   `geometry_msgs/Twist`로 변환해 발행.
2. `SafetyInterceptor`의 `proximity_distance_m` 입력을 실제 LiDAR(`sensor_msgs/LaserScan`)
   최소 거리값으로 연결 (현재는 테스트용 `mock_proximity_distance`만 존재).
3. Gazebo에서 TurtleBot3 시뮬레이션으로 먼저 검증 후 실물 배포.
4. Sim-to-Real 갭 완화를 위한 LoRA 파인튜닝 파이프라인은 Phase 3 후반부, 실물 주행
   데이터가 축적된 이후 착수 (PRD 5절과 일치).

## 5. 마일스톤 요약

| 마일스톤 | 범위 | 완료 기준 |
| --- | --- | --- |
| M0 | 1절의 버그 수정 | 기존 테스트 통과 + 신규 회귀 테스트 |
| M1 | VLM 백본 1종 통합 (§2.1) | 목업 대비 동일 인터페이스로 실제 모델 추론 성공, 지연시간 측정치 확보 |
| M2 | 씬 그래프 연동 + 버퍼 임베딩 캐싱 (§2.2, 2.3) | 에이전트 루프에서 오브젝트가 실제로 씬 그래프에 누적되는지 통합 테스트로 확인 |
| M3 | 테스트 커버리지 확충 (§2.5) | 4개 핵심 모듈 모두 단위 테스트 보유 |
| M4 | 시뮬레이터 어댑터 (§3) | Habitat-Sim 위에서 에이전트 루프 1회 전체 실행 |
| M5 | ROS2/TurtleBot3 브리지 (§4) | Gazebo 시뮬레이션에서 자연어 지시 → 실제 `Twist` 발행 확인 |

## 6. 명시적으로 보류하는 항목

- Phase 1의 Matterport3D 데이터 로더, DUET Dual-Scale Graph Transformer, SR/SPL/NE
  대시보드는 본 계획에서 별도 트랙으로 분리하며 착수하지 않는다 (PRD 오너 승인 시
  재검토).
- Isaac Sim 어댑터, Unitree/Hello Robot 연동은 TurtleBot3 경로가 안정화된 이후로
  미룬다.
