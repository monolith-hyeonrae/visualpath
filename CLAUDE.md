# VisualPath - Claude Session Context

> 최종 업데이트: 2026-02-05
> 상태: **통합 Module API 완료**

## 프로젝트 역할

**범용 영상 분석 프레임워크** (재사용 가능):
- **통합 Module 인터페이스** (Extractor/Fusion 통합 완료)
- **NodeSpec 기반 선언적 노드 정의** (ModuleSpec, ExtractSpec 등)
- Worker 격리 실행 (venv, process, thread)
- Plugin discovery (entry_points 기반)
- Observability (트레이싱, 로깅)
- 파이프라인 실행 엔진
- **실행 백엔드 추상화** (Simple, Pathway)

## 아키텍처 위치

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어                                             │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │ visualbase  │ ───→ │ visualpath  │ ← 현재 패키지     │
│  │ (미디어 I/O)│      │ (분석 프레임워크)               │
│  └─────────────┘      └─────────────┘                   │
└─────────────────────────────────────────────────────────┘
                              │
                    앱에서 사용 (facemoment 등)
```

## 핵심 제공 기능

| 모듈 | 기능 | 설명 |
|------|------|------|
| `core.Module` | **통합 모듈 ABC** | Observation 또는 FusionResult 반환 |
| `core.Observation` | 분석 결과 컨테이너 | Analyzer 모듈 출력 타입 |
| `core.FusionResult` | 트리거 결과 컨테이너 | Trigger 모듈 출력 타입 |
| `core.BaseExtractor` | 분석기 ABC (deprecated) | Module 사용 권장 |
| `core.BaseFusion` | 결정기 ABC (deprecated) | Module 사용 권장 |
| `core.IsolationLevel` | 격리 수준 | INLINE, THREAD, PROCESS, VENV |
| `flow.specs.*` | **NodeSpec 선언적 스펙** | ModuleSpec, ExtractSpec 등 |
| `process.WorkerLauncher` | Worker 팩토리 | 격리 수준별 Worker 생성 |
| `process.VenvWorker` | venv 격리 Worker | ML 의존성 충돌 해결 |
| `plugin.discover_*` | 플러그인 검색 | entry_points 기반 |
| `observability` | 트레이싱 | TraceLevel, Sink |
| `backends.ExecutionBackend` | 실행 백엔드 ABC | 파이프라인 실행 추상화 |
| `backends.SimpleBackend` | 순차 실행 | 기본 백엔드 |
| `backends.PathwayBackend` | 스트리밍 실행 | Pathway 기반 (옵션) |

## 디렉토리 구조

```
visualpath/
├── core/
│   ├── module.py        # Module ABC (통합 인터페이스)
│   ├── extractor.py     # BaseExtractor, Observation (deprecated)
│   ├── fusion.py        # BaseFusion, FusionResult (deprecated)
│   ├── isolation.py     # IsolationLevel
│   └── path.py          # Path utilities
├── flow/
│   ├── node.py          # FlowNode ABC (spec 프로퍼티 포함)
│   ├── specs.py         # NodeSpec frozen dataclasses (18종)
│   ├── interpreter.py   # SimpleInterpreter (spec 기반 실행)
│   ├── graph.py         # FlowGraph DAG
│   ├── executor.py      # GraphExecutor (interpreter 위임)
│   ├── builder.py       # FlowGraphBuilder (fluent API)
│   └── nodes/
│       ├── source.py    # SourceNode → SourceSpec
│       ├── path.py      # PathNode → ModuleSpec / ExtractSpec
│       ├── filter.py    # FilterNode 등 → FilterSpec 등
│       ├── sampler.py   # SamplerNode 등 → SampleSpec 등
│       ├── branch.py    # BranchNode 등 → BranchSpec 등
│       └── join.py      # JoinNode 등 → JoinSpec 등
├── process/
│   ├── worker.py        # BaseWorker, InlineWorker
│   ├── launcher.py      # WorkerLauncher
│   ├── ipc.py           # ExtractorProcess, FusionProcess
│   ├── mapper.py        # ObservationMapper
│   └── orchestrator.py  # ExtractorOrchestrator
├── plugin/
│   └── discovery.py     # discover_extractors, discover_fusions
├── backends/
│   ├── protocols.py     # ML Backend protocols
│   ├── base.py          # ExecutionBackend ABC
│   ├── simple/          # SimpleBackend
│   │   └── backend.py   # SimpleBackend (GraphExecutor 위임)
│   └── pathway/         # Pathway 스트리밍 백엔드
│       ├── backend.py   # PathwayBackend (_find_fusion → spec 기반)
│       ├── connector.py # VideoConnectorSubject
│       ├── operators.py # Extractor UDF
│       ├── converter.py # FlowGraphConverter (spec 기반 dispatch)
│       └── stats.py     # PathwayStats
└── observability/
    ├── records.py       # TraceRecord types
    └── sinks.py         # Sink 구현체들
```

## 사용 예시

### 통합 Module API (권장)

```python
from visualpath.core import Module, Observation, FusionResult
from visualpath.flow import FlowGraphBuilder, GraphExecutor

# Analyzer 모듈: Observation 반환
class FaceDetector(Module):
    depends = []  # 의존성 없음

    @property
    def name(self) -> str:
        return "face_detect"

    def process(self, frame, deps=None) -> Observation:
        faces = self._detect(frame.data)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"face_count": len(faces)},
        )

# Trigger 모듈: FusionResult 반환
class SmileTrigger(Module):
    depends = ["face_detect"]  # face_detect 의존

    @property
    def name(self) -> str:
        return "smile_trigger"

    @property
    def is_trigger(self) -> bool:
        return True

    def process(self, frame, deps=None) -> FusionResult:
        face_obs = deps.get("face_detect") if deps else None
        if not face_obs or face_obs.signals.get("face_count", 0) == 0:
            return FusionResult(should_trigger=False)
        return FusionResult(should_trigger=True, score=0.9)

# FlowGraph 빌드
graph = (FlowGraphBuilder()
    .source("frames")
    .path("analysis", modules=[FaceDetector(), SmileTrigger()])
    .on_trigger(lambda data: print(f"Trigger: {data}"))
    .build())

# 실행
with GraphExecutor(graph) as executor:
    for frame in video:
        executor.process(frame)
```

### Legacy API (하위 호환)

```python
from visualpath.core import BaseExtractor, Observation, BaseFusion
from visualpath.process import WorkerLauncher
from visualpath.core.isolation import IsolationLevel

# 앱에서 Extractor 구현
class MyExtractor(BaseExtractor):
    def extract(self, frame) -> Observation:
        ...

# Worker로 격리 실행
launcher = WorkerLauncher()
worker = launcher.create(
    extractor_cls=MyExtractor,
    isolation=IsolationLevel.VENV,
    venv_path="/opt/venv-my"
)
```

## 실행 백엔드

```python
import visualpath as vp

# 기본 Simple 백엔드 (순차 처리)
triggers = vp.run("video.mp4", ["face"], backend="simple")

# Pathway 백엔드 (스트리밍)
# pip install visualpath[pathway] 필요
triggers = vp.run("video.mp4", ["face"], backend="pathway")
```

### 백엔드 비교

| 백엔드 | 특징 | 용도 |
|--------|------|------|
| **SimpleBackend** | 모듈식 컴포넌트, 순차/병렬 처리 | 로컬 비디오, 개발/디버깅, 배치 처리 |
| **PathwayBackend** | 스트리밍, 백프레셔, 워터마크 | 실시간 처리, 복잡한 동기화 |

### SimpleBackend 컴포넌트

```python
from visualpath.backends.simple import (
    SimpleBackend,
    # 스케줄러: 프레임 선택/드롭 전략
    PassThroughScheduler,   # 모든 프레임 처리
    KeyframeScheduler,      # N번째 프레임만 처리
    AdaptiveRateScheduler,  # 목표 FPS 유지
    # 실행기: Extractor 실행 전략
    SequentialExecutor,     # 순차 실행
    ThreadPoolExecutor,     # 병렬 실행
    TimeoutExecutor,        # 타임아웃 있는 병렬 실행
    # 동기화: Observation 정렬 전략
    NoSyncSynchronizer,     # 즉시 전달
    TimeWindowSync,         # 시간 윈도우 기반 그룹핑
    BarrierSync,            # 모든 소스 대기
)

# 컴포넌트 조합
backend = SimpleBackend(
    scheduler=AdaptiveRateScheduler(target_fps=10),
    executor=ThreadPoolExecutor(max_workers=4),
    synchronizer=TimeWindowSync(window_ns=100_000_000),
)

# 또는 팩토리 함수 사용
from visualpath.backends.simple import (
    create_parallel_backend,
    create_realtime_backend,
    create_batch_backend,
)
backend = create_realtime_backend(target_fps=10)
```

## on_trigger 콜백 (비즈니스 로직은 앱에서)

```python
# visualpath는 Action을 모름 - 콜백만 제공
def run_pipeline(source, on_trigger: Callable[[Trigger], None]):
    for frame in source:
        trigger = fusion.update(observations)
        if trigger:
            on_trigger(trigger)  # 앱에서 처리

# 앱(facemoment)에서 Action 정의
def handle_trigger(trigger):
    clip = clipper.extract(trigger.frame_id)
    save_clip(clip)
```

## 테스트

```bash
cd ~/repo/monolith/visualpath
uv run pytest tests/ -v
```

## 의존성

- 코어: visualbase, numpy
- 옵션:
  - `pyzmq` (IPC)
  - `pathway>=0.8.0` (스트리밍 백엔드)

```bash
# Pathway 백엔드 설치
pip install visualpath[pathway]
```

## 문서

- `docs/architecture.md`: 아키텍처 및 플러그인 시스템
- `docs/stream-synchronization.md`: 스트림 동기화

## 관련 패키지

- **visualbase**: 미디어 I/O 기반
- **facemoment**: visualpath를 사용하는 981파크 특화 앱

---

## 개발 철학 및 의사 결정 기록

### NodeSpec 선언적 전환 (2026-02-04)

#### 문제 의식

FlowGraphConverter가 노드의 private 속성(`node._path`, `node._condition`,
`node._every_nth` 등)에 직접 접근하여 Pathway 연산자를 생성하고 있었음.
이는 노드 내부 구현과 백엔드 간 강결합을 만들고, 노드 구현 변경 시
converter가 깨지는 구조적 문제.

#### 설계 원칙

1. **spec + process() 공존** — spec은 선언적 의미, process()는 실행 fallback.
   기존 노드가 spec을 override하지 않아도 process()로 동작함.
2. **Fusion 별도 노드 없음** — fusion은 ExtractSpec의 일부로 포함.
   face_detection → face_expression 같은 의존성 체인과 본질적으로 같음.
3. **Source 외부 주입** — FlowGraph는 소스에 대해 모름.
4. **시간 의미론은 그래프에 속함** — JoinSpec에 window_ns, lateness_ns 포함.
   백엔드 기본값은 spec에 값이 없을 때의 fallback.
5. **불변 spec** — 모든 spec은 `frozen=True` dataclass.
   컬렉션 필드는 `tuple` 사용 (hashable, immutable).

#### 핵심 결정

| 결정 | 근거 |
|------|------|
| spec은 abstract가 아님 (기본 None) | 커스텀 노드가 spec 없이 process()만 구현해도 동작해야 함 |
| `PathNode.parallel`은 기본 False | Path._parallel(ThreadPool용)과 Pathway 스트림 분기는 다른 개념. 명시적 opt-in 필요 |
| Converter는 `isinstance(node.spec, SpecType)`으로 dispatch | isinstance(node, NodeClass) 대비: 노드 타입과 변환 로직이 분리됨 |
| JoinSpec.window_ns가 converter 기본값보다 우선 | 시간 의미론은 그래프 설계자가 정의, 백엔드는 fallback만 제공 |
| Pathway 병렬 분기는 의존성 그래프 분석 기반 | 독립 extractor만 별도 UDF로 분리, 의존성 체인은 하나의 UDF에 유지 |

#### 구현 범위

```
신규: flow/specs.py (17개 frozen dataclass)
신규: tests/test_flow_specs.py (64개 테스트)
수정: flow/node.py — spec 프로퍼티 추가
수정: flow/nodes/*.py (6개 파일) — 15개 노드에 spec 구현
수정: backends/pathway/converter.py — spec 기반 dispatch + 의존성 분석
수정: backends/pathway/backend.py — _find_fusion spec 기반
수정: flow/__init__.py — spec export
```

---

### Module 통합 완료 (2026-02-05)

#### 구현 내용

`Module` ABC를 통해 Extractor와 Fusion을 통합하는 작업 완료:

```python
# 통합된 Module 인터페이스
class Module(ABC):
    depends: List[str] = []  # 의존성 선언

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def process(self, frame, deps=None) -> ModuleOutput:
        # ModuleOutput = Union[Observation, FusionResult, None]
        ...

    @property
    def is_trigger(self) -> bool:
        return False  # True면 trigger 모듈

    def initialize(self) -> None: pass
    def cleanup(self) -> None: pass
    def reset(self) -> None: pass
```

#### 모듈 역할 결정

반환 타입으로 역할이 결정됨:
- `Observation` 반환 → Analyzer 모듈 (분석)
- `FusionResult` 반환 → Trigger 모듈 (결정)

#### API 통합 현황

| 컴포넌트 | 상태 | 설명 |
|----------|------|------|
| `core.Module` | ✅ 완료 | 통합 ABC |
| `flow.ModuleSpec` | ✅ 완료 | 통합 스펙 |
| `PathNode(modules=[...])` | ✅ 완료 | 통합 API |
| `FlowGraphBuilder.path(modules=[...])` | ✅ 완료 | 통합 빌더 API |
| `SimpleInterpreter` | ✅ 완료 | ModuleSpec 해석 |
| Legacy API (extractors/fusion) | ✅ 유지 | 하위 호환 |

#### 하위 호환성

기존 `BaseExtractor`, `BaseFusion` API는 deprecated로 유지:
- `PathNode(extractors=[...], fusion=...)` → `ExtractSpec` 반환
- `PathNode(modules=[...])` → `ModuleSpec` 반환

---

### Pathway 활용 현황

| Pathway 기능 | 현재 상태 | 비고 |
|-------------|----------|------|
| per-extractor 병렬 UDF | 구현 완료 | `parallel=True` + 의존성 분석 |
| interval_join (temporal) | 구현 완료 | JoinSpec.window_ns 우선 사용 |
| watermark/late arrival | 부분 구현 | JoinSpec.lateness_ns 전달 가능 |
| stateful fusion | subscribe 콜백 | Pathway UDF는 stateless, 현실적 선택 |
| 윈도우 집계 (windowby) | 미구현 | 향후 과제 |
| 증분 연산 | 미구현 | 실시간 모델 별도 설계 필요 |
