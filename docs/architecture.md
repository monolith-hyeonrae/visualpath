# VisualPath 아키텍처

> 문서 작성일: 2026-02-02
> 상태: 구현 완료

## 개요

이 문서는 visualpath 패키지의 아키텍처와 플러그인 생태계 비전을 설명합니다.

---

## 3계층 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    visualbase                            │
│              (미디어 소스 기반 레이어)                    │
│  - Frame, Source, Stream 추상화                          │
│  - 비디오/카메라/스트림 통합 인터페이스                   │
│  - 클립 추출                                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    visualpath                            │
│              (분석 플랫폼 코어 레이어)                    │
│  - BaseExtractor, Observation 인터페이스                 │
│  - Plugin discovery & loading                            │
│  - Worker orchestration (격리 실행)                      │
│  - Fusion framework                                      │
│  - Observability system                                  │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │facemoment│    │ plugin-A │    │ plugin-B │
    │ (plugin) │    │          │    │          │
    │  - Face  │    │  - OCR   │    │ - Object │
    │  - Pose  │    │  - Text  │    │ - Scene  │
    │  - Gesture│   │          │    │          │
    └──────────┘    └──────────┘    └──────────┘
```

### 계층별 역할

| 레이어 | 패키지 | 역할 | 의존성 |
|--------|--------|------|--------|
| **Media** | visualbase | 미디어 소스, Frame, 클립 추출 | opencv, numpy |
| **Platform** | visualpath | 플러그인 프레임워크, IPC, Fusion | visualbase, pyzmq |
| **Plugin** | facemoment 등 | 순수 분석 로직 (Extractor 구현체) | visualpath, ML libs |

---

## visualpath 패키지 구조

```
visualpath/
├── core/
│   ├── extractor.py     # BaseExtractor, Observation
│   ├── fusion.py        # BaseFusion, FusionResult
│   ├── isolation.py     # IsolationLevel (INLINE, THREAD, PROCESS, VENV)
│   └── path.py          # Path utilities
├── process/
│   ├── worker.py        # BaseWorker, InlineWorker
│   ├── launcher.py      # WorkerLauncher
│   ├── ipc.py           # ExtractorProcess, FusionProcess
│   ├── mapper.py        # ObservationMapper
│   └── orchestrator.py  # ExtractorOrchestrator
├── plugin/
│   └── discovery.py     # discover_extractors, discover_fusions
├── backends/
│   └── protocols.py     # Backend protocols
└── observability/
    ├── records.py       # TraceRecord types
    └── sinks.py         # Sink 구현체들
```

---

## 핵심 인터페이스

### BaseExtractor

```python
from visualpath.core import BaseExtractor, Observation
from typing import Any
import numpy as np

class MyExtractor(BaseExtractor):
    """Custom extractor plugin example."""

    name = "my_extractor"
    version = "1.0.0"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # 백엔드 초기화

    def extract(self, frame: np.ndarray, frame_id: int) -> list[Observation]:
        # 분석 로직
        results = self._analyze(frame)

        return [
            Observation(
                source="my_extractor",
                frame_id=frame_id,
                data=result,
                confidence=result.score,
            )
            for result in results
        ]

    def reset(self) -> None:
        # 상태 초기화
        pass
```

### BaseFusion

```python
from visualpath.core import BaseFusion, FusionResult
from visualpath.core import Observation

class MyFusion(BaseFusion):
    """Custom fusion plugin example."""

    name = "my_fusion"

    def process(
        self,
        observations: dict[str, list[Observation]]
    ) -> FusionResult:
        # 여러 extractor 결과 결합
        triggers = self._analyze_patterns(observations)

        return FusionResult(
            triggers=triggers,
            metadata={"processed_at": time.time()}
        )
```

---

## 플러그인 시스템

### entry_points 등록

플러그인은 `pyproject.toml`에서 entry_points로 등록됩니다:

```toml
[project.entry-points."visualpath.extractors"]
face = "mypackage.extractors.face:FaceExtractor"
pose = "mypackage.extractors.pose:PoseExtractor"
custom = "mypackage.extractors.custom:CustomExtractor"

[project.entry-points."visualpath.fusions"]
highlight = "mypackage.fusion.highlight:HighlightFusion"
custom = "mypackage.fusion.custom:CustomFusion"
```

### 플러그인 Discovery

```python
from visualpath.plugin import discover_extractors, discover_fusions

# 설치된 모든 Extractor 발견
extractors = discover_extractors()
# {'face': <class FaceExtractor>, 'pose': <class PoseExtractor>, ...}

# 설치된 모든 Fusion 발견
fusions = discover_fusions()
# {'highlight': <class HighlightFusion>, ...}
```

---

## Worker 격리 실행

ML 라이브러리 간 의존성 충돌을 해결하기 위해 Worker별 독립 실행을 지원합니다.

### IsolationLevel

```python
from visualpath.core.isolation import IsolationLevel

# 격리 수준
IsolationLevel.INLINE    # 같은 프로세스, 같은 스레드
IsolationLevel.THREAD    # 같은 프로세스, 별도 스레드
IsolationLevel.PROCESS   # 별도 프로세스
IsolationLevel.VENV      # 별도 프로세스 + 별도 venv
```

### WorkerLauncher

```python
from visualpath.process import WorkerLauncher
from visualpath.core.isolation import IsolationLevel

launcher = WorkerLauncher()

# venv 격리 Worker 생성
worker = launcher.create(
    extractor_cls=FaceExtractor,
    isolation=IsolationLevel.VENV,
    venv_path="/opt/venv-face"
)

# 프레임 처리
observation = worker.extract(frame)
```

---

## on_trigger 콜백 패턴

visualpath는 **비즈니스 로직을 포함하지 않습니다**. Action 처리는 앱에서 콜백으로 구현합니다.

```python
# visualpath는 Action을 모름 - 콜백만 제공
def run_pipeline(source, on_trigger: Callable[[Trigger], None]):
    for frame in source:
        observations = orchestrator.extract_all(frame)
        trigger = fusion.update(observations)
        if trigger:
            on_trigger(trigger)  # 앱에서 처리

# 앱에서 Action 정의
def handle_trigger(trigger):
    clip = clipper.extract(trigger.frame_id)
    save_clip(clip)

# 실행
run_pipeline(video_source, on_trigger=handle_trigger)
```

---

## Observability

### TraceLevel

| 레벨 | 용도 | 오버헤드 |
|------|------|----------|
| OFF | 프로덕션 기본 | 0% |
| MINIMAL | Trigger만 로깅 | <1% |
| NORMAL | 프레임 요약 + 상태 전환 | ~5% |
| VERBOSE | 모든 Signal + 타이밍 | ~15% |

### 사용법

```python
from visualpath.observability import ObservabilityHub, TraceLevel

hub = ObservabilityHub(level=TraceLevel.NORMAL)

# 레코드 발생
hub.emit(TimingRecord(frame_id=100, component="face", processing_ms=42.3))

# Sink 연결
hub.add_sink(FileSink("trace.jsonl"))
hub.add_sink(ConsoleSink())
```

---

---

## 데이터 흐름

### A. Inline Extraction

```
Frame → InlineWorker.process()
        → extractor.extract(frame)
        → Observation
```

### B. Thread-based Extraction

```
Frame → ThreadWorker.process()
        → ThreadPoolExecutor.submit(extractor.extract, frame)
        → Future.result() (blocking)
        → Observation
```

### C. Venv-based Extraction (ZMQ IPC)

```
Frame → VenvWorker.process()
        → Serialize frame to JPEG + JSON
        → ZMQ send {"type": "extract", "frame": {...}}
        → Subprocess receives, loads extractor, calls extract()
        → Serialize observation to JSON
        → ZMQ send {"observation": {...}}
        → Deserialize observation
        → Observation
```

### D. Path Processing

```
Frame → Path.process(frame)
        ├─ extract_all() → parallel extractors → [Observations]
        └─ fusion.update(each obs) → FusionResults
```

### E. Orchestrator Processing

```
Frame → PathOrchestrator.process_all(frame)
        ├─ Path1.process(frame) → {path1: [FusionResults]}
        ├─ Path2.process(frame) → {path2: [FusionResults]}
        └─ ... → {path_n: [FusionResults]}
```

---

## 핵심 데이터 타입

### Observation[T]

```python
@dataclass
class Observation(Generic[T]):
    source: str                      # Extractor 이름
    frame_id: int                    # 프레임 식별자
    t_ns: int                        # 타임스탬프 (나노초, 소스 기준)
    signals: Dict[str, float]        # 스칼라 피처 값
    data: Optional[T] = None         # 도메인별 데이터
    metadata: Dict[str, Any]         # 추가 정보
    timing: Optional[Dict[str, float]] = None  # 컴포넌트별 처리 시간 (ms)
```

### FusionResult

```python
@dataclass
class FusionResult:
    should_trigger: bool              # 트리거 발생 여부
    trigger: Optional[Trigger]        # 트리거 객체
    score: float                      # 신뢰도 [0, 1]
    reason: str                       # 사람이 읽을 수 있는 이유
    observations_used: int            # 처리된 observation 수
    metadata: Dict[str, Any]          # 추가 컨텍스트
```

### WorkerResult

```python
@dataclass
class WorkerResult:
    observation: Optional[Observation]
    error: Optional[str] = None
    timing_ms: float = 0.0
```

---

## Worker 구현 상세

### InlineWorker (IsolationLevel.INLINE)

- **격리**: 동일 프로세스, 동일 스레드
- **오버헤드**: 0 (직접 메서드 호출)
- **용도**: 간단하고 빠른 Extractor

### ThreadWorker (IsolationLevel.THREAD)

- **격리**: 동일 프로세스, 다른 스레드
- **구현**: 단일 스레드 ThreadPoolExecutor
- **메서드**:
  - `process()`: 동기 submit + wait
  - `process_async()`: 논블로킹 Future 반환
- **용도**: I/O 바운드 Extractor

### VenvWorker (IsolationLevel.VENV)

- **격리**: 다른 venv + 다른 프로세스
- **통신**: ZMQ REQ-REP 패턴 (IPC 소켓)
- **주요 기능**:
  - 임시 IPC 소켓 파일 생성
  - `subprocess.Popen`으로 subprocess 관리
  - Handshake 프로토콜 (ping/pong)
  - ZMQ 불가 시 InlineWorker로 폴백
  - JPEG 인코딩으로 프레임 전송
  - 처리 및 handshake 타임아웃

### 메시지 프로토콜

```json
// Request
{"type": "extract", "frame": {...}}
{"type": "ping"}
{"type": "shutdown"}

// Response
{"observation": {...}}
{"type": "pong", "extractor": "face"}
{"error": "...", "traceback": "..."}
```

---

## Observation Mapper

### DefaultObservationMapper

JSON 기반 직렬화:
- `to_message(observation) -> Optional[str]`
- `from_message(message) -> Optional[Observation]`

### CompositeMapper

Chain of Responsibility 패턴:
- 여러 mapper를 순서대로 시도
- 도메인별 커스텀 직렬화 지원
- `add_mapper()`로 런타임 등록

---

## 에러 핸들링 및 복원력

| 상황 | 처리 방식 |
|------|----------|
| **Extractor 실패** | extract() 내부에서 캐치, 로깅, 다른 extractor 계속 |
| **Worker 타임아웃** | `as_completed(timeout=...)`로 처리 |
| **Subprocess 실패** | VenvWorker가 InlineWorker로 폴백 |
| **ZMQ 에러** | WorkerResult.error 필드로 반환 |
| **Sink 실패** | hub.emit()에서 무시 (메인 처리 영향 없음) |
| **시그널 핸들링** | ExtractorProcess/FusionProcess가 SIGINT, SIGTERM 처리 |

---

## 확장 포인트

### A. 격리 설정

```python
config = IsolationConfig(
    default_level=IsolationLevel.PROCESS,
    overrides={"face": IsolationLevel.VENV},
    venv_paths={"face": "/opt/venvs/face"},
)
```

### B. 플러그인 등록

```python
# pyproject.toml
[project.entry-points."visualpath.extractors"]
face = "myplugin.extractors:FaceExtractor"

# 또는 런타임 등록
registry = PluginRegistry()
registry.register_extractor("custom", MyExtractor)
```

### C. Observability 설정

```python
hub = ObservabilityHub.get_instance()
hub.configure(
    level=TraceLevel.NORMAL,
    sinks=[FileSink("/tmp/trace.jsonl"), ConsoleSink()]
)
```

### D. 커스텀 Mapper

```python
class FaceMapper(ObservationMapper):
    def to_message(self, obs: Observation) -> Optional[str]:
        # 도메인별 직렬화
        ...

composite = CompositeMapper([FaceMapper(), DefaultObservationMapper()])
```

---

## 관련 문서

- [Stream Synchronization](./stream-synchronization.md): 스트림 동기화 아키텍처
- visualbase/CLAUDE.md: 미디어 I/O 라이브러리
- facemoment/CLAUDE.md: 981파크 분석 앱 (플러그인 예시)
