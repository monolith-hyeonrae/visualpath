# visualpath

영상 분석 파이프라인 플랫폼. 플러그인 기반 extractor를 다양한 격리 수준으로 실행합니다.

## 핵심 기능

| 기능 | 설명 |
|------|------|
| **Extractor 인터페이스** | `BaseExtractor`, `Observation` 추상 클래스 |
| **Plugin Discovery** | `entry_points` 기반 자동 플러그인 탐색 |
| **격리 실행** | Inline, Thread, Process, Venv 수준 지원 |
| **VenvWorker** | 독립 venv에서 subprocess로 실행 (의존성 충돌 해결) |
| **Fusion** | 여러 extractor 결과를 통합하여 결정 |
| **Observability** | TraceLevel 기반 로깅/추적 |

## 설치

```bash
# 기본 설치
uv pip install -e .

# ZMQ IPC 지원 (VenvWorker 사용 시)
uv pip install -e ".[zmq]"
```

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Main Process                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ InlineWorker │  │ ThreadWorker │  │  VenvWorker  │       │
│  │  (동일 스레드) │  │  (별도 스레드) │  │ (별도 프로세스) │       │
│  └──────────────┘  └──────────────┘  └──────┬───────┘       │
└─────────────────────────────────────────────┼───────────────┘
                                              │ ZMQ IPC
┌─────────────────────────────────────────────┼───────────────┐
│  Subprocess (venv-face/bin/python)          │               │
│  ┌──────────────────────────────────────────▼─────────────┐ │
│  │  visualpath.process.worker                             │ │
│  │    - Entry point로 extractor 로드                       │ │
│  │    - Frame 수신 → extract() → Observation 응답          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 사용법

### 1. Extractor 작성

```python
from visualpath.core import BaseExtractor, Observation
from visualbase import Frame

class MyExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "my_extractor"

    def extract(self, frame: Frame) -> Observation:
        # 분석 로직
        score = self._analyze(frame.data)

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"score": score},
        )
```

### 2. 플러그인 등록 (pyproject.toml)

```toml
[project.entry-points."visualpath.extractors"]
my_extractor = "mypackage.extractors:MyExtractor"
```

### 3. 플러그인 탐색 및 사용

```python
from visualpath.plugin import discover_extractors, create_extractor

# 등록된 extractor 목록
extractors = discover_extractors()
print(list(extractors.keys()))  # ['my_extractor', 'face', 'pose', ...]

# 인스턴스 생성
extractor = create_extractor("my_extractor")
```

### 4. Worker로 실행

```python
from visualpath.process import WorkerLauncher
from visualpath.core import IsolationLevel

# Inline (동일 프로세스)
worker = WorkerLauncher.create(
    level=IsolationLevel.INLINE,
    extractor=my_extractor,
)

# Thread (별도 스레드)
worker = WorkerLauncher.create(
    level=IsolationLevel.THREAD,
    extractor=my_extractor,
)

# Venv (별도 venv에서 subprocess로 실행)
worker = WorkerLauncher.create(
    level=IsolationLevel.VENV,
    extractor=None,  # subprocess에서 entry_point로 로드
    venv_path="/path/to/venv",
    extractor_name="my_extractor",
)

# 실행
worker.start()
result = worker.process(frame)
print(result.observation.signals)
worker.stop()
```

### 5. VenvWorker로 의존성 충돌 해결

서로 다른 의존성을 가진 extractor들을 각각의 venv에서 실행:

```bash
# 각 worker용 venv 생성
uv venv venv-face && source venv-face/bin/activate
uv pip install -e "mypackage[face,zmq]"

uv venv venv-pose && source venv-pose/bin/activate
uv pip install -e "mypackage[pose,zmq]"
```

```python
from visualpath.process import VenvWorker

# Face worker (insightface, onnxruntime-gpu)
face_worker = VenvWorker(
    extractor=None,
    venv_path="/path/to/venv-face",
    extractor_name="face",
)

# Pose worker (ultralytics, torch)
pose_worker = VenvWorker(
    extractor=None,
    venv_path="/path/to/venv-pose",
    extractor_name="pose",
)

# 각각 독립된 프로세스에서 실행 - 의존성 충돌 없음!
face_worker.start()
pose_worker.start()
```

## 모듈 구조

```
visualpath/
├── core/
│   ├── extractor.py    # BaseExtractor, Observation, DummyExtractor
│   ├── fusion.py       # BaseFusion, FusionResult
│   ├── isolation.py    # IsolationLevel, IsolationConfig
│   └── path.py         # Path, PathOrchestrator
├── process/
│   ├── launcher.py     # WorkerLauncher, InlineWorker, ThreadWorker, VenvWorker
│   ├── worker.py       # Subprocess 진입점 (visualpath-worker CLI)
│   ├── mapper.py       # ObservationMapper, CompositeMapper
│   ├── ipc.py          # ExtractorProcess, FusionProcess
│   └── orchestrator.py # ExtractorOrchestrator
├── plugin/
│   └── discovery.py    # discover_extractors, PluginRegistry
├── backends/
│   └── protocols.py    # DetectionBackend, DetectionResult
└── observability/
    ├── records.py      # TraceRecord 타입들
    └── sinks.py        # FileSink, ConsoleSink, MemorySink
```

## IPC 아키텍처 (A-B*-C 패턴)

분산 환경에서의 프레임 처리 파이프라인:

```
A (Ingest) ←── TRIG ──← C (Fusion)
    ↓                      ↑
  Video                 OBS messages
    ↓                      │
    ├→ B1: Extractor1 ───→ ┤
    ├→ B2: Extractor2 ───→ ├→ Fusion
    └→ B3: Extractor3 ───→ ┘
```

- **ExtractorProcess**: 프레임 읽기 → 추출 → Observation 전송
- **FusionProcess**: Observation 수신 → frame_id 기준 정렬 (100ms 윈도우) → Trigger 판단

## 디자인 패턴

| 패턴 | 적용 위치 | 용도 |
|------|----------|------|
| **Abstract Base Class** | BaseExtractor, BaseFusion, BaseWorker | 인터페이스 정의 |
| **Generic Type** | `Observation[T]` | 도메인별 데이터 타입 지원 |
| **Protocol** | ObservationMapper, DetectionBackend | 구조적 서브타이핑 |
| **Factory Method** | WorkerLauncher.create() | 격리 수준별 Worker 생성 |
| **Singleton** | ObservabilityHub | 전역 트레이싱 설정 |
| **Chain of Responsibility** | CompositeMapper | 플러그인 가능한 직렬화 |
| **Context Manager** | Path, PathOrchestrator, BaseExtractor | 리소스 관리 |

## 성능 특성

| 격리 수준 | 오버헤드 | 용도 |
|----------|---------|------|
| INLINE | 0% | 빠른 경량 Extractor |
| THREAD | ~2-5% | I/O 바운드 작업 |
| PROCESS | ~5-10% | 메모리 격리 필요 시 |
| VENV | ~10-20% | ML 의존성 충돌 해결 |

| Observability 레벨 | 오버헤드 | 용도 |
|-------------------|---------|------|
| OFF | 0% | 프로덕션 기본 |
| MINIMAL | <1% | 중요 이벤트만 |
| NORMAL | ~5% | 프레임 요약 |
| VERBOSE | ~15% | 전체 디버깅 |

## 스레드 안전성

모든 주요 컴포넌트는 스레드 안전하게 설계되었습니다:

- **ObservabilityHub**: Lock 기반 emit() 및 sink 관리
- **FileSink/ConsoleSink/MemorySink**: Lock 기반 동기화
- **ExtractorOrchestrator**: ThreadPoolExecutor 사용
- **Path/PathOrchestrator**: ThreadPoolExecutor 사용

## 테스트

```bash
uv run pytest tests/ -v
```

## 문서

- [아키텍처 상세](docs/architecture.md): 아키텍처, 데이터 흐름, 확장 포인트
- [스트림 동기화](docs/stream-synchronization.md): A-B*-C 동기화 문제와 해결책
