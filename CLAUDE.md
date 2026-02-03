# VisualPath - Claude Session Context

> 최종 업데이트: 2026-02-03
> 상태: **구현 완료**

## 프로젝트 역할

**범용 영상 분석 프레임워크** (재사용 가능):
- Extractor/Fusion 추상화 인터페이스
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
| `core.BaseExtractor` | 분석기 ABC | Extractor 구현 인터페이스 |
| `core.Observation` | 분석 결과 컨테이너 | Extractor 출력 타입 |
| `core.BaseFusion` | 결정기 ABC | Trigger 판단 인터페이스 |
| `core.IsolationLevel` | 격리 수준 | INLINE, THREAD, PROCESS, VENV |
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
│   ├── extractor.py     # BaseExtractor, Observation
│   ├── fusion.py        # BaseFusion, FusionResult
│   ├── isolation.py     # IsolationLevel
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
│   ├── protocols.py     # ML Backend protocols
│   ├── base.py          # ExecutionBackend ABC
│   ├── simple/          # SimpleBackend (모듈식 백엔드)
│   │   ├── backend.py   # SimpleBackend + 팩토리 함수
│   │   ├── scheduler.py # 프레임 스케줄링 전략
│   │   ├── synchronizer.py  # Observation 동기화 전략
│   │   ├── buffer.py    # 백프레셔 버퍼 전략
│   │   └── executor.py  # Extractor 실행 전략
│   └── pathway/         # Pathway 스트리밍 백엔드
│       ├── backend.py   # PathwayBackend
│       ├── connector.py # VideoConnectorSubject
│       ├── operators.py # Extractor/Fusion UDF
│       └── converter.py # FlowGraph → Pathway
└── observability/
    ├── records.py       # TraceRecord types
    └── sinks.py         # Sink 구현체들
```

## 사용 예시

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
