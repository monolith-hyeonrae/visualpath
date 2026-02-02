# VisualPath - Claude Session Context

> 최종 업데이트: 2026-02-02
> 상태: **구현 완료**

## 프로젝트 역할

**범용 영상 분석 프레임워크** (재사용 가능):
- Extractor/Fusion 추상화 인터페이스
- Worker 격리 실행 (venv, process, thread)
- Plugin discovery (entry_points 기반)
- Observability (트레이싱, 로깅)
- 파이프라인 실행 엔진

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
│   └── protocols.py     # Backend protocols
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
- 옵션: pyzmq (IPC)

## 문서

- `docs/architecture.md`: 아키텍처 및 플러그인 시스템
- `docs/stream-synchronization.md`: 스트림 동기화

## 관련 패키지

- **visualbase**: 미디어 I/O 기반
- **facemoment**: visualpath를 사용하는 981파크 특화 앱
