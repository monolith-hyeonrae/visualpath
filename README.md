# visualpath

영상 분석 파이프라인 플랫폼.

## Quick Start

```python
import visualpath as vp

# 비디오 처리 (one-liner)
triggers = vp.run("video.mp4", extractors=["face", "pose"])
```

### Custom Extractor (3줄)

```python
@vp.extractor("brightness")
def check_brightness(frame):
    return {"brightness": float(frame.data.mean())}
```

### Custom Fusion (4줄)

```python
@vp.fusion(sources=["face"], cooldown=2.0)
def smile_detector(face):
    if face.get("happy", 0) > 0.5:
        return vp.trigger("smile", score=face["happy"])
```

### 실행

```python
# 간단히
triggers = vp.run("video.mp4", extractors=["brightness"])

# 콜백과 함께
vp.run("video.mp4", ["face"], fusion=smile_detector,
       on_trigger=lambda t: print(f"Trigger: {t.label}"))

# 상세 결과
result = vp.process("video.mp4", ["face", "pose"])
print(f"{len(result.triggers)} triggers in {result.frame_count} frames")
```

## 설치

```bash
uv pip install -e .
```

## API Reference

### High-level API

| 함수 | 용도 |
|------|------|
| `@vp.extractor(name)` | 함수를 extractor로 변환 |
| `@vp.fusion(sources, cooldown)` | 함수를 fusion으로 변환 |
| `vp.trigger(reason, score)` | 트리거 생성 |
| `vp.run(video, extractors)` | 비디오 처리 (triggers 반환) |
| `vp.process(video, extractors)` | 비디오 처리 (상세 결과) |
| `vp.list_extractors()` | 사용 가능한 extractor 목록 |
| `vp.list_fusions()` | 사용 가능한 fusion 목록 |
| `vp.get_extractor(name)` | 이름으로 extractor 가져오기 |

### Extractor Options

```python
# 초기화/정리 함수
model = None

def load_model():
    global model
    model = MyModel()

@vp.extractor("detector", init=load_model, cleanup=lambda: model.close())
def detect(frame):
    return {"count": len(model.detect(frame.data))}

# Context manager로 사용
with check_brightness:
    obs = check_brightness.extract(frame)
```

### Fusion Options

```python
@vp.fusion(
    sources=["face", "pose"],  # 필요한 extractor들
    name="my_fusion",          # 이름 (기본: 함수명)
    cooldown=3.0,              # 트리거 간 최소 간격 (초)
)
def multi_source(face, pose):
    if face.get("happy") > 0.5 and pose.get("wave"):
        return vp.trigger("greeting", score=0.9, face_id=face.get("id"))
```

### Return Values

```python
# Extractor: dict 반환 (자동 변환)
@vp.extractor("objects")
def detect(frame):
    return {
        "count": 3.0,                    # signals에 저장
        "boxes": [[10, 20, 30, 40]],     # data에 저장 (non-scalar)
    }

# Fusion: vp.trigger() 또는 None 반환
@vp.fusion(sources=["objects"])
def alert(objects):
    if objects.get("count", 0) > 5:
        return vp.trigger("crowded", score=objects["count"] / 10)
    # return None (또는 생략) = 트리거 없음
```

---

## Advanced Usage

### Class-based Extractor

복잡한 상태 관리가 필요한 경우:

```python
from visualpath.core import BaseExtractor, Observation

class MyExtractor(BaseExtractor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None

    @property
    def name(self) -> str:
        return "my_extractor"

    def initialize(self) -> None:
        self.model = load_model()

    def cleanup(self) -> None:
        self.model.close()

    def extract(self, frame) -> Observation:
        score = self.model.analyze(frame.data)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"score": score},
        )
```

### Plugin Registration

플러그인으로 배포하려면 `pyproject.toml`에 entry point 등록:

```toml
[project.entry-points."visualpath.extractors"]
my_extractor = "mypackage.extractors:MyExtractor"

[project.entry-points."visualpath.fusions"]
my_fusion = "mypackage.fusions:MyFusion"
```

### Worker Isolation

ML 의존성 충돌 해결을 위한 격리 실행:

```python
from visualpath.process import WorkerLauncher
from visualpath.core import IsolationLevel

# Inline (동일 프로세스, 기본)
worker = WorkerLauncher.create(
    level=IsolationLevel.INLINE,
    extractor=my_extractor,
)

# Thread (별도 스레드)
worker = WorkerLauncher.create(
    level=IsolationLevel.THREAD,
    extractor=my_extractor,
)

# Venv (별도 venv에서 subprocess)
worker = WorkerLauncher.create(
    level=IsolationLevel.VENV,
    venv_path="/path/to/venv-face",
    extractor_name="face",  # entry_point로 로드
)

worker.start()
result = worker.process(frame)
worker.stop()
```

### VenvWorker로 의존성 충돌 해결

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
    venv_path="/path/to/venv-face",
    extractor_name="face",
)

# Pose worker (ultralytics, torch)
pose_worker = VenvWorker(
    venv_path="/path/to/venv-pose",
    extractor_name="pose",
)

# 각각 독립된 프로세스에서 실행
face_worker.start()
pose_worker.start()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  vp.run("video.mp4", extractors=["face", "pose"])           │
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Source    │ ──► │  Extractor  │ ──► │   Fusion    │   │
│  │  (frames)   │     │ (parallel)  │     │ (triggers)  │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### IPC Architecture (A-B*-C)

분산 환경에서의 처리:

```
A (Ingest) ←── TRIG ──← C (Fusion)
    ↓                      ↑
  Video                 OBS messages
    ↓                      │
    ├→ B1: Extractor1 ───→ ┤
    ├→ B2: Extractor2 ───→ ├→ Fusion
    └→ B3: Extractor3 ───→ ┘
```

## Module Structure

```
visualpath/
├── api.py              # High-level API (@extractor, @fusion, run, process)
├── core/
│   ├── extractor.py    # BaseExtractor, Observation
│   ├── fusion.py       # BaseFusion, FusionResult
│   └── isolation.py    # IsolationLevel
├── process/
│   ├── launcher.py     # WorkerLauncher, VenvWorker
│   └── worker.py       # Subprocess entry point
├── plugin/
│   └── discovery.py    # discover_extractors, PluginRegistry
├── flow/               # FlowGraph (DAG-based pipeline)
└── observability/      # Tracing & logging
```

## Performance

| 격리 수준 | 오버헤드 | 용도 |
|----------|---------|------|
| INLINE | 0% | 빠른 경량 Extractor |
| THREAD | ~2-5% | I/O 바운드 작업 |
| PROCESS | ~5-10% | 메모리 격리 |
| VENV | ~10-20% | ML 의존성 충돌 해결 |

## Test

```bash
uv run pytest tests/ -v
```

## Documentation

- [Architecture](docs/architecture.md)
- [Stream Synchronization](docs/stream-synchronization.md)
