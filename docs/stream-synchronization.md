# 스트림 동기화

visualpath의 A-B*-C 아키텍처에서 여러 데이터 스트림을 동기화하는 문제와 해결책을 설명합니다.

## 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [visualbase와의 연동](#visualbase와의-연동)
3. [문제 1: Extractor 처리 시간 불균형](#문제-1-extractor-처리-시간-불균형)
4. [문제 2: 프레임 드롭과 백프레셔](#문제-2-프레임-드롭과-백프레셔)
5. [문제 3: OBS 동기화 지연](#문제-3-obs-동기화-지연)
6. [해결책: ExtractorOrchestrator](#해결책-extractororchestrator)
7. [해결책: 시간 윈도우 정렬](#해결책-시간-윈도우-정렬)
8. [Observability 연동](#observability-연동)
9. [설정 가이드](#설정-가이드)
10. [Pathway 백엔드](#pathway-백엔드) ← **구현 완료**

---

## 아키텍처 개요

```
              ┌─────────────────────────────────┐
              │      visualbase (A 모듈)        │
              │  - VideoSource: 프레임 공급     │
              │  - ClipExtractor: 클립 추출     │
              │  - Daemon: ZMQ 프레임 배포      │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                  ▼
        ┌──────────┐   ┌──────────┐      ┌──────────┐
        │ Extractor│   │ Extractor│      │ Extractor│  (B* 모듈)
        │    A     │   │    B     │      │    C     │
        └────┬─────┘   └────┬─────┘      └────┬─────┘
             │              │                  │
             │    OBS 메시지│                  │
             └──────────────┼──────────────────┘
                            ▼
                      ┌──────────┐
                      │  Fusion  │  (C 모듈)
                      │  Process │
                      └────┬─────┘
                           │  TRIG 메시지
                           ▼
              ┌─────────────────────────────────┐
              │      visualbase (A 모듈)        │
              │  - ClipExtractor로 클립 생성    │
              └─────────────────────────────────┘
```

---

## visualbase와의 연동

### visualbase 주요 컴포넌트

| 컴포넌트 | 위치 | 역할 |
|----------|------|------|
| `VideoSource` | `sources/` | 파일/카메라/스트림에서 Frame 생성 |
| `ClipExtractor` | `core/` | TRIG 메시지 기반 클립 추출 |
| `Daemon` | `daemon.py` | ZMQ 기반 프레임 배포 서버 |
| `Frame` | `core/` | 타임스탬프 포함 프레임 데이터 |

### IPC 인터페이스 (`visualbase.ipc`)

visualpath는 visualbase의 IPC 인터페이스를 통해 통신합니다:

```python
from visualbase.ipc.interfaces import VideoReader, MessageSender, MessageReceiver
from visualbase.ipc.factory import TransportFactory

# 프레임 수신 (A→B)
reader = TransportFactory.create_video_reader("fifo", "/tmp/vid.mjpg")

# OBS 메시지 송신 (B→C)
sender = TransportFactory.create_message_sender("uds", "/tmp/obs.sock")

# TRIG 메시지 수신 (C→A)
receiver = TransportFactory.create_message_receiver("uds", "/tmp/trig.sock")
```

### 메시지 형식

| 메시지 | 방향 | 내용 |
|--------|------|------|
| `Observation` | B→C | Extractor 분석 결과 |
| `TRIGMessage` | C→A | 트리거 이벤트 (시작/끝 시간, reason, score) |

### 클립 추출 흐름

```
1. Fusion이 TRIG 메시지 생성
   TRIGMessage(t_start_ns, t_end_ns, reason, score)

2. visualbase ClipExtractor가 TRIG 수신
   clip = extractor.extract(source, t_start_ns, t_end_ns)

3. 클립 파일 저장
   clip.save(output_path)
```

---

## 문제 1: Extractor 처리 시간 불균형

각 Extractor의 처리 시간이 크게 다릅니다. Fusion에서 모든 결과를 기다리면 가장 느린 Extractor가 전체 파이프라인의 병목이 됩니다.

```
시간 →
        0ms      20ms     40ms     60ms     80ms    100ms
        │        │        │        │        │        │
Ext-A   ├────────────────────────────┤                    42ms
        │        │        │        │        │        │
Ext-B   ├────────────┤                                    22ms
        │        │        │        │        │        │
Ext-C   ├────────────────────────────────────────────┤    58ms ← 병목!
        │        │        │        │        │        │
Ext-D   ├────┤                                             8ms
        │        │        │        │        │        │
                                                     ▼
                                              Fusion 대기 완료
```

### 컴포넌트별 처리 시간 예시

| Extractor | 일반 시간 | 변동성 | 원인 |
|-----------|-----------|--------|------|
| ML 기반 | 30-60ms | 중간 | 신경망 추론 |
| 경량 ML | 15-30ms | 낮음 | 최적화된 모델 |
| Heavy ML | 40-80ms | 높음 | 복잡한 모델 |
| 이미지 분석 | 5-10ms | 매우 낮음 | 순수 이미지 처리 |

**문제**: 같은 프레임에 대한 OBS가 서로 다른 시간에 Fusion에 도착합니다.

---

## 문제 2: 프레임 드롭과 백프레셔

처리 속도가 입력 속도를 따라가지 못하면 프레임이 누적되고, 결국 메모리 부족이나 지연이 발생합니다.

```
입력 FPS: 30        처리 FPS: 10
      │                  │
      ▼                  ▼
  ┌───────┐          ┌───────┐
  │Frame 1│──────────│Process│
  │Frame 2│          │       │
  │Frame 3│  ← 누적! │ 처리중│
  │Frame 4│          │       │
  │Frame 5│          │       │
  │Frame 6│          └───────┘
  │  ...  │
  └───────┘
      │
      ▼
  큐 깊이 증가 → 메모리 증가 → 지연 증가
```

### 드롭 전략

```
┌─────────────────────────────────────────────────────────────┐
│                    드롭 전략 선택                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Skip Oldest (기본)                                      │
│     큐가 가득 차면 가장 오래된 프레임 버림                   │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [F2] [F3] [F4] [F5]    │ ← F6 도착               │
│     │  ↓                          │                         │
│     │ 버림 [F2] [F3] [F4] [F5] [F6]│                         │
│     └─────────────────────────────┘                         │
│                                                             │
│  2. Skip Intermediate                                       │
│     최신과 가장 오래된 것만 유지                             │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [--] [--] [--] [F5]    │                         │
│     └─────────────────────────────┘                         │
│                                                             │
│  3. Keyframe Only                                           │
│     N 프레임마다 하나만 처리                                 │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [--] [--] [F4] [--] [--]│                        │
│     └─────────────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**문제**: 모든 프레임에 모든 Extractor의 OBS가 있지 않습니다.

---

## 문제 3: OBS 동기화 지연

여러 Extractor에서 같은 프레임에 대한 OBS(Observation) 메시지가 서로 다른 시간에 Fusion에 도착합니다.

```
Frame #100 에 대한 OBS 도착 타이밍:

시간 →   0ms     20ms    40ms    60ms    80ms   100ms   120ms
         │       │       │       │       │       │       │
Ext-D    ●───────────────────────────────────────────────────
         │  도착 (8ms)
         │       │       │       │       │       │       │
Ext-B    ────────●───────────────────────────────────────────
                 │  도착 (22ms)
                 │       │       │       │       │       │
Ext-A    ────────────────────────●───────────────────────────
                                 │  도착 (45ms)
                                 │       │       │       │
Ext-C    ────────────────────────────────────────────●───────
                                                     │  도착 (95ms)
         │       │       │       │       │       │   │
         └───────┴───────┴───────┴───────┴───────┴───┘
                                                     │
                                          Alignment Window
                                             (100ms)
```

**문제**: 윈도우 기반 분석은 가변적인 지연을 추가합니다.

---

## 해결책: ExtractorOrchestrator

동일 프로세스 내 사용 시 Orchestrator가 병렬 실행 + 타임아웃으로 동기화를 단순화합니다.

```
                    ┌─────────────────────────────────────┐
                    │      ExtractorOrchestrator          │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
        ThreadPool               ThreadPool               ThreadPool
        ┌───────┐               ┌───────┐               ┌───────┐
        │ Ext-A │               │ Ext-B │               │ Ext-C │
        └───┬───┘               └───┬───┘               └───┬───┘
            │                       │                       │
            │ 42ms                  │ 22ms                  │ 58ms
            ▼                       ▼                       ▼
        ┌───────────────────────────────────────────────────────┐
        │              as_completed(timeout=150ms)              │
        │                                                       │
        │   Ext-A ────────────────┐                             │
        │   Ext-B ────┐           │                             │
        │   Ext-D ────┤           │                             │
        │             ▼           ▼                             │
        │           수집        수집                            │
        │                                     Ext-C 도착        │
        │                                         │             │
        │                                         ▼             │
        │                                       수집            │
        └───────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              List[Observation]
```

### 알고리즘: 타임아웃 기반 수집

```python
from visualpath.process import ExtractorOrchestrator

class ExtractorOrchestrator:
    def extract_all(self, frame):
        # 모든 Extractor를 스레드 풀에 제출
        futures = {executor.submit(ext.extract, frame): ext for ext in self._extractors}
        observations = []

        # 타임아웃으로 결과 수집
        for future in as_completed(futures, timeout=self._timeout):
            try:
                obs = future.result()
                observations.append(obs)
            except TimeoutError:
                # 늦은 Extractor는 건너뜀
                _hub.emit(FrameDropRecord(reason="timeout"))

        return observations
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `timeout` | 0.15 | 프레임당 최대 대기 시간 (초) |
| `max_workers` | 4 | ThreadPool 워커 수 |

### 트레이드오프

| 타임아웃 짧게 | 타임아웃 길게 |
|---------------|---------------|
| 빠른 응답 | 더 완전한 데이터 |
| 느린 Extractor 누락 | 전체 지연 증가 |
| 10 FPS 달성 용이 | 정확도 우선 |

---

## 해결책: 시간 윈도우 정렬

분산 처리 (A-B*-C) 환경에서 FusionProcess가 OBS를 정렬합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    FusionProcess                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  OBS 버퍼 (frame_id → List[OBS])                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Frame 98: [A, B, C, D] ✓ 완료                       │   │
│  │ Frame 99: [A, B, C, D] ✓ 완료                       │   │
│  │ Frame 100: [D, B, A] ← C 대기중                      │   │
│  │ Frame 101: [D, B] ← A, C 대기중                      │   │
│  │ Frame 102: [D] ← 도착 중...                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  워터마크: Frame 99 (100ms 경과)                            │
│            │                                                │
│            ▼                                                │
│  Frame 98, 99 처리 → Fusion 결정 → 결과 전송                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 알고리즘: 윈도우 기반 정렬

```python
from visualpath.process import FusionProcess

ALIGNMENT_WINDOW_NS = 100_000_000  # 100ms

def _process_aligned_observations(self):
    current_t_ns = max(self._frame_timestamps.values())

    for frame_id in list(self._obs_buffer.keys()):
        t_ns = self._frame_timestamps[frame_id]
        age_ns = current_t_ns - t_ns

        if age_ns > ALIGNMENT_WINDOW_NS:
            # 100ms 지났으면 도착한 OBS로 처리
            observations = self._obs_buffer.pop(frame_id)
            self._process_frame(frame_id, observations)

            # 동기화 지연 기록
            if age_ns > ALIGNMENT_WINDOW_NS * 1.5:
                missing = self._get_missing_sources(frame_id)
                _hub.emit(SyncDelayRecord(
                    frame_id=frame_id,
                    delay_ms=(age_ns - ALIGNMENT_WINDOW_NS) / 1_000_000,
                    waiting_for=missing
                ))
```

### 동작 방식

1. **OBS 버퍼링**: 수신한 OBS 메시지를 frame_id별로 저장
2. **윈도우 대기**: 100ms 경과 전까지 프레임 처리 보류
3. **가용 데이터 처리**: 윈도우 내 도착한 OBS로 처리
4. **다음 진행**: 처리된 프레임을 버퍼에서 삭제

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `alignment_window_ns` | 100,000,000 | 정렬 윈도우 (100ms) |
| `late_arrival_threshold` | 1.5 | 지연 경고 배수 |

### 트레이드오프

| 현재 접근법 (단순 시간 윈도우) | |
|------|------|
| **장점** | **단점** |
| 단순한 구현 | 고정 지연 (100ms) |
| 현재 용도에 적합 | 늦게 도착하는 OBS 누락 |
| 낮은 오버헤드 | 순서 역전 처리 안됨 |

---

## Observability 연동

타이밍/동기화 관련 trace 레코드를 통해 문제를 진단합니다.

### 타이밍 레코드

```python
from visualpath.observability import TimingRecord, FrameDropRecord, SyncDelayRecord

TimingRecord(
    frame_id=100,
    component="extractor_a",
    processing_ms=58.2,
    queue_depth=3,
)

FrameDropRecord(
    frame_id=1500,
    dropped_frame_ids=[1498, 1499],
    reason="backpressure",    # 또는 "timeout"
)

SyncDelayRecord(
    frame_id=100,
    delay_ms=45.0,
    waiting_for=["extractor_c"],
)
```

### 동기화 상태 시각화 (VERBOSE)

```
┌─────────────────────────────────────────────────────────────┐
│ Sync Status                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Frame 100:  Ext-A ✓   Ext-B ✓   Ext-C ⏳   Ext-D ✓         │
│             ├──42ms──┤├─22ms─┤├──??ms──┤   ├──8ms──┤        │
│                                   │                         │
│                                   └─ 대기중 (55ms 경과)     │
│                                                             │
│ Alignment Window: ████████████████░░░░ 55/100ms            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 타이밍 분석 워크플로우

```bash
# 병목 식별
cat trace.jsonl | jq -s '
    [.[] | select(.record_type=="timing")]
    | group_by(.component)
    | map({
        component: .[0].component,
        avg_ms: ([.[].processing_ms] | add / length),
        max_ms: ([.[].processing_ms] | max)
      })'

# 동기화 지연이 발생한 프레임 찾기
cat trace.jsonl | jq 'select(.record_type=="sync_delay")'

# 느린 컴포넌트 확인 (50ms 초과)
cat trace.jsonl | jq 'select(.record_type=="timing") | select(.processing_ms > 50)'

# 프레임 드롭 원인 분석
cat trace.jsonl | jq 'select(.record_type=="frame_drop") | {reason, count: (.dropped_frame_ids | length)}'
```

### 일반적인 문제와 해결책

| 증상 | 원인 | 해결책 |
|------|------|--------|
| 특정 Extractor 지연 많음 | CPU에서 Heavy ML 느림 | GPU 사용 또는 해당 Extractor 제외 |
| 큐 깊이 계속 증가 | 처리가 입력보다 느림 | target_fps 낮추거나 프레임 스킵 |
| OBS 누락 | Extractor 타임아웃 | timeout 늘리거나 해당 Extractor 제외 |
| 트리거 지연 | alignment_window 너무 김 | 윈도우 줄이기 (정확도 트레이드오프) |
| 불규칙한 FPS | GC 또는 I/O 스파이크 | 버퍼 크기 조정, 메모리 프로파일링 |

### 타이밍 측정 지점

| 위치 | 측정 내용 | 레코드 유형 |
|------|----------|-------------|
| Extractor.extract() | 단일 추출 시간 | `TimingRecord` |
| ExtractorProcess._process_frame() | 전체 프레임 처리 | `TimingRecord` |
| ExtractorOrchestrator.extract_all() | 병렬 추출 | `TimingRecord` |
| FusionProcess._process_frame_observations() | Fusion 결정 시간 | `TimingRecord` |

모든 타이밍은 `time.perf_counter_ns()`를 사용하여 나노초 정밀도로 캡처됩니다.

---

## 설정 가이드

### 실시간 처리 (10 FPS 목표)

```python
from visualpath.process import FusionProcess, ExtractorOrchestrator

FusionProcess(
    alignment_window_ns=100_000_000,  # 100ms - 10 FPS에서 한 프레임
)

ExtractorOrchestrator(
    timeout=0.15,  # 프레임당 최대 150ms
)
```

### 배치 처리 (오프라인)

```python
# 정확도를 위해 더 긴 윈도우 사용 가능
FusionProcess(
    alignment_window_ns=200_000_000,  # 200ms
)

ExtractorOrchestrator(
    timeout=1.0,  # 느린 처리 허용
)
```

### 저지연 (라이브 프리뷰)

```python
# 정확도보다 속도 우선
FusionProcess(
    alignment_window_ns=50_000_000,  # 50ms
)

# 느린 Extractor 건너뛰기
orchestrator = ExtractorOrchestrator(
    extractors=[FastExtractor(), LightExtractor()],  # heavy 제외
    timeout=0.08,
)
```

---

## Pathway 백엔드

> **상태: 구현 완료** (2026-02-03)

Pathway 스트리밍 엔진을 실행 백엔드로 통합하여 위에서 설명한 동기화 문제들을 해결합니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  vp.process_video("video.mp4", backend="pathway")                 │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼─────────────────────────────┐
│  ExecutionBackend ABC   │                              │
│  ┌──────────────────────┴───────────────────────────┐ │
│  │         run() / run_graph()                      │ │
│  └──────────────────────────────────────────────────┘ │
│       │                              │                 │
│  ┌────┴────┐                   ┌────┴────┐           │
│  │ Simple  │                   │ Pathway │           │
│  │ Backend │                   │ Backend │           │
│  │(기본)   │                   │  (NEW)  │           │
│  └─────────┘                   └─────────┘           │
└────────────────────────────────────────────────────────┘
```

### Pathway가 제공하는 기능

| 기능 | SimpleBackend | PathwayBackend |
|------|---------------|----------------|
| 윈도우 정렬 | 100ms 고정, 수동 버퍼 | TumblingWindow + watermark |
| Late arrival | 드롭 | `allowed_lateness` 처리 |
| 백프레셔 | 수동 큐 관리 | Rust 엔진 내장 |
| 병렬 처리 | ThreadPoolExecutor | Pathway workers |
| Multi-path 동기화 | JoinNode (수동) | `interval_join` |

### 사용법

```python
import visualpath as vp

# Simple 백엔드 (기본값)
result = vp.process_video("video.mp4", modules=[face_detector], backend="simple")

# Pathway 백엔드 (스트리밍)
result = vp.process_video("video.mp4", modules=[face_detector], backend="pathway")
```

### PathwayBackend 설정

```python
from visualpath.backends.pathway import PathwayBackend

backend = PathwayBackend(
    window_ns=100_000_000,           # 100ms 윈도우
    allowed_lateness_ns=50_000_000,  # 50ms 지연 허용
    autocommit_ms=100,               # 커밋 간격
    n_workers=2,                     # 워커 수
)
```

### Operator 매핑

| visualpath | Pathway | 설명 |
|-----------|---------|------|
| `Extractor.extract()` | `flat_map()` | Frame → Observation |
| `JoinNode` (100ms) | `interval_join()` | 시간 기반 동기화 |
| `BaseFusion.update()` | `stateful_map()` | 상태 유지 Fusion |
| 백프레셔 | 내장 | Rust 엔진 자동 처리 |

### 설치

```bash
# Pathway 백엔드 설치
pip install visualpath[pathway]
```

### 구현 모듈

```
visualpath/backends/pathway/
├── __init__.py      # PathwayBackend export
├── backend.py       # PathwayBackend 구현
├── connector.py     # VideoConnectorSubject (Frame → Pathway)
├── operators.py     # Extractor/Fusion UDF 래퍼
└── converter.py     # FlowGraph → Pathway 변환
```

### 이벤트 시간 윈도잉 예시

```python
# Pathway 내부에서 사용되는 윈도잉
frames_table.windowby(
    pw.this.t_ns,
    window=pw.temporal.tumbling(window_ns),
    behavior=pw.temporal.common_behavior(
        delay=allowed_lateness_ns
    ),
)
```

### Multi-Extractor 동기화

```python
# interval_join으로 여러 Extractor 결과 동기화
left_obs.interval_join(
    right_obs,
    pw.left.t_ns,
    pw.right.t_ns,
    pw.temporal.interval(-window_ns, window_ns),
)
```
