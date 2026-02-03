"""Tests for SimpleBackend composable components.

Tests for:
- Schedulers (frame selection strategies)
- Synchronizers (observation alignment)
- Buffers (backpressure management)
- Executors (parallel execution)
- SimpleBackend composition
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import time

from visualpath.core import BaseExtractor, Observation, BaseFusion, FusionResult


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class SlowExtractor(BaseExtractor):
    """Extractor with configurable delay."""

    def __init__(self, name: str, delay_ms: float = 10.0, value: float = 0.5):
        self._name = name
        self._delay_ms = delay_ms
        self._value = value
        self._extract_count = 0

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame, deps=None) -> Optional[Observation]:
        time.sleep(self._delay_ms / 1000)
        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class CountingExtractor(BaseExtractor):
    """Extractor that counts calls."""

    def __init__(self, name: str, value: float = 0.5):
        self._name = name
        self._value = value
        self._extract_count = 0

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame, deps=None) -> Optional[Observation]:
        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value, "count": self._extract_count},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class ThresholdFusion(BaseFusion):
    """Simple fusion for testing."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._update_count = 0

    def update(self, observation: Observation) -> FusionResult:
        self._update_count += 1
        value = observation.signals.get("value", 0)
        if value > self._threshold:
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=observation.t_ns,
                pre_sec=2.0,
                post_sec=2.0,
                label="threshold",
                score=value,
            )
            return FusionResult(should_trigger=True, trigger=trigger, score=value)
        return FusionResult(should_trigger=False)

    def reset(self) -> None:
        self._update_count = 0

    @property
    def is_gate_open(self) -> bool:
        return True

    @property
    def in_cooldown(self) -> bool:
        return False


def make_frame(frame_id: int = 1, t_ns: int = 1_000_000) -> MockFrame:
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_ns,
        data=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    return [make_frame(frame_id=i, t_ns=i * interval_ns) for i in range(count)]


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestPassThroughScheduler:
    """Tests for PassThroughScheduler."""

    def test_passes_all_frames(self):
        from visualpath.backends.simple import PassThroughScheduler

        scheduler = PassThroughScheduler()
        frames = make_frames(10)

        result = list(scheduler.schedule(iter(frames)))

        assert len(result) == 10
        assert scheduler.stats.frames_received == 10
        assert scheduler.stats.frames_processed == 10
        assert scheduler.stats.frames_dropped == 0

    def test_reset(self):
        from visualpath.backends.simple import PassThroughScheduler

        scheduler = PassThroughScheduler()
        list(scheduler.schedule(iter(make_frames(5))))

        scheduler.reset()

        assert scheduler.stats.frames_received == 0


class TestKeyframeScheduler:
    """Tests for KeyframeScheduler."""

    def test_every_third_frame(self):
        from visualpath.backends.simple import KeyframeScheduler

        scheduler = KeyframeScheduler(every_n=3)
        frames = make_frames(9)

        result = list(scheduler.schedule(iter(frames)))

        assert len(result) == 3  # Frames 0, 3, 6
        assert scheduler.stats.frames_dropped == 6

    def test_every_frame(self):
        from visualpath.backends.simple import KeyframeScheduler

        scheduler = KeyframeScheduler(every_n=1)
        frames = make_frames(5)

        result = list(scheduler.schedule(iter(frames)))

        assert len(result) == 5

    def test_invalid_every_n(self):
        from visualpath.backends.simple import KeyframeScheduler

        with pytest.raises(ValueError):
            KeyframeScheduler(every_n=0)


class TestSkipOldestScheduler:
    """Tests for SkipOldestScheduler."""

    def test_basic(self):
        from visualpath.backends.simple import SkipOldestScheduler

        scheduler = SkipOldestScheduler(max_queue=10)
        frames = make_frames(5)

        result = list(scheduler.schedule(iter(frames)))

        assert len(result) == 5
        assert scheduler.stats.frames_received == 5
        assert scheduler.stats.frames_processed == 5


# =============================================================================
# Synchronizer Tests
# =============================================================================


class TestNoSyncSynchronizer:
    """Tests for NoSyncSynchronizer."""

    def test_immediate_windows(self):
        from visualpath.backends.simple import NoSyncSynchronizer

        sync = NoSyncSynchronizer()
        obs = Observation(source="test", frame_id=1, t_ns=1000, signals={})

        windows = sync.add(obs)

        assert len(windows) == 1
        assert len(windows[0].all_observations()) == 1

    def test_stats(self):
        from visualpath.backends.simple import NoSyncSynchronizer

        sync = NoSyncSynchronizer()
        for i in range(5):
            sync.add(Observation(source="test", frame_id=i, t_ns=i * 1000, signals={}))

        assert sync.stats.windows_completed == 5
        assert sync.stats.observations_synced == 5


class TestTimeWindowSync:
    """Tests for TimeWindowSync."""

    def test_single_window(self):
        from visualpath.backends.simple import TimeWindowSync

        sync = TimeWindowSync(window_ns=100_000_000)  # 100ms

        # Add single observation
        obs1 = Observation(source="a", frame_id=1, t_ns=10_000_000, signals={})
        windows = sync.add(obs1)

        assert len(windows) == 0  # Not closed yet
        assert sync.stats.observations_synced == 1

    def test_late_arrival_dropped(self):
        from visualpath.backends.simple import TimeWindowSync

        sync = TimeWindowSync(window_ns=100_000_000, allowed_lateness_ns=0)

        # Advance watermark
        sync.add(Observation(source="a", frame_id=1, t_ns=500_000_000, signals={}))

        # Late observation
        late = Observation(source="b", frame_id=0, t_ns=100_000_000, signals={})
        windows = sync.add(late)

        assert sync.stats.observations_dropped == 1

    def test_flush(self):
        from visualpath.backends.simple import TimeWindowSync

        sync = TimeWindowSync(window_ns=100_000_000)

        sync.add(Observation(source="a", frame_id=1, t_ns=10_000_000, signals={}))

        windows = sync.flush()

        assert len(windows) == 1
        assert len(windows[0].all_observations()) == 1

    def test_multiple_windows(self):
        from visualpath.backends.simple import TimeWindowSync

        sync = TimeWindowSync(window_ns=100_000_000)  # 100ms

        # Window 1: 0-100ms
        sync.add(Observation(source="a", frame_id=1, t_ns=50_000_000, signals={}))
        # Window 2: 100-200ms
        sync.add(Observation(source="a", frame_id=2, t_ns=150_000_000, signals={}))
        # Window 3: 300-400ms (triggers flush of earlier windows)
        windows = sync.add(Observation(source="a", frame_id=3, t_ns=350_000_000, signals={}))

        # At least one window should be closed
        assert sync.stats.observations_synced == 3


class TestBarrierSync:
    """Tests for BarrierSync."""

    def test_waits_for_all_sources(self):
        from visualpath.backends.simple import BarrierSync

        sync = BarrierSync(sources=["a", "b"])

        obs_a = Observation(source="a", frame_id=1, t_ns=1000, signals={})
        windows1 = sync.add(obs_a)
        assert len(windows1) == 0  # Waiting for b

        obs_b = Observation(source="b", frame_id=1, t_ns=1000, signals={})
        windows2 = sync.add(obs_b)
        assert len(windows2) == 1  # Complete

    def test_empty_sources_raises(self):
        from visualpath.backends.simple import BarrierSync

        with pytest.raises(ValueError):
            BarrierSync(sources=[])


# =============================================================================
# Buffer Tests
# =============================================================================


class TestBoundedBuffer:
    """Tests for BoundedBuffer."""

    def test_basic_operations(self):
        from visualpath.backends.simple import BoundedBuffer

        buffer = BoundedBuffer(max_size=3)

        assert buffer.is_empty
        assert not buffer.is_full

        buffer.put(1)
        buffer.put(2)
        buffer.put(3)

        assert buffer.is_full
        assert buffer.size == 3
        assert buffer.get() == 1
        assert buffer.get() == 2

    def test_drop_oldest(self):
        from visualpath.backends.simple import BoundedBuffer, OverflowPolicy

        buffer = BoundedBuffer(max_size=2, overflow_policy=OverflowPolicy.DROP_OLDEST)

        buffer.put(1)
        buffer.put(2)
        buffer.put(3)  # Drops 1

        assert buffer.get() == 2
        assert buffer.get() == 3
        assert buffer.stats.items_dropped == 1

    def test_drop_newest(self):
        from visualpath.backends.simple import BoundedBuffer, OverflowPolicy

        buffer = BoundedBuffer(max_size=2, overflow_policy=OverflowPolicy.DROP_NEWEST)

        buffer.put(1)
        buffer.put(2)
        result = buffer.put(3)  # Dropped

        assert result is False
        assert buffer.get() == 1
        assert buffer.get() == 2
        assert buffer.stats.items_dropped == 1


class TestPriorityBuffer:
    """Tests for PriorityBuffer."""

    def test_priority_ordering(self):
        from visualpath.backends.simple import PriorityBuffer

        buffer = PriorityBuffer(max_size=3, priority_fn=lambda x: x)

        buffer.put(1)
        buffer.put(3)
        buffer.put(2)

        # Should get highest priority first
        assert buffer.get() == 3
        assert buffer.get() == 2
        assert buffer.get() == 1

    def test_evicts_lowest_priority(self):
        from visualpath.backends.simple import PriorityBuffer

        buffer = PriorityBuffer(max_size=2, priority_fn=lambda x: x)

        buffer.put(2)
        buffer.put(3)
        buffer.put(1)  # Lowest priority, evicted immediately

        assert buffer.get() == 3
        assert buffer.get() == 2
        assert buffer.stats.items_dropped == 1


# =============================================================================
# Executor Tests
# =============================================================================


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    def test_executes_all(self):
        from visualpath.backends.simple import SequentialExecutor

        executor = SequentialExecutor()
        ext1 = CountingExtractor("ext1", value=0.5)
        ext2 = CountingExtractor("ext2", value=0.7)
        frame = make_frame()

        results = executor.execute(frame, [ext1, ext2])

        assert len(results) == 2
        assert all(r.success for r in results)
        assert ext1._extract_count == 1
        assert ext2._extract_count == 1

    def test_stats(self):
        from visualpath.backends.simple import SequentialExecutor

        executor = SequentialExecutor()
        ext = CountingExtractor("test")

        for frame in make_frames(3):
            executor.execute(frame, [ext])

        assert executor.stats.frames_processed == 3
        assert executor.stats.extractions_completed == 3


class TestThreadPoolExecutor:
    """Tests for ThreadPoolExecutor."""

    def test_parallel_execution(self):
        from visualpath.backends.simple import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)
        ext1 = SlowExtractor("ext1", delay_ms=20)
        ext2 = SlowExtractor("ext2", delay_ms=20)
        frame = make_frame()

        start = time.perf_counter()
        results = executor.execute(frame, [ext1, ext2])
        elapsed = (time.perf_counter() - start) * 1000

        assert len(results) == 2
        assert all(r.success for r in results)
        # Should be faster than sequential (40ms)
        assert elapsed < 35

        executor.shutdown()


class TestTimeoutExecutor:
    """Tests for TimeoutExecutor."""

    def test_timeout_slow_extractor(self):
        from visualpath.backends.simple import TimeoutExecutor

        executor = TimeoutExecutor(timeout_ms=50, max_workers=2)
        fast_ext = CountingExtractor("fast")
        slow_ext = SlowExtractor("slow", delay_ms=200)
        frame = make_frame()

        results = executor.execute(frame, [fast_ext, slow_ext])

        # Fast should succeed, slow should timeout
        fast_result = next(r for r in results if r.extractor_name == "fast")
        slow_result = next(r for r in results if r.extractor_name == "slow")

        assert fast_result.success
        assert not slow_result.success
        assert slow_result.error == "timeout"

        executor.shutdown()


# =============================================================================
# SimpleBackend Integration Tests
# =============================================================================


class TestSimpleBackendComposition:
    """Tests for SimpleBackend with different component combinations."""

    def test_default_backend(self):
        from visualpath.backends.simple import SimpleBackend

        backend = SimpleBackend()
        ext = CountingExtractor("test", value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(5)

        triggers = backend.run(iter(frames), [ext], fusion)

        assert len(triggers) == 5
        assert ext._extract_count == 5

    def test_with_keyframe_scheduler(self):
        from visualpath.backends.simple import SimpleBackend, KeyframeScheduler

        backend = SimpleBackend(scheduler=KeyframeScheduler(every_n=2))
        ext = CountingExtractor("test")
        frames = make_frames(6)

        backend.run(iter(frames), [ext])

        assert ext._extract_count == 3  # Frames 0, 2, 4

    def test_with_parallel_executor(self):
        from visualpath.backends.simple import SimpleBackend, ThreadPoolExecutor

        backend = SimpleBackend(executor=ThreadPoolExecutor(max_workers=2))
        ext1 = CountingExtractor("ext1")
        ext2 = CountingExtractor("ext2")
        frames = make_frames(3)

        backend.run(iter(frames), [ext1, ext2])

        assert ext1._extract_count == 3
        assert ext2._extract_count == 3

        backend.cleanup()

    def test_get_stats(self):
        from visualpath.backends.simple import SimpleBackend

        backend = SimpleBackend()
        ext = CountingExtractor("test")
        frames = make_frames(5)

        backend.run(iter(frames), [ext])
        stats = backend.get_stats()

        assert "scheduler" in stats
        assert "executor" in stats
        assert "synchronizer" in stats
        assert stats["scheduler"]["frames_processed"] == 5


class TestFactoryFunctions:
    """Tests for SimpleBackend factory functions."""

    def test_create_default_backend(self):
        from visualpath.backends.simple import create_default_backend

        backend = create_default_backend()
        assert backend is not None

    def test_create_parallel_backend(self):
        from visualpath.backends.simple import create_parallel_backend

        backend = create_parallel_backend(max_workers=2)
        assert backend is not None
        backend.cleanup()

    def test_create_parallel_backend_with_timeout(self):
        from visualpath.backends.simple import create_parallel_backend

        backend = create_parallel_backend(max_workers=2, timeout_ms=100)
        assert backend is not None
        backend.cleanup()

    def test_create_realtime_backend(self):
        from visualpath.backends.simple import create_realtime_backend

        backend = create_realtime_backend(target_fps=10)
        assert backend is not None
        backend.cleanup()

    def test_create_batch_backend(self):
        from visualpath.backends.simple import create_batch_backend

        backend = create_batch_backend(max_workers=4)
        assert backend is not None
        backend.cleanup()
