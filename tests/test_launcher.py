"""Tests for WorkerLauncher and worker implementations."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional
from unittest import mock
import time

from visualpath.core import BaseExtractor, Observation, IsolationLevel
from visualpath.process import (
    WorkerLauncher,
    InlineWorker,
    ThreadWorker,
    ProcessWorker,
    VenvWorker,
    WorkerResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class SimpleExtractor(BaseExtractor):
    """Simple extractor for testing."""

    def __init__(self, delay_ms: float = 0, fail: bool = False):
        self._delay_ms = delay_ms
        self._fail = fail
        self._initialized = False
        self._cleaned_up = False
        self._extract_count = 0

    @property
    def name(self) -> str:
        return "simple"

    def extract(self, frame) -> Optional[Observation]:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)

        if self._fail:
            raise RuntimeError("Extraction failed")

        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"count": self._extract_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


# =============================================================================
# WorkerResult Tests
# =============================================================================


class TestWorkerResult:
    """Tests for WorkerResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        obs = Observation(source="test", frame_id=1, t_ns=1000)
        result = WorkerResult(observation=obs, timing_ms=10.5)

        assert result.observation is not None
        assert result.error is None
        assert result.timing_ms == 10.5

    def test_error_result(self):
        """Test error result."""
        result = WorkerResult(observation=None, error="Something failed", timing_ms=5.0)

        assert result.observation is None
        assert result.error == "Something failed"


# =============================================================================
# InlineWorker Tests
# =============================================================================


class TestInlineWorker:
    """Tests for InlineWorker."""

    def test_lifecycle(self):
        """Test worker start/stop lifecycle."""
        extractor = SimpleExtractor()
        worker = InlineWorker(extractor)

        assert not worker.is_running
        assert not extractor._initialized

        worker.start()

        assert worker.is_running
        assert extractor._initialized

        worker.stop()

        assert not worker.is_running
        assert extractor._cleaned_up

    def test_process_success(self):
        """Test successful frame processing."""
        extractor = SimpleExtractor()
        worker = InlineWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None
        assert result.observation.frame_id == 1
        assert result.error is None
        assert result.timing_ms > 0

    def test_process_error(self):
        """Test frame processing with error."""
        extractor = SimpleExtractor(fail=True)
        worker = InlineWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is None
        assert result.error is not None
        assert "failed" in result.error.lower()

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        extractor = SimpleExtractor()
        worker = InlineWorker(extractor)

        worker.start()

        for i in range(5):
            frame = MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
            result = worker.process(frame)
            assert result.observation is not None
            assert result.observation.signals["count"] == i + 1

        worker.stop()


# =============================================================================
# ThreadWorker Tests
# =============================================================================


class TestThreadWorker:
    """Tests for ThreadWorker."""

    def test_lifecycle(self):
        """Test worker start/stop lifecycle."""
        extractor = SimpleExtractor()
        worker = ThreadWorker(extractor)

        assert not worker.is_running
        assert not extractor._initialized

        worker.start()

        assert worker.is_running
        assert extractor._initialized

        worker.stop()

        assert not worker.is_running
        assert extractor._cleaned_up

    def test_process_success(self):
        """Test successful frame processing."""
        extractor = SimpleExtractor()
        worker = ThreadWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None
        assert result.observation.frame_id == 1
        assert result.error is None

    def test_process_with_delay(self):
        """Test processing with delay (simulating work)."""
        extractor = SimpleExtractor(delay_ms=50)
        worker = ThreadWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None
        assert result.timing_ms >= 50

    def test_process_not_running(self):
        """Test processing when worker not running."""
        extractor = SimpleExtractor()
        worker = ThreadWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        result = worker.process(frame)

        assert result.observation is None
        assert result.error is not None
        assert "not running" in result.error.lower()

    def test_process_async(self):
        """Test async processing."""
        extractor = SimpleExtractor(delay_ms=50)
        worker = ThreadWorker(extractor)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        future = worker.process_async(frame)

        # Future should complete
        obs = future.result(timeout=1.0)
        worker.stop()

        assert obs is not None
        assert obs.frame_id == 1


# =============================================================================
# ProcessWorker Tests
# =============================================================================


class TestProcessWorker:
    """Tests for ProcessWorker.

    These tests mock ZMQ as unavailable to force inline fallback,
    since the real subprocess tests are in test_venv_worker.py.
    """

    def test_lifecycle(self):
        """Test worker start/stop lifecycle."""
        extractor = SimpleExtractor()

        # Mock ZMQ unavailable to force inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(extractor)

            worker.start()
            assert worker.is_running

            worker.stop()
            assert not worker.is_running

    def test_process_success(self):
        """Test successful frame processing."""
        extractor = SimpleExtractor()
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        # Mock ZMQ unavailable to force inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(extractor)

            worker.start()
            result = worker.process(frame)
            worker.stop()

            assert result.observation is not None
            assert result.observation.frame_id == 1


# =============================================================================
# VenvWorker Tests
# =============================================================================


class TestVenvWorker:
    """Tests for VenvWorker (currently falls back to inline)."""

    def test_lifecycle(self):
        """Test worker start/stop lifecycle."""
        extractor = SimpleExtractor()
        worker = VenvWorker(extractor, venv_path="/tmp/test-venv")

        worker.start()
        assert worker.is_running

        worker.stop()
        assert not worker.is_running

    def test_process_success(self):
        """Test successful frame processing."""
        extractor = SimpleExtractor()
        worker = VenvWorker(extractor, venv_path="/tmp/test-venv")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None


# =============================================================================
# WorkerLauncher Factory Tests
# =============================================================================


class TestWorkerLauncher:
    """Tests for WorkerLauncher factory."""

    def test_create_inline(self):
        """Test creating inline worker."""
        extractor = SimpleExtractor()
        worker = WorkerLauncher.create(IsolationLevel.INLINE, extractor)

        assert isinstance(worker, InlineWorker)

    def test_create_thread(self):
        """Test creating thread worker."""
        extractor = SimpleExtractor()
        worker = WorkerLauncher.create(IsolationLevel.THREAD, extractor)

        assert isinstance(worker, ThreadWorker)

    def test_create_process(self):
        """Test creating process worker."""
        extractor = SimpleExtractor()
        worker = WorkerLauncher.create(IsolationLevel.PROCESS, extractor)

        assert isinstance(worker, ProcessWorker)

    def test_create_venv(self):
        """Test creating venv worker."""
        extractor = SimpleExtractor()
        worker = WorkerLauncher.create(
            IsolationLevel.VENV,
            extractor,
            venv_path="/tmp/test-venv",
        )

        assert isinstance(worker, VenvWorker)

    def test_create_venv_without_path_raises(self):
        """Test that creating venv worker without path raises."""
        extractor = SimpleExtractor()

        with pytest.raises(ValueError, match="venv_path"):
            WorkerLauncher.create(IsolationLevel.VENV, extractor)

    def test_create_container_not_implemented(self):
        """Test that container isolation is not implemented."""
        extractor = SimpleExtractor()

        with pytest.raises(NotImplementedError):
            WorkerLauncher.create(IsolationLevel.CONTAINER, extractor)

    def test_worker_with_kwargs(self):
        """Test passing kwargs to worker."""
        extractor = SimpleExtractor()
        worker = WorkerLauncher.create(
            IsolationLevel.THREAD,
            extractor,
            queue_size=20,
        )

        assert isinstance(worker, ThreadWorker)
        assert worker._queue_size == 20
