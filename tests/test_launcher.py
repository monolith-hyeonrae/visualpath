"""Tests for WorkerLauncher and worker implementations."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional
from unittest import mock
import time

from visualpath.core import Module, Observation, IsolationLevel
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


class SimpleModule(Module):
    """Simple module for testing."""

    def __init__(self, delay_ms: float = 0, fail: bool = False):
        self._delay_ms = delay_ms
        self._fail = fail
        self._initialized = False
        self._cleaned_up = False
        self._extract_count = 0

    @property
    def name(self) -> str:
        return "simple"

    def process(self, frame, deps=None) -> Optional[Observation]:
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
        module = SimpleModule()
        worker = InlineWorker(module)

        assert not worker.is_running
        assert not module._initialized

        worker.start()

        assert worker.is_running
        assert module._initialized

        worker.stop()

        assert not worker.is_running
        assert module._cleaned_up

    def test_process_success(self):
        """Test successful frame processing."""
        module = SimpleModule()
        worker = InlineWorker(module)
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
        module = SimpleModule(fail=True)
        worker = InlineWorker(module)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is None
        assert result.error is not None
        assert "failed" in result.error.lower()

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        module = SimpleModule()
        worker = InlineWorker(module)

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
        module = SimpleModule()
        worker = ThreadWorker(module)

        assert not worker.is_running
        assert not module._initialized

        worker.start()

        assert worker.is_running
        assert module._initialized

        worker.stop()

        assert not worker.is_running
        assert module._cleaned_up

    def test_process_success(self):
        """Test successful frame processing."""
        module = SimpleModule()
        worker = ThreadWorker(module)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None
        assert result.observation.frame_id == 1
        assert result.error is None

    def test_process_with_delay(self):
        """Test processing with delay (simulating work)."""
        module = SimpleModule(delay_ms=50)
        worker = ThreadWorker(module)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        worker.start()
        result = worker.process(frame)
        worker.stop()

        assert result.observation is not None
        assert result.timing_ms >= 50

    def test_process_not_running(self):
        """Test processing when worker not running."""
        module = SimpleModule()
        worker = ThreadWorker(module)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        result = worker.process(frame)

        assert result.observation is None
        assert result.error is not None
        assert "not running" in result.error.lower()

    def test_process_async(self):
        """Test async processing."""
        module = SimpleModule(delay_ms=50)
        worker = ThreadWorker(module)
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
        module = SimpleModule()

        # Mock ZMQ unavailable to force inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(module)

            worker.start()
            assert worker.is_running

            worker.stop()
            assert not worker.is_running

    def test_process_success(self):
        """Test successful frame processing."""
        module = SimpleModule()
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        # Mock ZMQ unavailable to force inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(module)

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
        module = SimpleModule()
        worker = VenvWorker(module, venv_path="/tmp/test-venv")

        worker.start()
        assert worker.is_running

        worker.stop()
        assert not worker.is_running

    def test_process_success(self):
        """Test successful frame processing."""
        module = SimpleModule()
        worker = VenvWorker(module, venv_path="/tmp/test-venv")
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
        module = SimpleModule()
        worker = WorkerLauncher.create(IsolationLevel.INLINE, module)

        assert isinstance(worker, InlineWorker)

    def test_create_thread(self):
        """Test creating thread worker."""
        module = SimpleModule()
        worker = WorkerLauncher.create(IsolationLevel.THREAD, module)

        assert isinstance(worker, ThreadWorker)

    def test_create_process(self):
        """Test creating process worker."""
        module = SimpleModule()
        worker = WorkerLauncher.create(IsolationLevel.PROCESS, module)

        assert isinstance(worker, ProcessWorker)

    def test_create_venv(self):
        """Test creating venv worker."""
        module = SimpleModule()
        worker = WorkerLauncher.create(
            IsolationLevel.VENV,
            module,
            venv_path="/tmp/test-venv",
        )

        assert isinstance(worker, VenvWorker)

    def test_create_venv_without_path_raises(self):
        """Test that creating venv worker without path raises."""
        module = SimpleModule()

        with pytest.raises(ValueError, match="venv_path"):
            WorkerLauncher.create(IsolationLevel.VENV, module)

    def test_create_container_not_implemented(self):
        """Test that container isolation is not implemented."""
        module = SimpleModule()

        with pytest.raises(NotImplementedError):
            WorkerLauncher.create(IsolationLevel.CONTAINER, module)

    def test_worker_with_kwargs(self):
        """Test passing kwargs to worker."""
        module = SimpleModule()
        worker = WorkerLauncher.create(
            IsolationLevel.THREAD,
            module,
            queue_size=20,
        )

        assert isinstance(worker, ThreadWorker)
        assert worker._queue_size == 20
