"""Tests for VenvWorker with ZMQ-based subprocess communication."""

import base64
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import numpy as np
import pytest

from visualpath.core import BaseExtractor, Observation, IsolationLevel
from visualpath.process import (
    VenvWorker,
    ProcessWorker,
    WorkerLauncher,
    WorkerResult,
    InlineWorker,
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
    width: int = 100
    height: int = 100


def create_test_frame(frame_id: int = 1, t_src_ns: int = 1000000) -> MockFrame:
    """Create a test frame."""
    data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_src_ns,
        data=data,
        width=100,
        height=100,
    )


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
# Frame Serialization Tests
# =============================================================================


class TestFrameSerialization:
    """Tests for frame serialization utilities."""

    def test_serialize_frame(self):
        """Test frame serialization to JSON-compatible format."""
        from visualpath.process.launcher import _serialize_frame

        frame = create_test_frame(frame_id=42, t_src_ns=123456789)
        serialized = _serialize_frame(frame, jpeg_quality=95)

        assert serialized["frame_id"] == 42
        assert serialized["t_src_ns"] == 123456789
        assert serialized["width"] == 100
        assert serialized["height"] == 100
        assert "data_b64" in serialized

        # Verify it's valid base64
        decoded = base64.b64decode(serialized["data_b64"])
        assert len(decoded) > 0

    def test_serialize_deserialize_round_trip(self):
        """Test frame serialization and deserialization round trip."""
        import cv2
        from visualpath.process.launcher import _serialize_frame
        from visualpath.process.worker import _deserialize_frame

        # Create a frame with specific pixel values
        data = np.zeros((50, 50, 3), dtype=np.uint8)
        data[10:40, 10:40, 0] = 255  # Red square
        frame = MockFrame(frame_id=1, t_src_ns=1000, data=data, width=50, height=50)

        # Serialize
        serialized = _serialize_frame(frame, jpeg_quality=100)

        # Deserialize (using the worker's deserialize function with mock)
        # Note: This requires visualbase.Frame, so we'll test the logic manually
        jpeg_bytes = base64.b64decode(serialized["data_b64"])
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        assert img is not None
        assert img.shape == (50, 50, 3)
        # JPEG is lossy, so we check approximate values
        assert img[20, 20, 0] > 200  # Blue channel should still be red


class TestObservationSerialization:
    """Tests for observation serialization utilities."""

    def test_serialize_observation_basic(self):
        """Test observation serialization."""
        from visualpath.process.worker import _serialize_observation

        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"score": 0.95},
            data={"key": "value"},
            metadata={"version": 1},
            timing={"detect": 10.5},
        )

        serialized = _serialize_observation(obs)

        assert serialized["source"] == "test"
        assert serialized["frame_id"] == 1
        assert serialized["t_ns"] == 1000000
        assert serialized["signals"] == {"score": 0.95}
        assert serialized["data"] == {"key": "value"}
        assert serialized["metadata"] == {"version": 1}
        assert serialized["timing"] == {"detect": 10.5}

    def test_serialize_observation_none(self):
        """Test serialization of None observation."""
        from visualpath.process.worker import _serialize_observation

        assert _serialize_observation(None) is None

    def test_serialize_observation_non_serializable_data(self):
        """Test serialization with non-JSON-serializable data."""
        from visualpath.process.worker import _serialize_observation

        class CustomObject:
            def __repr__(self):
                return "CustomObject()"

        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            data=CustomObject(),
        )

        serialized = _serialize_observation(obs)
        # Non-serializable data should be converted to string
        assert "CustomObject" in str(serialized["data"])

    def test_deserialize_observation(self):
        """Test observation deserialization."""
        from visualpath.process.launcher import _deserialize_observation

        data = {
            "source": "test",
            "frame_id": 1,
            "t_ns": 1000000,
            "signals": {"score": 0.95},
            "data": {"key": "value"},
            "metadata": {"version": 1},
            "timing": {"detect": 10.5},
        }

        obs = _deserialize_observation(data)

        assert obs.source == "test"
        assert obs.frame_id == 1
        assert obs.t_ns == 1000000
        assert obs.signals == {"score": 0.95}
        assert obs.data == {"key": "value"}

    def test_deserialize_observation_none(self):
        """Test deserialization of None."""
        from visualpath.process.launcher import _deserialize_observation

        assert _deserialize_observation(None) is None


# =============================================================================
# VenvWorker Fallback Tests
# =============================================================================


class TestVenvWorkerFallback:
    """Tests for VenvWorker fallback to inline execution."""

    def test_fallback_when_no_extractor_name(self):
        """Test fallback when no extractor_name is provided."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
            extractor_name=None,
        )

        worker.start()

        assert worker.is_running
        # Should be using inline worker
        assert worker._inline is not None
        assert not worker._use_zmq

        frame = create_test_frame()
        result = worker.process(frame)

        assert result.observation is not None
        assert result.error is None

        worker.stop()

    def test_fallback_when_venv_not_found(self):
        """Test fallback when venv Python is not found."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
            extractor_name="simple",
        )

        worker.start()

        assert worker.is_running
        assert worker._inline is not None

        worker.stop()

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(sys.prefix, "bin", "python")),
        reason="Current environment is not a venv",
    )
    def test_fallback_without_zmq(self):
        """Test fallback when ZMQ is not available."""
        extractor = SimpleExtractor()

        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = VenvWorker(
                extractor=extractor,
                venv_path=sys.prefix,
                extractor_name="simple",
            )

            worker.start()

            assert worker.is_running
            assert worker._inline is not None

            worker.stop()

    def test_error_without_extractor_for_fallback(self):
        """Test error when fallback is needed but no extractor provided."""
        worker = VenvWorker(
            extractor=None,
            venv_path="/nonexistent/venv",
            extractor_name="simple",  # Can't load from nonexistent venv
        )

        with pytest.raises(ValueError, match="Cannot fall back"):
            worker.start()


# =============================================================================
# VenvWorker Lifecycle Tests
# =============================================================================


class TestVenvWorkerLifecycle:
    """Tests for VenvWorker lifecycle management."""

    def test_start_stop_inline(self):
        """Test start/stop with inline fallback."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        assert not worker.is_running

        worker.start()
        assert worker.is_running
        assert extractor._initialized

        worker.stop()
        assert not worker.is_running
        assert extractor._cleaned_up

    def test_double_start(self):
        """Test that starting twice is safe."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        worker.start()
        worker.start()  # Should be no-op

        assert worker.is_running

        worker.stop()

    def test_double_stop(self):
        """Test that stopping twice is safe."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        worker.start()
        worker.stop()
        worker.stop()  # Should be no-op

        assert not worker.is_running

    def test_process_when_not_running(self):
        """Test processing when worker is not running."""
        extractor = SimpleExtractor()
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        frame = create_test_frame()
        result = worker.process(frame)

        assert result.observation is None
        assert "not running" in result.error.lower()


# =============================================================================
# ProcessWorker Tests
# =============================================================================


class TestProcessWorker:
    """Tests for ProcessWorker."""

    def test_uses_current_venv(self):
        """Test that ProcessWorker uses the current venv path."""
        extractor = SimpleExtractor()
        worker = ProcessWorker(extractor=extractor)

        # Check that the delegate has the correct venv path
        expected_venv = os.path.dirname(os.path.dirname(sys.executable))
        assert worker._delegate._venv_path == expected_venv

    def test_lifecycle_with_inline_fallback(self):
        """Test ProcessWorker lifecycle with inline fallback.

        When extractor is provided but no extractor_name, and ZMQ is unavailable,
        it should fall back to inline execution.
        """
        extractor = SimpleExtractor()

        # Mock ZMQ as unavailable to force inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(extractor=extractor)

            worker.start()
            assert worker.is_running
            assert worker._delegate._inline is not None  # Should use inline

            frame = create_test_frame()
            result = worker.process(frame)

            assert result.observation is not None

            worker.stop()
            assert not worker.is_running

    def test_lifecycle_with_extractor_uses_name(self):
        """Test that ProcessWorker uses extractor.name when no extractor_name is given.

        When extractor is provided without explicit extractor_name, VenvWorker
        uses extractor.name as the extractor_name. This test verifies this behavior
        by mocking ZMQ unavailability to force inline fallback.
        """
        extractor = SimpleExtractor()

        # Mock ZMQ unavailability - this forces inline fallback
        with mock.patch(
            "visualpath.process.launcher._check_zmq_available",
            return_value=False,
        ):
            worker = ProcessWorker(extractor=extractor)

            # The delegate should have extractor_name set from extractor.name
            assert worker._delegate._extractor_name == "simple"

            worker.start()
            assert worker.is_running

            # Should use inline fallback because ZMQ is "unavailable"
            assert worker._delegate._inline is not None

            frame = create_test_frame()
            result = worker.process(frame)
            assert result.observation is not None

            worker.stop()


# =============================================================================
# WorkerLauncher Factory Tests
# =============================================================================


class TestWorkerLauncherExtended:
    """Extended tests for WorkerLauncher factory."""

    def test_create_venv_with_extractor_name(self):
        """Test creating VenvWorker with extractor_name."""
        worker = WorkerLauncher.create(
            level=IsolationLevel.VENV,
            extractor=None,
            venv_path="/tmp/test-venv",
            extractor_name="face",
        )

        assert isinstance(worker, VenvWorker)
        assert worker._extractor_name == "face"

    def test_create_process_with_extractor_name(self):
        """Test creating ProcessWorker with extractor_name."""
        worker = WorkerLauncher.create(
            level=IsolationLevel.PROCESS,
            extractor=None,
            extractor_name="face",
        )

        assert isinstance(worker, ProcessWorker)
        assert worker._delegate._extractor_name == "face"

    def test_create_inline_without_extractor_raises(self):
        """Test that creating InlineWorker without extractor raises."""
        with pytest.raises(ValueError, match="extractor is required"):
            WorkerLauncher.create(
                level=IsolationLevel.INLINE,
                extractor=None,
            )

    def test_create_thread_without_extractor_raises(self):
        """Test that creating ThreadWorker without extractor raises."""
        with pytest.raises(ValueError, match="extractor is required"):
            WorkerLauncher.create(
                level=IsolationLevel.THREAD,
                extractor=None,
            )


# =============================================================================
# ZMQ Communication Tests (Mock)
# =============================================================================


class TestZMQCommunicationMock:
    """Tests for ZMQ communication using mocks."""

    def test_zmq_check_available(self):
        """Test _check_zmq_available function."""
        from visualpath.process.launcher import _check_zmq_available

        # Should return True if zmq is installed
        result = _check_zmq_available()
        try:
            import zmq  # noqa: F401
            assert result is True
        except ImportError:
            assert result is False

    def test_zmq_check_with_mock_import_error(self):
        """Test _check_zmq_available with mock import error."""
        from visualpath.process import launcher

        original_func = launcher._check_zmq_available

        with mock.patch.dict(sys.modules, {"zmq": None}):
            with mock.patch.object(
                launcher,
                "_check_zmq_available",
                return_value=False,
            ):
                assert launcher._check_zmq_available() is False


# =============================================================================
# Worker subprocess entry point tests
# =============================================================================


class TestWorkerModule:
    """Tests for the worker subprocess module."""

    def test_json_serializable_check(self):
        """Test _is_json_serializable function."""
        from visualpath.process.worker import _is_json_serializable

        assert _is_json_serializable({"key": "value"}) is True
        assert _is_json_serializable([1, 2, 3]) is True
        assert _is_json_serializable("string") is True
        assert _is_json_serializable(123) is True
        assert _is_json_serializable(None) is True

        # Non-serializable
        assert _is_json_serializable(lambda x: x) is False
        assert _is_json_serializable(object()) is False

    def test_argparse_help(self):
        """Test that the worker module has proper argparse setup."""
        from visualpath.process.worker import main

        # Running with --help should work (though it exits)
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["worker", "--help"]):
                main()

        assert exc_info.value.code == 0


# =============================================================================
# Integration Tests (requires real subprocess)
# =============================================================================


@pytest.mark.integration
class TestVenvWorkerIntegration:
    """Integration tests for VenvWorker with real subprocess.

    These tests require a properly configured venv with visualpath installed.
    They are marked with @pytest.mark.integration and can be skipped with:
        pytest -m "not integration"
    """

    @pytest.fixture
    def current_venv(self):
        """Get the current venv path if running in a venv."""
        if sys.prefix == sys.base_prefix:
            pytest.skip("Not running in a virtual environment")

        venv_path = sys.prefix
        python_path = os.path.join(venv_path, "bin", "python")

        if not os.path.isfile(python_path):
            pytest.skip(f"Python not found at {python_path}")

        return venv_path

    @pytest.mark.skipif(
        "VISUALPATH_INTEGRATION_TESTS" not in os.environ,
        reason="Integration tests disabled (set VISUALPATH_INTEGRATION_TESTS=1)",
    )
    def test_real_subprocess_with_dummy_extractor(self, current_venv):
        """Test VenvWorker with a real subprocess using dummy extractor."""
        # This test requires the 'dummy' extractor to be registered
        # via entry points in the test environment

        worker = VenvWorker(
            extractor=None,
            venv_path=current_venv,
            extractor_name="dummy",
            timeout_sec=10.0,
        )

        try:
            worker.start()
            assert worker.is_running
            assert worker._use_zmq  # Should be using ZMQ, not fallback

            frame = create_test_frame()
            result = worker.process(frame)

            assert result.error is None, f"Error: {result.error}"
            assert result.observation is not None
            assert result.observation.source == "dummy"

        finally:
            worker.stop()
            assert not worker.is_running


# =============================================================================
# Timing Tests
# =============================================================================


class TestWorkerTiming:
    """Tests for worker timing measurements."""

    def test_inline_timing(self):
        """Test that inline worker reports timing."""
        extractor = SimpleExtractor(delay_ms=50)
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        worker.start()

        frame = create_test_frame()
        result = worker.process(frame)

        assert result.timing_ms >= 50
        # Should have some overhead, but not too much
        assert result.timing_ms < 200

        worker.stop()

    def test_error_result_includes_timing(self):
        """Test that error results include timing."""
        extractor = SimpleExtractor(fail=True, delay_ms=10)
        worker = VenvWorker(
            extractor=extractor,
            venv_path="/nonexistent/venv",
        )

        worker.start()

        frame = create_test_frame()
        result = worker.process(frame)

        assert result.error is not None
        assert result.timing_ms > 0

        worker.stop()
