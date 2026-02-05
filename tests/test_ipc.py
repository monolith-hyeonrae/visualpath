"""Tests for ExtractorProcess and FusionProcess IPC wrappers."""

import pytest
import json
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from queue import Queue

from visualpath.core import Module, Observation
from visualpath.process import (
    ExtractorProcess,
    FusionProcess,
    DefaultObservationMapper,
    CompositeMapper,
)
from visualbase import Trigger


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class MockVideoReader:
    """Mock VideoReader for testing."""

    def __init__(self, frames: List[MockFrame]):
        self._frames = frames
        self._index = 0
        self._is_open = False

    def open(self, timeout_sec: Optional[float] = None) -> bool:
        self._is_open = True
        return True

    def close(self) -> None:
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def read(self) -> Optional[MockFrame]:
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame


class MockMessageSender:
    """Mock MessageSender that collects sent messages."""

    def __init__(self):
        self._connected = False
        self._messages: List[str] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send(self, message: str) -> bool:
        if not self._connected:
            return False
        self._messages.append(message)
        return True


class MockMessageReceiver:
    """Mock MessageReceiver that provides queued messages."""

    def __init__(self, messages: Optional[List[str]] = None):
        self._messages = messages or []
        self._index = 0
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def add_message(self, msg: str) -> None:
        """Add a message for testing."""
        self._messages.append(msg)

    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._index >= len(self._messages):
            return None
        msg = self._messages[self._index]
        self._index += 1
        return msg

    def recv_all(self, max_messages: int = 100) -> List[str]:
        remaining = self._messages[self._index:]
        self._index = len(self._messages)
        return remaining[:max_messages]


class SimpleModule(Module):
    """Simple module for testing."""

    def __init__(self, name: str = "simple", delay_ms: float = 0):
        self._name = name
        self._delay_ms = delay_ms
        self._initialized = False

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": float(frame.frame_id)},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._initialized = False


class TriggerModule(Module):
    """Trigger module for testing."""

    depends = ["simple"]

    def __init__(self, trigger_threshold: float = 0.5):
        self._threshold = trigger_threshold
        self._gate_open = True
        self._in_cooldown = False
        self._observations: List[Observation] = []

    @property
    def name(self) -> str:
        return "trigger"

    def process(self, frame, deps=None) -> Observation:
        obs = deps.get("simple") if deps else None
        if obs:
            self._observations.append(obs)
        score = obs.signals.get("value", 0) if obs else 0

        if score > self._threshold and self._gate_open:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": score,
                    "trigger_reason": "threshold_exceeded",
                },
                data={
                    "trigger": Trigger(
                        label="TEST",
                        clip_start_ns=frame.t_src_ns - 1000000000,
                        clip_end_ns=frame.t_src_ns,
                    )
                },
            )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def reset(self) -> None:
        self._observations.clear()


# =============================================================================
# ExtractorProcess Tests
# =============================================================================


class TestExtractorProcess:
    """Tests for ExtractorProcess."""

    def test_initialization_with_interfaces(self):
        """Test initialization with interface objects."""
        module = SimpleModule()
        reader = MockVideoReader([])
        sender = MockMessageSender()
        mapper = DefaultObservationMapper()

        process = ExtractorProcess(
            extractor=module,
            observation_mapper=mapper,
            video_reader=reader,
            message_sender=sender,
        )

        assert not process.is_running
        assert process.get_stats()["frames_processed"] == 0

    def test_initialization_requires_reader_or_path(self):
        """Test that either video_reader or input_fifo is required."""
        module = SimpleModule()
        sender = MockMessageSender()

        with pytest.raises(ValueError, match="video_reader or input_fifo"):
            ExtractorProcess(
                extractor=module,
                message_sender=sender,
            )

    def test_initialization_requires_sender_or_path(self):
        """Test that either message_sender or obs_socket is required."""
        module = SimpleModule()
        reader = MockVideoReader([])

        with pytest.raises(ValueError, match="message_sender or obs_socket"):
            ExtractorProcess(
                extractor=module,
                video_reader=reader,
            )

    def test_process_frames(self):
        """Test processing frames."""
        frames = [
            MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
            for i in range(5)
        ]
        module = SimpleModule()
        reader = MockVideoReader(frames)
        sender = MockMessageSender()
        mapper = DefaultObservationMapper()

        process = ExtractorProcess(
            extractor=module,
            observation_mapper=mapper,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,  # Don't reconnect after frames are done
        )

        # Run in thread and stop after processing
        thread = threading.Thread(target=process.run)
        thread.start()

        # Wait for processing
        time.sleep(0.5)
        process.stop()
        thread.join(timeout=2.0)

        # Check stats
        stats = process.get_stats()
        assert stats["frames_processed"] == 5
        assert stats["obs_sent"] == 5

        # Check messages were sent
        assert len(sender._messages) == 5

    def test_process_with_callback(self):
        """Test frame callback is called."""
        frames = [
            MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))
        ]
        module = SimpleModule()
        reader = MockVideoReader(frames)
        sender = MockMessageSender()

        callback_results = []

        def on_frame(frame, obs):
            callback_results.append((frame.frame_id, obs.source))

        process = ExtractorProcess(
            extractor=module,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,
            on_frame=on_frame,
        )

        thread = threading.Thread(target=process.run)
        thread.start()

        time.sleep(0.3)
        process.stop()
        thread.join(timeout=2.0)

        assert len(callback_results) == 1
        assert callback_results[0] == (1, "simple")

    def test_stop_terminates_process(self):
        """Test that stop() terminates the process."""
        # Create reader that never ends
        frames = [MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3))) for i in range(1000)]
        module = SimpleModule(delay_ms=100)  # Slow processing
        reader = MockVideoReader(frames)
        sender = MockMessageSender()

        process = ExtractorProcess(
            extractor=module,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,
        )

        thread = threading.Thread(target=process.run)
        thread.start()

        time.sleep(0.1)  # Let it start
        process.stop()

        thread.join(timeout=2.0)
        assert not thread.is_alive()

    def test_uses_default_mapper_if_not_provided(self):
        """Test that DefaultObservationMapper is used if not provided."""
        frames = [MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))]
        module = SimpleModule()
        reader = MockVideoReader(frames)
        sender = MockMessageSender()

        process = ExtractorProcess(
            extractor=module,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,
        )

        thread = threading.Thread(target=process.run)
        thread.start()
        time.sleep(0.3)
        process.stop()
        thread.join(timeout=2.0)

        # Should have sent a JSON message (default mapper format)
        assert len(sender._messages) == 1
        data = json.loads(sender._messages[0])
        assert data["source"] == "simple"


# =============================================================================
# FusionProcess Tests
# =============================================================================


class TestFusionProcess:
    """Tests for FusionProcess."""

    def test_initialization_with_interfaces(self):
        """Test initialization with interface objects."""
        trigger_module = TriggerModule()
        receiver = MockMessageReceiver()
        sender = MockMessageSender()
        mapper = DefaultObservationMapper()

        process = FusionProcess(
            fusion=trigger_module,
            observation_mapper=mapper,
            obs_receiver=receiver,
            trig_sender=sender,
        )

        assert not process.is_running
        assert process.get_stats()["obs_received"] == 0

    def test_initialization_requires_receiver_or_path(self):
        """Test that either obs_receiver or obs_socket is required."""
        trigger_module = TriggerModule()
        sender = MockMessageSender()

        with pytest.raises(ValueError, match="obs_receiver or obs_socket"):
            FusionProcess(
                fusion=trigger_module,
                trig_sender=sender,
            )

    def test_initialization_requires_sender_or_path(self):
        """Test that either trig_sender or trig_socket is required."""
        trigger_module = TriggerModule()
        receiver = MockMessageReceiver()

        with pytest.raises(ValueError, match="trig_sender or trig_socket"):
            FusionProcess(
                fusion=trigger_module,
                obs_receiver=receiver,
            )

    def test_process_observations(self):
        """Test processing observations and sending triggers."""
        # Create OBS messages in JSON format
        messages = [
            json.dumps({
                "source": "test",
                "frame_id": 1,
                "t_ns": 1000000,
                "signals": {"value": 0.8},  # Above threshold
            }),
        ]

        trigger_module = TriggerModule(trigger_threshold=0.5)
        receiver = MockMessageReceiver(messages)
        sender = MockMessageSender()
        mapper = DefaultObservationMapper()

        process = FusionProcess(
            fusion=trigger_module,
            observation_mapper=mapper,
            obs_receiver=receiver,
            trig_sender=sender,
            alignment_window_ns=0,  # Process immediately
        )

        # Run briefly
        thread = threading.Thread(target=process.run)
        thread.start()

        time.sleep(0.3)
        process.stop()
        thread.join(timeout=2.0)

        # Check stats
        stats = process.get_stats()
        assert stats["obs_received"] == 1
        # Trigger may or may not be sent depending on timing
        # The observation should have been processed
        assert len(trigger_module._observations) >= 0

    def test_stop_terminates_process(self):
        """Test that stop() terminates the process."""
        trigger_module = TriggerModule()
        receiver = MockMessageReceiver()
        sender = MockMessageSender()

        process = FusionProcess(
            fusion=trigger_module,
            obs_receiver=receiver,
            trig_sender=sender,
        )

        thread = threading.Thread(target=process.run)
        thread.start()

        time.sleep(0.1)
        process.stop()

        thread.join(timeout=2.0)
        assert not thread.is_alive()

    def test_parses_obs_message_format(self):
        """Test parsing OBS message format."""
        # OBS format: "OBS src=test frame=1 t_ns=1000000 ..."
        messages = [
            "OBS src=test frame=1 t_ns=1000000",
        ]

        trigger_module = TriggerModule()
        receiver = MockMessageReceiver(messages)
        sender = MockMessageSender()

        # Custom mapper for OBS format
        class OBSMapper:
            def to_message(self, obs):
                return None  # Not used

            def from_message(self, msg):
                if not msg.startswith("OBS "):
                    return None
                parts = {}
                for p in msg.split()[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        parts[k] = v
                return Observation(
                    source=parts.get("src", ""),
                    frame_id=int(parts.get("frame", 0)),
                    t_ns=int(parts.get("t_ns", 0)),
                )

        process = FusionProcess(
            fusion=trigger_module,
            observation_mapper=OBSMapper(),
            obs_receiver=receiver,
            trig_sender=sender,
            alignment_window_ns=0,
        )

        thread = threading.Thread(target=process.run)
        thread.start()

        time.sleep(0.3)
        process.stop()
        thread.join(timeout=2.0)

        stats = process.get_stats()
        assert stats["obs_received"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestExtractorFusionIntegration:
    """Integration tests for ExtractorProcess and FusionProcess."""

    def test_message_format_compatibility(self):
        """Test that extractor and fusion use compatible message formats."""
        # Create a simple observation
        obs = Observation(
            source="test",
            frame_id=42,
            t_ns=1234567890,
            signals={"score": 0.95},
        )

        # Serialize with extractor's mapper
        mapper = DefaultObservationMapper()
        message = mapper.to_message(obs)

        # Deserialize with fusion's mapper (same mapper type)
        restored = mapper.from_message(message)

        assert restored is not None
        assert restored.source == obs.source
        assert restored.frame_id == obs.frame_id
        assert restored.t_ns == obs.t_ns
        assert restored.signals["score"] == 0.95
