"""Tests for visualpath core module."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import (
    Module,
    Observation,
    IsolationLevel,
    IsolationConfig,
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

    def __init__(self, extract_value: float = 0.5):
        self._extract_value = extract_value
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return "simple"

    def process(self, frame, deps=None) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._extract_value},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


class HeavyModule(Module):
    """Module that recommends VENV isolation."""

    @property
    def name(self) -> str:
        return "heavy"

    @property
    def recommended_isolation(self) -> IsolationLevel:
        return IsolationLevel.VENV

    def process(self, frame, deps=None) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
        )


class TriggerModule(Module):
    """Trigger module for testing."""

    depends = ["simple"]

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._gate_open = False
        self._cooldown = False

    @property
    def name(self) -> str:
        return "trigger"

    def process(self, frame, deps=None) -> Observation:
        obs = deps.get("simple") if deps else None
        value = obs.signals.get("value", 0) if obs else 0
        if value > self._threshold and self._gate_open:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "should_trigger": True,
                    "trigger_score": value,
                    "trigger_reason": "threshold_exceeded",
                },
            )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def reset(self) -> None:
        self._gate_open = False
        self._cooldown = False


# =============================================================================
# Observation Tests
# =============================================================================


class TestObservation:
    """Tests for Observation dataclass."""

    def test_basic_observation(self):
        """Test creating a basic observation."""
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
        )
        assert obs.source == "test"
        assert obs.frame_id == 1
        assert obs.t_ns == 1000000
        assert obs.signals == {}
        assert obs.data is None
        assert obs.metadata == {}

    def test_observation_with_signals(self):
        """Test observation with signals."""
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"confidence": 0.9, "score": 0.8},
        )
        assert obs.signals["confidence"] == 0.9
        assert obs.signals["score"] == 0.8

    def test_observation_with_data(self):
        """Test observation with generic data."""
        data = [{"id": 1, "bbox": (0, 0, 100, 100)}]
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            data=data,
        )
        assert obs.data == data

    def test_observation_with_timing(self):
        """Test observation with timing info."""
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            timing={"detect_ms": 10.5, "process_ms": 5.2},
        )
        assert obs.timing["detect_ms"] == 10.5

    def test_observation_trigger_properties(self):
        """Test observation trigger helper properties."""
        obs = Observation(
            source="trigger_module",
            frame_id=1,
            t_ns=1000000,
            signals={
                "should_trigger": True,
                "trigger_score": 0.85,
                "trigger_reason": "expression_spike",
            },
        )
        assert obs.should_trigger is True
        assert obs.trigger_score == 0.85
        assert obs.trigger_reason == "expression_spike"

    def test_observation_no_trigger(self):
        """Test observation without trigger signals."""
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
        )
        assert obs.should_trigger is False
        assert obs.trigger_score == 0.0
        assert obs.trigger_reason == ""


# =============================================================================
# Module Tests
# =============================================================================


class TestModule:
    """Tests for Module abstract base class."""

    def test_simple_module(self):
        """Test simple module implementation."""
        module = SimpleModule(extract_value=0.7)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        obs = module.process(frame)

        assert obs is not None
        assert obs.source == "simple"
        assert obs.frame_id == 1
        assert obs.signals["value"] == 0.7

    def test_module_lifecycle(self):
        """Test module initialize/cleanup lifecycle."""
        module = SimpleModule()

        assert not module._initialized
        assert not module._cleaned_up

        module.initialize()
        assert module._initialized

        module.cleanup()
        assert module._cleaned_up

    def test_module_context_manager(self):
        """Test module as context manager."""
        module = SimpleModule()

        with module as m:
            assert m._initialized
            assert not m._cleaned_up

        assert m._cleaned_up

    def test_module_extract_alias(self):
        """Test extract() is an alias for process()."""
        module = SimpleModule(extract_value=0.5)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        # extract() should work just like process()
        obs = module.extract(frame)

        assert obs is not None
        assert obs.source == "simple"
        assert obs.signals["value"] == 0.5


# =============================================================================
# Trigger Module Tests
# =============================================================================


class TestTriggerModule:
    """Tests for trigger module functionality."""

    def test_trigger_no_trigger(self):
        """Test trigger module with gate closed (no trigger)."""
        module = TriggerModule(threshold=0.5)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))
        deps = {
            "simple": Observation(
                source="simple",
                frame_id=1,
                t_ns=1000000,
                signals={"value": 0.7},
            )
        }

        result = module.process(frame, deps)

        assert result.should_trigger is False

    def test_trigger_fires(self):
        """Test trigger module with gate open and threshold exceeded."""
        module = TriggerModule(threshold=0.5)
        module._gate_open = True
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        deps = {
            "simple": Observation(
                source="simple",
                frame_id=1,
                t_ns=1000000,
                signals={"value": 0.7},
            )
        }

        result = module.process(frame, deps)

        assert result.should_trigger is True
        assert result.trigger_score == 0.7
        assert result.trigger_reason == "threshold_exceeded"

    def test_trigger_reset(self):
        """Test trigger module reset."""
        module = TriggerModule()
        module._gate_open = True
        module._cooldown = True

        module.reset()

        assert not module._gate_open
        assert not module._cooldown


# =============================================================================
# IsolationLevel Tests
# =============================================================================


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels_ordered(self):
        """Test isolation levels are ordered by isolation degree."""
        assert IsolationLevel.INLINE < IsolationLevel.THREAD
        assert IsolationLevel.THREAD < IsolationLevel.PROCESS
        assert IsolationLevel.PROCESS < IsolationLevel.VENV
        assert IsolationLevel.VENV < IsolationLevel.CONTAINER

    def test_from_string(self):
        """Test parsing isolation level from string."""
        assert IsolationLevel.from_string("inline") == IsolationLevel.INLINE
        assert IsolationLevel.from_string("THREAD") == IsolationLevel.THREAD
        assert IsolationLevel.from_string("Process") == IsolationLevel.PROCESS
        assert IsolationLevel.from_string("venv") == IsolationLevel.VENV
        assert IsolationLevel.from_string("container") == IsolationLevel.CONTAINER

    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            IsolationLevel.from_string("invalid")


# =============================================================================
# IsolationConfig Tests
# =============================================================================


class TestIsolationConfig:
    """Tests for IsolationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IsolationConfig()

        assert config.default_level == IsolationLevel.INLINE
        assert config.overrides == {}

    def test_get_level_default(self):
        """Test get_level returns default when no override."""
        config = IsolationConfig(default_level=IsolationLevel.PROCESS)

        level = config.get_level("any_extractor")

        assert level == IsolationLevel.PROCESS

    def test_get_level_override(self):
        """Test get_level respects override."""
        config = IsolationConfig(
            default_level=IsolationLevel.PROCESS,
            overrides={"face": IsolationLevel.VENV},
        )

        assert config.get_level("face") == IsolationLevel.VENV
        assert config.get_level("pose") == IsolationLevel.PROCESS

    def test_get_level_respects_recommended(self):
        """Test get_level respects recommended when higher than default."""
        config = IsolationConfig(default_level=IsolationLevel.INLINE)

        # Recommended higher than default
        level = config.get_level("heavy", recommended=IsolationLevel.VENV)
        assert level == IsolationLevel.VENV

        # Recommended lower than default
        config2 = IsolationConfig(default_level=IsolationLevel.PROCESS)
        level2 = config2.get_level("simple", recommended=IsolationLevel.INLINE)
        assert level2 == IsolationLevel.PROCESS

    def test_get_level_override_beats_recommended(self):
        """Test override takes precedence over recommended."""
        config = IsolationConfig(
            default_level=IsolationLevel.INLINE,
            overrides={"heavy": IsolationLevel.INLINE},  # Force inline for debugging
        )

        level = config.get_level("heavy", recommended=IsolationLevel.VENV)
        assert level == IsolationLevel.INLINE

    def test_venv_paths(self):
        """Test venv path configuration."""
        config = IsolationConfig(
            venv_paths={
                "face": "/opt/venvs/face",
                "pose": "/opt/venvs/pose",
            }
        )

        assert config.get_venv_path("face") == "/opt/venvs/face"
        assert config.get_venv_path("pose") == "/opt/venvs/pose"
        assert config.get_venv_path("unknown") is None

    def test_container_images(self):
        """Test container image configuration."""
        config = IsolationConfig(
            container_images={
                "face": "myrepo/face:latest",
            }
        )

        assert config.get_container_image("face") == "myrepo/face:latest"
        assert config.get_container_image("unknown") is None
