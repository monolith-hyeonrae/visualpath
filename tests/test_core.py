"""Tests for visualpath core module."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import (
    BaseExtractor,
    Observation,
    BaseFusion,
    FusionResult,
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


class SimpleExtractor(BaseExtractor):
    """Simple extractor for testing."""

    def __init__(self, extract_value: float = 0.5):
        self._extract_value = extract_value
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return "simple"

    def extract(self, frame) -> Optional[Observation]:
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


class HeavyExtractor(BaseExtractor):
    """Extractor that recommends VENV isolation."""

    @property
    def name(self) -> str:
        return "heavy"

    @property
    def recommended_isolation(self) -> IsolationLevel:
        return IsolationLevel.VENV

    def extract(self, frame) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
        )


class SimpleFusion(BaseFusion):
    """Simple fusion for testing."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._gate_open = False
        self._cooldown = False

    def update(self, observation: Observation) -> FusionResult:
        value = observation.signals.get("value", 0)
        if value > self._threshold and self._gate_open:
            return FusionResult(
                should_trigger=True,
                score=value,
                reason="threshold_exceeded",
            )
        return FusionResult(should_trigger=False)

    def reset(self) -> None:
        self._gate_open = False
        self._cooldown = False

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown


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


# =============================================================================
# BaseExtractor Tests
# =============================================================================


class TestBaseExtractor:
    """Tests for BaseExtractor abstract base class."""

    def test_simple_extractor(self):
        """Test simple extractor implementation."""
        extractor = SimpleExtractor(extract_value=0.7)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "simple"
        assert obs.frame_id == 1
        assert obs.signals["value"] == 0.7

    def test_extractor_lifecycle(self):
        """Test extractor initialize/cleanup lifecycle."""
        extractor = SimpleExtractor()

        assert not extractor._initialized
        assert not extractor._cleaned_up

        extractor.initialize()
        assert extractor._initialized

        extractor.cleanup()
        assert extractor._cleaned_up

    def test_extractor_context_manager(self):
        """Test extractor as context manager."""
        extractor = SimpleExtractor()

        with extractor as e:
            assert e._initialized
            assert not e._cleaned_up

        assert e._cleaned_up

    def test_default_recommended_isolation(self):
        """Test default recommended isolation is INLINE."""
        extractor = SimpleExtractor()
        assert extractor.recommended_isolation == IsolationLevel.INLINE

    def test_custom_recommended_isolation(self):
        """Test custom recommended isolation."""
        extractor = HeavyExtractor()
        assert extractor.recommended_isolation == IsolationLevel.VENV


# =============================================================================
# BaseFusion Tests
# =============================================================================


class TestBaseFusion:
    """Tests for BaseFusion abstract base class."""

    def test_simple_fusion_no_trigger(self):
        """Test fusion with gate closed (no trigger)."""
        fusion = SimpleFusion(threshold=0.5)
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"value": 0.7},
        )

        result = fusion.update(obs)

        assert not result.should_trigger
        assert result.trigger is None

    def test_simple_fusion_trigger(self):
        """Test fusion with gate open and threshold exceeded."""
        fusion = SimpleFusion(threshold=0.5)
        fusion._gate_open = True

        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"value": 0.7},
        )

        result = fusion.update(obs)

        assert result.should_trigger
        assert result.score == 0.7
        assert result.reason == "threshold_exceeded"

    def test_fusion_reset(self):
        """Test fusion reset."""
        fusion = SimpleFusion()
        fusion._gate_open = True
        fusion._cooldown = True

        fusion.reset()

        assert not fusion.is_gate_open
        assert not fusion.in_cooldown


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


# =============================================================================
# FusionResult Tests
# =============================================================================


class TestFusionResult:
    """Tests for FusionResult dataclass."""

    def test_basic_result(self):
        """Test basic fusion result."""
        result = FusionResult(should_trigger=False)

        assert not result.should_trigger
        assert result.trigger is None
        assert result.score == 0.0
        assert result.reason == ""

    def test_trigger_result(self):
        """Test fusion result with trigger."""
        result = FusionResult(
            should_trigger=True,
            score=0.85,
            reason="expression_spike",
            observations_used=3,
            metadata={"face_id": 1},
        )

        assert result.should_trigger
        assert result.score == 0.85
        assert result.reason == "expression_spike"
        assert result.observations_used == 3
        assert result.metadata["face_id"] == 1
