"""Tests for Path and PathOrchestrator."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from visualpath.core import (
    BaseExtractor,
    Observation,
    BaseFusion,
    FusionResult,
    Path,
    PathConfig,
    PathOrchestrator,
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


class CountingExtractor(BaseExtractor):
    """Extractor that counts calls for testing."""

    def __init__(self, name: str, return_value: float = 0.5):
        self._name = name
        self._return_value = return_value
        self._extract_count = 0
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame) -> Optional[Observation]:
        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._return_value, "call_count": self._extract_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


class ThresholdFusion(BaseFusion):
    """Simple fusion for testing."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._gate_open = True
        self._cooldown = False
        self._update_count = 0

    def update(self, observation: Observation) -> FusionResult:
        self._update_count += 1
        value = observation.signals.get("value", 0)
        if value > self._threshold:
            return FusionResult(
                should_trigger=True,
                score=value,
                reason="threshold_exceeded",
                observations_used=1,
            )
        return FusionResult(should_trigger=False)

    def reset(self) -> None:
        self._update_count = 0

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown


# =============================================================================
# PathConfig Tests
# =============================================================================


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic path config."""
        from visualpath.core.isolation import IsolationLevel

        config = PathConfig(
            name="human",
            extractors=["face", "pose"],
            fusion="highlight",
        )

        assert config.name == "human"
        assert config.extractors == ["face", "pose"]
        assert config.fusion == "highlight"
        assert config.default_isolation == IsolationLevel.INLINE

    def test_config_with_isolation(self):
        """Test config with isolation level."""
        from visualpath.core.isolation import IsolationLevel

        config = PathConfig(
            name="scene",
            extractors=["object", "ocr"],
            default_isolation=IsolationLevel.VENV,
        )

        assert config.default_isolation == IsolationLevel.VENV

    def test_config_with_extractor_overrides(self):
        """Test config with per-extractor overrides."""
        config = PathConfig(
            name="human",
            extractors=["face", "pose"],
            extractor_config={
                "face": {"confidence_threshold": 0.8},
                "pose": {"max_persons": 5},
            },
        )

        assert config.extractor_config["face"]["confidence_threshold"] == 0.8
        assert config.extractor_config["pose"]["max_persons"] == 5


# =============================================================================
# Path Tests
# =============================================================================


class TestPath:
    """Tests for Path class."""

    def test_basic_path_creation(self):
        """Test creating a basic path."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(
            name="test",
            extractors=[e1, e2],
        )

        assert path.name == "test"
        assert len(path.extractors) == 2
        assert path.fusion is None

    def test_path_with_fusion(self):
        """Test path with fusion module."""
        e1 = CountingExtractor("ext1")
        fusion = ThresholdFusion()

        path = Path(
            name="test",
            extractors=[e1],
            fusion=fusion,
        )

        assert path.fusion is fusion

    def test_path_lifecycle(self):
        """Test path initialize/cleanup lifecycle."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(name="test", extractors=[e1, e2])

        assert not e1._initialized
        assert not e2._initialized

        path.initialize()

        assert e1._initialized
        assert e2._initialized

        path.cleanup()

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_path_context_manager(self):
        """Test path as context manager."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")

        path = Path(name="test", extractors=[e1, e2])

        with path as p:
            assert e1._initialized
            assert e2._initialized
            assert p is path

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_extract_all_sequential(self):
        """Test extracting from all extractors sequentially."""
        e1 = CountingExtractor("ext1", return_value=0.3)
        e2 = CountingExtractor("ext2", return_value=0.7)

        path = Path(name="test", extractors=[e1, e2], parallel=False)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            observations = path.extract_all(frame)

        assert len(observations) == 2
        assert observations[0].source == "ext1"
        assert observations[0].signals["value"] == 0.3
        assert observations[1].source == "ext2"
        assert observations[1].signals["value"] == 0.7

    def test_extract_all_parallel(self):
        """Test extracting from all extractors in parallel."""
        e1 = CountingExtractor("ext1", return_value=0.3)
        e2 = CountingExtractor("ext2", return_value=0.7)
        e3 = CountingExtractor("ext3", return_value=0.5)

        path = Path(name="test", extractors=[e1, e2, e3], parallel=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            observations = path.extract_all(frame)

        assert len(observations) == 3
        sources = {obs.source for obs in observations}
        assert sources == {"ext1", "ext2", "ext3"}

    def test_extract_all_not_initialized_raises(self):
        """Test that extract_all raises when not initialized."""
        e1 = CountingExtractor("ext1")
        path = Path(name="test", extractors=[e1])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            path.extract_all(frame)

    def test_process_with_fusion(self):
        """Test processing through extractors and fusion."""
        e1 = CountingExtractor("ext1", return_value=0.7)  # Above threshold
        e2 = CountingExtractor("ext2", return_value=0.3)  # Below threshold
        fusion = ThresholdFusion(threshold=0.5)

        path = Path(name="test", extractors=[e1, e2], fusion=fusion, parallel=False)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            results = path.process(frame)

        assert len(results) == 2
        # Results depend on order of observations
        trigger_count = sum(1 for r in results if r.should_trigger)
        no_trigger_count = sum(1 for r in results if not r.should_trigger)
        assert trigger_count == 1  # ext1 exceeds threshold
        assert no_trigger_count == 1  # ext2 below threshold

    def test_process_no_fusion(self):
        """Test processing without fusion returns empty."""
        e1 = CountingExtractor("ext1")
        path = Path(name="test", extractors=[e1])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with path:
            results = path.process(frame)

        assert results == []


# =============================================================================
# PathOrchestrator Tests
# =============================================================================


class TestPathOrchestrator:
    """Tests for PathOrchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator."""
        path1 = Path(name="path1", extractors=[CountingExtractor("ext1")])
        path2 = Path(name="path2", extractors=[CountingExtractor("ext2")])

        orchestrator = PathOrchestrator([path1, path2])

        assert len(orchestrator.paths) == 2

    def test_orchestrator_lifecycle(self):
        """Test orchestrator initialize/cleanup lifecycle."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        path1 = Path(name="path1", extractors=[e1])
        path2 = Path(name="path2", extractors=[e2])

        orchestrator = PathOrchestrator([path1, path2])

        assert not e1._initialized
        assert not e2._initialized

        orchestrator.initialize()

        assert e1._initialized
        assert e2._initialized

        orchestrator.cleanup()

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_orchestrator_context_manager(self):
        """Test orchestrator as context manager."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        path1 = Path(name="path1", extractors=[e1])
        path2 = Path(name="path2", extractors=[e2])

        orchestrator = PathOrchestrator([path1, path2])

        with orchestrator as o:
            assert e1._initialized
            assert e2._initialized
            assert o is orchestrator

        assert e1._cleaned_up
        assert e2._cleaned_up

    def test_process_all_sequential(self):
        """Test processing all paths sequentially."""
        e1 = CountingExtractor("ext1", return_value=0.7)
        e2 = CountingExtractor("ext2", return_value=0.3)
        fusion1 = ThresholdFusion(threshold=0.5)
        fusion2 = ThresholdFusion(threshold=0.5)

        path1 = Path(name="path1", extractors=[e1], fusion=fusion1, parallel=False)
        path2 = Path(name="path2", extractors=[e2], fusion=fusion2, parallel=False)

        orchestrator = PathOrchestrator([path1, path2], parallel=False)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            results = orchestrator.process_all(frame)

        assert "path1" in results
        assert "path2" in results
        assert len(results["path1"]) == 1
        assert len(results["path2"]) == 1
        assert results["path1"][0].should_trigger  # 0.7 > 0.5
        assert not results["path2"][0].should_trigger  # 0.3 < 0.5

    def test_process_all_parallel(self):
        """Test processing all paths in parallel."""
        e1 = CountingExtractor("ext1", return_value=0.7)
        e2 = CountingExtractor("ext2", return_value=0.8)
        fusion1 = ThresholdFusion(threshold=0.5)
        fusion2 = ThresholdFusion(threshold=0.5)

        path1 = Path(name="path1", extractors=[e1], fusion=fusion1)
        path2 = Path(name="path2", extractors=[e2], fusion=fusion2)

        orchestrator = PathOrchestrator([path1, path2], parallel=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            results = orchestrator.process_all(frame)

        assert "path1" in results
        assert "path2" in results
        assert results["path1"][0].should_trigger
        assert results["path2"][0].should_trigger

    def test_process_all_not_initialized_raises(self):
        """Test that process_all raises when not initialized."""
        path = Path(name="path1", extractors=[CountingExtractor("ext1")])
        orchestrator = PathOrchestrator([path])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.process_all(frame)

    def test_extract_all(self):
        """Test extracting observations from all paths."""
        e1 = CountingExtractor("ext1")
        e2 = CountingExtractor("ext2")
        e3 = CountingExtractor("ext3")

        path1 = Path(name="path1", extractors=[e1, e2])
        path2 = Path(name="path2", extractors=[e3])

        orchestrator = PathOrchestrator([path1, path2])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with orchestrator:
            observations = orchestrator.extract_all(frame)

        assert "path1" in observations
        assert "path2" in observations
        assert len(observations["path1"]) == 2
        assert len(observations["path2"]) == 1

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        e1 = CountingExtractor("ext1")
        path = Path(name="path1", extractors=[e1])
        orchestrator = PathOrchestrator([path])

        frames = [
            MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
            for i in range(5)
        ]

        with orchestrator:
            for frame in frames:
                orchestrator.extract_all(frame)

        assert e1._extract_count == 5
