"""Tests for SimpleBackend execute() interface.

Tests verify:
- SimpleBackend.execute() with FlowGraph
- PipelineResult structure
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import BaseExtractor, Observation, BaseFusion, FusionResult
from visualpath.backends.base import ExecutionBackend, PipelineResult


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
    """Extractor that counts calls."""

    def __init__(self, name: str, value: float = 0.5):
        self._name = name
        self._value = value
        self._extract_count = 0
        self._initialized = False
        self._cleaned_up = False

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
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


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
# SimpleBackend execute() Tests
# =============================================================================


class TestSimpleBackendExecute:
    """Tests for SimpleBackend.execute() with FlowGraph."""

    def test_execute_single_extractor(self):
        """Test execute() with a single extractor."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingExtractor("test", value=0.3)
        graph = FlowGraph.from_pipeline([ext])
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert isinstance(result, PipelineResult)
        assert result.frame_count == 5
        assert ext._extract_count == 5
        assert len(result.triggers) == 0  # No fusion

    def test_execute_with_fusion(self):
        """Test execute() with fusion that triggers."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingExtractor("test", value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        graph = FlowGraph.from_pipeline([ext], fusion=fusion)
        frames = make_frames(5)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 5
        assert ext._extract_count == 5
        assert len(result.triggers) == 5  # All frames trigger

    def test_execute_no_trigger(self):
        """Test execute() when fusion doesn't fire."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingExtractor("test", value=0.3)
        fusion = ThresholdFusion(threshold=0.5)
        graph = FlowGraph.from_pipeline([ext], fusion=fusion)
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert result.frame_count == 3
        assert len(result.triggers) == 0

    def test_execute_multiple_extractors(self):
        """Test execute() with multiple extractors."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext1 = CountingExtractor("ext1", value=0.3)
        ext2 = CountingExtractor("ext2", value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        graph = FlowGraph.from_pipeline([ext1, ext2], fusion=fusion)
        frames = make_frames(3)

        result = backend.execute(iter(frames), graph)

        assert ext1._extract_count == 3
        assert ext2._extract_count == 3
        assert result.frame_count == 3

    def test_execute_empty_frames(self):
        """Test execute() with empty frame iterator."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingExtractor("test")
        graph = FlowGraph.from_pipeline([ext])

        result = backend.execute(iter([]), graph)

        assert result.frame_count == 0
        assert result.triggers == []

    def test_execute_with_flowgraph_builder(self):
        """Test execute() with a FlowGraphBuilder-built graph."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow import FlowGraphBuilder

        backend = SimpleBackend()
        ext = CountingExtractor("ext1", value=0.7)
        fusion = ThresholdFusion(threshold=0.5)

        graph = (FlowGraphBuilder()
            .source("frames")
            .sample(every_nth=2)
            .path("main", extractors=[ext], fusion=fusion)
            .build())

        triggered = []
        graph.on_trigger(lambda d: triggered.append(d))

        frames = make_frames(6)
        result = backend.execute(iter(frames), graph)

        # Only every 2nd frame is processed
        assert ext._extract_count == 3
        assert len(triggered) == 3

    def test_execute_with_callback(self):
        """Test execute() with on_trigger callback registered on graph."""
        from visualpath.backends.simple import SimpleBackend
        from visualpath.flow.graph import FlowGraph

        backend = SimpleBackend()
        ext = CountingExtractor("test", value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        graph = FlowGraph.from_pipeline([ext], fusion=fusion)
        frames = make_frames(3)

        callback_data = []
        graph.on_trigger(lambda d: callback_data.append(d))

        result = backend.execute(iter(frames), graph)

        assert len(callback_data) == 3
        assert len(result.triggers) == 3


# =============================================================================
# ExecutionBackend ABC Tests
# =============================================================================


class TestExecutionBackend:
    """Tests for ExecutionBackend abstract base class."""

    def test_is_abstract(self):
        """Test that ExecutionBackend cannot be instantiated."""
        with pytest.raises(TypeError):
            ExecutionBackend()

    def test_simple_backend_implements_interface(self):
        """Test SimpleBackend implements ExecutionBackend."""
        from visualpath.backends.simple import SimpleBackend

        backend = SimpleBackend()
        assert isinstance(backend, ExecutionBackend)
        assert hasattr(backend, "execute")
        assert hasattr(backend, "name")


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_default_values(self):
        result = PipelineResult()
        assert result.triggers == []
        assert result.frame_count == 0
        assert result.stats == {}

    def test_with_values(self):
        result = PipelineResult(triggers=["t1"], frame_count=10, stats={"key": "val"})
        assert result.triggers == ["t1"]
        assert result.frame_count == 10
        assert result.stats == {"key": "val"}
