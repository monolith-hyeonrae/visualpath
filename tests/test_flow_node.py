"""Tests for flow nodes."""

import pytest
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import (
    BaseExtractor,
    Observation,
    BaseFusion,
    FusionResult,
)
from visualpath.flow import (
    FlowData,
    FlowNode,
    SourceNode,
    PathNode,
    FilterNode,
    ObservationFilter,
    SignalThresholdFilter,
    SamplerNode,
    RateLimiterNode,
    TimestampSamplerNode,
    BranchNode,
    FanOutNode,
    MultiBranchNode,
    ConditionalFanOutNode,
    JoinNode,
    CascadeFusionNode,
    CollectorNode,
)
from visualpath.core.path import Path


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


def make_frame(frame_id: int = 1, t_ns: int = 1_000_000) -> MockFrame:
    """Create a mock frame."""
    return MockFrame(
        frame_id=frame_id,
        t_src_ns=t_ns,
        data=np.zeros((100, 100, 3), dtype=np.uint8),
    )


def make_flow_data(
    frame: Optional[MockFrame] = None,
    path_id: str = "default",
    timestamp_ns: int = 0,
    observations: Optional[List[Observation]] = None,
) -> FlowData:
    """Create a FlowData for testing."""
    return FlowData(
        frame=frame or make_frame(),
        observations=observations or [],
        results=[],
        metadata={},
        path_id=path_id,
        timestamp_ns=timestamp_ns,
    )


# =============================================================================
# FlowData Tests
# =============================================================================


class TestFlowData:
    """Tests for FlowData dataclass."""

    def test_basic_creation(self):
        """Test creating FlowData."""
        data = FlowData()

        assert data.frame is None
        assert data.observations == []
        assert data.results == []
        assert data.metadata == {}
        assert data.path_id == "default"
        assert data.timestamp_ns == 0

    def test_with_values(self):
        """Test creating FlowData with values."""
        frame = make_frame()
        data = FlowData(
            frame=frame,
            path_id="human",
            timestamp_ns=1000000,
            metadata={"key": "value"},
        )

        assert data.frame is frame
        assert data.path_id == "human"
        assert data.timestamp_ns == 1000000
        assert data.metadata["key"] == "value"

    def test_clone(self):
        """Test cloning FlowData."""
        data = make_flow_data(path_id="original")
        clone = data.clone()

        assert clone.path_id == "original"
        assert clone is not data
        assert clone.observations is not data.observations

    def test_clone_with_overrides(self):
        """Test cloning with overrides."""
        data = make_flow_data(path_id="original")
        clone = data.clone(path_id="modified", timestamp_ns=5000)

        assert clone.path_id == "modified"
        assert clone.timestamp_ns == 5000
        assert data.path_id == "original"

    def test_with_path(self):
        """Test with_path helper."""
        data = make_flow_data(path_id="original")
        new_data = data.with_path("new_path")

        assert new_data.path_id == "new_path"
        assert data.path_id == "original"

    def test_add_observation(self):
        """Test adding observations."""
        data = make_flow_data()
        obs = Observation(source="test", frame_id=1, t_ns=1000)
        new_data = data.add_observation(obs)

        assert len(new_data.observations) == 1
        assert len(data.observations) == 0

    def test_add_result(self):
        """Test adding results."""
        data = make_flow_data()
        result = FusionResult(should_trigger=True, score=0.9)
        new_data = data.add_result(result)

        assert len(new_data.results) == 1
        assert len(data.results) == 0


# =============================================================================
# SourceNode Tests
# =============================================================================


class TestSourceNode:
    """Tests for SourceNode."""

    def test_basic_creation(self):
        """Test creating SourceNode."""
        node = SourceNode("my_source")

        assert node.name == "my_source"

    def test_process_frame(self):
        """Test converting Frame to FlowData."""
        node = SourceNode("source", default_path_id="main")
        frame = make_frame(frame_id=5, t_ns=5_000_000)

        data = node.process_frame(frame)

        assert data.frame is frame
        assert data.path_id == "main"
        assert data.timestamp_ns == 5_000_000

    def test_process_passthrough(self):
        """Test process passes through FlowData."""
        node = SourceNode("source")
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1
        assert outputs[0] is data


# =============================================================================
# FilterNode Tests
# =============================================================================


class TestFilterNode:
    """Tests for FilterNode."""

    def test_filter_passes(self):
        """Test filter that passes data."""
        node = FilterNode("always_pass", condition=lambda d: True)
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1

    def test_filter_blocks(self):
        """Test filter that blocks data."""
        node = FilterNode("always_block", condition=lambda d: False)
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 0

    def test_filter_with_condition(self):
        """Test filter with custom condition."""
        node = FilterNode(
            "has_metadata",
            condition=lambda d: "key" in d.metadata,
        )

        data_with_key = make_flow_data()
        data_with_key = data_with_key.clone(metadata={"key": "value"})

        data_without_key = make_flow_data()

        assert len(node.process(data_with_key)) == 1
        assert len(node.process(data_without_key)) == 0


class TestObservationFilter:
    """Tests for ObservationFilter."""

    def test_passes_with_observations(self):
        """Test filter passes when observations exist."""
        node = ObservationFilter("obs_filter", min_count=1)
        obs = Observation(source="test", frame_id=1, t_ns=1000)
        data = make_flow_data(observations=[obs])

        outputs = node.process(data)

        assert len(outputs) == 1

    def test_blocks_without_observations(self):
        """Test filter blocks when no observations."""
        node = ObservationFilter("obs_filter", min_count=1)
        data = make_flow_data(observations=[])

        outputs = node.process(data)

        assert len(outputs) == 0

    def test_min_count(self):
        """Test minimum count requirement."""
        node = ObservationFilter("obs_filter", min_count=3)
        obs = Observation(source="test", frame_id=1, t_ns=1000)

        data_1 = make_flow_data(observations=[obs])
        data_3 = make_flow_data(observations=[obs, obs, obs])

        assert len(node.process(data_1)) == 0
        assert len(node.process(data_3)) == 1


class TestSignalThresholdFilter:
    """Tests for SignalThresholdFilter."""

    def test_gt_comparison(self):
        """Test greater than comparison."""
        node = SignalThresholdFilter("score_filter", "score", threshold=0.5, comparison="gt")
        obs_low = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.3})
        obs_high = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.8})

        assert len(node.process(make_flow_data(observations=[obs_low]))) == 0
        assert len(node.process(make_flow_data(observations=[obs_high]))) == 1

    def test_le_comparison(self):
        """Test less than or equal comparison."""
        node = SignalThresholdFilter("score_filter", "score", threshold=0.5, comparison="le")
        obs_low = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.3})
        obs_high = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.8})

        assert len(node.process(make_flow_data(observations=[obs_low]))) == 1
        assert len(node.process(make_flow_data(observations=[obs_high]))) == 0


# =============================================================================
# SamplerNode Tests
# =============================================================================


class TestSamplerNode:
    """Tests for SamplerNode."""

    def test_every_1(self):
        """Test sampling every frame (no sampling)."""
        node = SamplerNode("sample", every_nth=1)

        for i in range(10):
            data = make_flow_data()
            outputs = node.process(data)
            assert len(outputs) == 1

    def test_every_3(self):
        """Test sampling every 3rd frame."""
        node = SamplerNode("sample", every_nth=3)
        results = []

        for i in range(9):
            data = make_flow_data()
            outputs = node.process(data)
            results.append(len(outputs))

        # Should pass on frames 3, 6, 9 (0-indexed: 2, 5, 8)
        assert results == [0, 0, 1, 0, 0, 1, 0, 0, 1]

    def test_reset(self):
        """Test resetting sampler."""
        node = SamplerNode("sample", every_nth=3)

        # Process 2 frames (not enough to trigger)
        node.process(make_flow_data())
        node.process(make_flow_data())

        # Reset
        node.reset()

        # Should need 3 more frames to trigger
        results = []
        for i in range(3):
            outputs = node.process(make_flow_data())
            results.append(len(outputs))

        assert results == [0, 0, 1]

    def test_invalid_every_nth(self):
        """Test that invalid every_nth raises error."""
        with pytest.raises(ValueError):
            SamplerNode("sample", every_nth=0)


class TestRateLimiterNode:
    """Tests for RateLimiterNode."""

    def test_first_frame_passes(self):
        """Test first frame always passes."""
        node = RateLimiterNode("limit", min_interval_ms=100)
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1

    def test_rate_limiting(self):
        """Test that rapid frames are filtered."""
        node = RateLimiterNode("limit", min_interval_ms=50)

        # First frame passes
        outputs1 = node.process(make_flow_data())
        assert len(outputs1) == 1

        # Immediate second frame should be blocked
        outputs2 = node.process(make_flow_data())
        assert len(outputs2) == 0

        # Wait and third frame should pass
        time.sleep(0.06)  # 60ms
        outputs3 = node.process(make_flow_data())
        assert len(outputs3) == 1


class TestTimestampSamplerNode:
    """Tests for TimestampSamplerNode."""

    def test_first_frame_passes(self):
        """Test first frame always passes."""
        node = TimestampSamplerNode("ts_sample", interval_ns=100_000_000)
        data = make_flow_data(timestamp_ns=0)

        outputs = node.process(data)

        assert len(outputs) == 1

    def test_timestamp_based_sampling(self):
        """Test sampling based on timestamps."""
        node = TimestampSamplerNode("ts_sample", interval_ns=100_000_000)  # 100ms

        # Frame at 0ms - passes (first frame)
        assert len(node.process(make_flow_data(timestamp_ns=0))) == 1

        # Frame at 50ms - blocked (not enough time)
        assert len(node.process(make_flow_data(timestamp_ns=50_000_000))) == 0

        # Frame at 100ms - passes (interval elapsed)
        assert len(node.process(make_flow_data(timestamp_ns=100_000_000))) == 1

        # Frame at 120ms - blocked
        assert len(node.process(make_flow_data(timestamp_ns=120_000_000))) == 0


# =============================================================================
# BranchNode Tests
# =============================================================================


class TestBranchNode:
    """Tests for BranchNode."""

    def test_branch_true(self):
        """Test branching when condition is true."""
        node = BranchNode(
            "branch",
            condition=lambda d: True,
            if_true="path_a",
            if_false="path_b",
        )
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1
        assert outputs[0].path_id == "path_a"

    def test_branch_false(self):
        """Test branching when condition is false."""
        node = BranchNode(
            "branch",
            condition=lambda d: False,
            if_true="path_a",
            if_false="path_b",
        )
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1
        assert outputs[0].path_id == "path_b"

    def test_branch_with_data_condition(self):
        """Test branching based on data content."""
        node = BranchNode(
            "branch",
            condition=lambda d: len(d.observations) > 0,
            if_true="has_obs",
            if_false="no_obs",
        )

        obs = Observation(source="test", frame_id=1, t_ns=1000)
        data_with_obs = make_flow_data(observations=[obs])
        data_without_obs = make_flow_data(observations=[])

        assert node.process(data_with_obs)[0].path_id == "has_obs"
        assert node.process(data_without_obs)[0].path_id == "no_obs"


class TestFanOutNode:
    """Tests for FanOutNode."""

    def test_fanout_multiple_paths(self):
        """Test fanning out to multiple paths."""
        node = FanOutNode("fanout", paths=["a", "b", "c"])
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 3
        path_ids = {o.path_id for o in outputs}
        assert path_ids == {"a", "b", "c"}

    def test_fanout_data_cloned(self):
        """Test that fanout clones data."""
        node = FanOutNode("fanout", paths=["a", "b"])
        data = make_flow_data()

        outputs = node.process(data)

        assert outputs[0] is not outputs[1]
        assert outputs[0].frame is outputs[1].frame  # Frame reference shared

    def test_fanout_empty_paths_raises(self):
        """Test that empty paths raises error."""
        with pytest.raises(ValueError):
            FanOutNode("fanout", paths=[])


class TestMultiBranchNode:
    """Tests for MultiBranchNode."""

    def test_first_match(self):
        """Test routing to first matching branch."""
        node = MultiBranchNode(
            "multi",
            branches=[
                (lambda d: d.metadata.get("type") == "a", "path_a"),
                (lambda d: d.metadata.get("type") == "b", "path_b"),
            ],
            default="default",
        )

        data_a = make_flow_data()
        data_a = data_a.clone(metadata={"type": "a"})

        outputs = node.process(data_a)

        assert len(outputs) == 1
        assert outputs[0].path_id == "path_a"

    def test_default_path(self):
        """Test routing to default when no match."""
        node = MultiBranchNode(
            "multi",
            branches=[
                (lambda d: False, "never"),
            ],
            default="default",
        )
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 1
        assert outputs[0].path_id == "default"

    def test_no_match_no_default(self):
        """Test no output when no match and no default."""
        node = MultiBranchNode(
            "multi",
            branches=[
                (lambda d: False, "never"),
            ],
            default=None,
        )
        data = make_flow_data()

        outputs = node.process(data)

        assert len(outputs) == 0


# =============================================================================
# JoinNode Tests
# =============================================================================


class TestJoinNode:
    """Tests for JoinNode."""

    def test_join_all_mode(self):
        """Test joining in 'all' mode."""
        node = JoinNode(
            "join",
            input_paths=["a", "b"],
            mode="all",
            window_ns=100_000_000,
            output_path_id="merged",
        )

        data_a = make_flow_data(path_id="a", timestamp_ns=0)
        data_b = make_flow_data(path_id="b", timestamp_ns=0)

        # First input - should buffer
        outputs_a = node.process(data_a)
        assert len(outputs_a) == 0

        # Second input - should emit
        outputs_b = node.process(data_b)
        assert len(outputs_b) == 1
        assert outputs_b[0].path_id == "merged"

    def test_join_any_mode(self):
        """Test joining in 'any' mode."""
        node = JoinNode(
            "join",
            input_paths=["a", "b"],
            mode="any",
            output_path_id="merged",
        )

        data_a = make_flow_data(path_id="a", timestamp_ns=0)

        # Should emit immediately
        outputs = node.process(data_a)
        assert len(outputs) == 1

    def test_join_merges_observations(self):
        """Test that join merges observations."""
        node = JoinNode(
            "join",
            input_paths=["a", "b"],
            mode="all",
            merge_observations=True,
        )

        obs_a = Observation(source="ext_a", frame_id=1, t_ns=1000)
        obs_b = Observation(source="ext_b", frame_id=1, t_ns=1000)

        data_a = make_flow_data(path_id="a", timestamp_ns=0, observations=[obs_a])
        data_b = make_flow_data(path_id="b", timestamp_ns=0, observations=[obs_b])

        node.process(data_a)
        outputs = node.process(data_b)

        assert len(outputs) == 1
        assert len(outputs[0].observations) == 2

    def test_join_passthrough_unknown_path(self):
        """Test that unknown paths pass through."""
        node = JoinNode(
            "join",
            input_paths=["a", "b"],
            mode="all",
        )

        data_c = make_flow_data(path_id="c", timestamp_ns=0)

        outputs = node.process(data_c)

        assert len(outputs) == 1
        assert outputs[0].path_id == "c"

    def test_flush(self):
        """Test flushing pending buffers."""
        node = JoinNode(
            "join",
            input_paths=["a", "b"],
            mode="all",
        )

        data_a = make_flow_data(path_id="a", timestamp_ns=0)
        node.process(data_a)

        # Flush should emit incomplete buffer
        outputs = node.flush()
        assert len(outputs) == 1


class TestCascadeFusionNode:
    """Tests for CascadeFusionNode."""

    def test_applies_fusion_function(self):
        """Test that fusion function is applied."""
        def add_score(data: FlowData) -> FlowData:
            new_meta = dict(data.metadata)
            new_meta["total_score"] = sum(r.score for r in data.results)
            return data.clone(metadata=new_meta)

        node = CascadeFusionNode("cascade", fusion_fn=add_score)

        result1 = FusionResult(should_trigger=False, score=0.3)
        result2 = FusionResult(should_trigger=False, score=0.5)
        data = make_flow_data()
        data = data.clone(results=[result1, result2])

        outputs = node.process(data)

        assert len(outputs) == 1
        assert outputs[0].metadata["total_score"] == 0.8


class TestCollectorNode:
    """Tests for CollectorNode."""

    def test_collect_by_batch_size(self):
        """Test collecting by batch size."""
        node = CollectorNode("collect", batch_size=3)

        # First two don't emit
        assert len(node.process(make_flow_data())) == 0
        assert len(node.process(make_flow_data())) == 0

        # Third emits
        outputs = node.process(make_flow_data())
        assert len(outputs) == 1
        assert outputs[0].metadata["_batch_size"] == 3

    def test_flush_partial(self):
        """Test flushing partial batch."""
        node = CollectorNode("collect", batch_size=5, emit_partial=True)

        node.process(make_flow_data())
        node.process(make_flow_data())

        outputs = node.flush()
        assert len(outputs) == 1
        assert outputs[0].metadata["_batch_size"] == 2


# =============================================================================
# PathNode Tests
# =============================================================================


class TestPathNode:
    """Tests for PathNode."""

    def test_with_existing_path(self):
        """Test wrapping existing Path."""
        ext = CountingExtractor("ext1", return_value=0.7)
        path = Path(name="test_path", extractors=[ext])

        node = PathNode(path=path)

        assert node.name == "test_path"
        assert node.path is path

    def test_with_components(self):
        """Test creating path from components."""
        ext = CountingExtractor("ext1")
        fusion = ThresholdFusion()

        node = PathNode(name="my_path", extractors=[ext], fusion=fusion)

        assert node.name == "my_path"

    def test_process_adds_observations(self):
        """Test that process adds observations."""
        ext = CountingExtractor("ext1", return_value=0.7)
        path = Path(name="test", extractors=[ext])
        node = PathNode(path=path, run_fusion=False)

        frame = make_frame()
        data = make_flow_data(frame=frame)

        with node:
            outputs = node.process(data)

        assert len(outputs) == 1
        assert len(outputs[0].observations) == 1
        assert outputs[0].observations[0].source == "ext1"

    def test_process_runs_fusion(self):
        """Test that process runs fusion when enabled."""
        ext = CountingExtractor("ext1", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        path = Path(name="test", extractors=[ext], fusion=fusion)
        node = PathNode(path=path, run_fusion=True)

        frame = make_frame()
        data = make_flow_data(frame=frame)

        with node:
            outputs = node.process(data)

        assert len(outputs) == 1
        assert len(outputs[0].results) == 1
        assert outputs[0].results[0].should_trigger  # 0.7 > 0.5

    def test_updates_path_id(self):
        """Test that process updates path_id."""
        ext = CountingExtractor("ext1")
        path = Path(name="human", extractors=[ext])
        node = PathNode(path=path, run_fusion=False)

        frame = make_frame()
        data = make_flow_data(frame=frame, path_id="original")

        with node:
            outputs = node.process(data)

        assert outputs[0].path_id == "human"

    def test_requires_name_or_path(self):
        """Test that either name or path is required."""
        with pytest.raises(ValueError):
            PathNode()
