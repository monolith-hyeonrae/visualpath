"""Tests for flow nodes and SimpleInterpreter.

The new architecture: nodes are declarative (spec only), the interpreter
executes them. Tests verify:
1. Node spec correctness (node declares correctly)
2. Interpreter execution (interpreter executes spec correctly)
"""

import pytest
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import (
    Module,
    Observation,
)
from visualpath.flow import (
    FlowData,
    FlowNode,
    SimpleInterpreter,
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


class CountingModule(Module):
    """Module that counts calls for testing."""

    def __init__(self, name: str, return_value: float = 0.5):
        self._name = name
        self._return_value = return_value
        self._extract_count = 0
        self._initialized = False
        self._cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
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


class ThresholdTriggerModule(Module):
    """Trigger module for testing."""

    def __init__(self, threshold: float = 0.5, depends_on: str = None):
        self._threshold = threshold
        self._gate_open = True
        self._cooldown = False
        self._update_count = 0
        self._depends_on = depends_on
        self.depends = [depends_on] if depends_on else []

    @property
    def name(self) -> str:
        return "threshold_trigger"

    def process(self, frame, deps=None) -> Observation:
        self._update_count += 1
        obs = None
        if deps:
            if self._depends_on and self._depends_on in deps:
                obs = deps[self._depends_on]
            else:
                for v in deps.values():
                    if hasattr(v, 'signals'):
                        obs = v
                        break

        value = obs.signals.get("value", 0) if obs else 0
        if value > self._threshold:
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
        self._update_count = 0


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
# FlowData Tests (unchanged - FlowData is a plain dataclass)
# =============================================================================


class TestFlowData:
    """Tests for FlowData dataclass."""

    def test_basic_creation(self):
        data = FlowData()
        assert data.frame is None
        assert data.observations == []
        assert data.results == []
        assert data.metadata == {}
        assert data.path_id == "default"
        assert data.timestamp_ns == 0

    def test_with_values(self):
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
        data = make_flow_data(path_id="original")
        clone = data.clone()
        assert clone.path_id == "original"
        assert clone is not data
        assert clone.observations is not data.observations

    def test_clone_with_overrides(self):
        data = make_flow_data(path_id="original")
        clone = data.clone(path_id="modified", timestamp_ns=5000)
        assert clone.path_id == "modified"
        assert clone.timestamp_ns == 5000
        assert data.path_id == "original"

    def test_with_path(self):
        data = make_flow_data(path_id="original")
        new_data = data.with_path("new_path")
        assert new_data.path_id == "new_path"
        assert data.path_id == "original"

    def test_add_observation(self):
        data = make_flow_data()
        obs = Observation(source="test", frame_id=1, t_ns=1000)
        new_data = data.add_observation(obs)
        assert len(new_data.observations) == 1
        assert len(data.observations) == 0

    def test_add_result(self):
        data = make_flow_data()
        result = Observation(
            source="trigger",
            frame_id=1,
            t_ns=1000,
            signals={"should_trigger": True, "trigger_score": 0.9},
        )
        new_data = data.add_result(result)
        assert len(new_data.results) == 1
        assert len(data.results) == 0


# =============================================================================
# SourceNode Tests (via interpreter)
# =============================================================================


class TestSourceNode:
    """Tests for SourceNode spec and interpreter execution."""

    def test_basic_creation(self):
        node = SourceNode("my_source")
        assert node.name == "my_source"

    def test_interpret_passthrough(self):
        """Interpreter passes data through for SourceSpec."""
        node = SourceNode("source")
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)

        assert len(outputs) == 1
        assert outputs[0] is data


# =============================================================================
# FilterNode Tests (via interpreter)
# =============================================================================


class TestFilterNode:
    """Tests for FilterNode via interpreter."""

    def test_filter_passes(self):
        node = FilterNode("always_pass", condition=lambda d: True)
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1

    def test_filter_blocks(self):
        node = FilterNode("always_block", condition=lambda d: False)
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 0

    def test_filter_with_condition(self):
        node = FilterNode(
            "has_metadata",
            condition=lambda d: "key" in d.metadata,
        )
        interp = SimpleInterpreter()

        data_with_key = make_flow_data()
        data_with_key = data_with_key.clone(metadata={"key": "value"})
        data_without_key = make_flow_data()

        assert len(interp.interpret(node, data_with_key)) == 1
        assert len(interp.interpret(node, data_without_key)) == 0


class TestObservationFilter:
    """Tests for ObservationFilter via interpreter."""

    def test_passes_with_observations(self):
        node = ObservationFilter("obs_filter", min_count=1)
        interp = SimpleInterpreter()
        obs = Observation(source="test", frame_id=1, t_ns=1000)
        data = make_flow_data(observations=[obs])

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1

    def test_blocks_without_observations(self):
        node = ObservationFilter("obs_filter", min_count=1)
        interp = SimpleInterpreter()
        data = make_flow_data(observations=[])

        outputs = interp.interpret(node, data)
        assert len(outputs) == 0

    def test_min_count(self):
        node = ObservationFilter("obs_filter", min_count=3)
        interp = SimpleInterpreter()
        obs = Observation(source="test", frame_id=1, t_ns=1000)

        data_1 = make_flow_data(observations=[obs])
        data_3 = make_flow_data(observations=[obs, obs, obs])

        assert len(interp.interpret(node, data_1)) == 0
        assert len(interp.interpret(node, data_3)) == 1


class TestSignalThresholdFilter:
    """Tests for SignalThresholdFilter via interpreter."""

    def test_gt_comparison(self):
        node = SignalThresholdFilter("score_filter", "score", threshold=0.5, comparison="gt")
        interp = SimpleInterpreter()
        obs_low = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.3})
        obs_high = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.8})

        assert len(interp.interpret(node, make_flow_data(observations=[obs_low]))) == 0
        assert len(interp.interpret(node, make_flow_data(observations=[obs_high]))) == 1

    def test_le_comparison(self):
        node = SignalThresholdFilter("score_filter", "score", threshold=0.5, comparison="le")
        interp = SimpleInterpreter()
        obs_low = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.3})
        obs_high = Observation(source="test", frame_id=1, t_ns=1000, signals={"score": 0.8})

        assert len(interp.interpret(node, make_flow_data(observations=[obs_low]))) == 1
        assert len(interp.interpret(node, make_flow_data(observations=[obs_high]))) == 0


# =============================================================================
# SamplerNode Tests (via interpreter - state is in interpreter)
# =============================================================================


class TestSamplerNode:
    """Tests for SamplerNode via interpreter."""

    def test_every_1(self):
        node = SamplerNode("sample", every_nth=1)
        interp = SimpleInterpreter()

        for i in range(10):
            data = make_flow_data()
            outputs = interp.interpret(node, data)
            assert len(outputs) == 1

    def test_every_3(self):
        node = SamplerNode("sample", every_nth=3)
        interp = SimpleInterpreter()
        results = []

        for i in range(9):
            data = make_flow_data()
            outputs = interp.interpret(node, data)
            results.append(len(outputs))

        assert results == [0, 0, 1, 0, 0, 1, 0, 0, 1]

    def test_reset(self):
        node = SamplerNode("sample", every_nth=3)
        interp = SimpleInterpreter()

        interp.interpret(node, make_flow_data())
        interp.interpret(node, make_flow_data())

        # Reset interpreter state for this node
        interp.reset_node(node.name)

        results = []
        for i in range(3):
            outputs = interp.interpret(node, make_flow_data())
            results.append(len(outputs))

        assert results == [0, 0, 1]

    def test_invalid_every_nth(self):
        with pytest.raises(ValueError):
            SamplerNode("sample", every_nth=0)


class TestRateLimiterNode:
    """Tests for RateLimiterNode via interpreter."""

    def test_first_frame_passes(self):
        node = RateLimiterNode("limit", min_interval_ms=100)
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1

    def test_rate_limiting(self):
        node = RateLimiterNode("limit", min_interval_ms=50)
        interp = SimpleInterpreter()

        # First frame passes
        outputs1 = interp.interpret(node, make_flow_data())
        assert len(outputs1) == 1

        # Immediate second frame should be blocked
        outputs2 = interp.interpret(node, make_flow_data())
        assert len(outputs2) == 0

        # Wait and third frame should pass
        time.sleep(0.06)
        outputs3 = interp.interpret(node, make_flow_data())
        assert len(outputs3) == 1


class TestTimestampSamplerNode:
    """Tests for TimestampSamplerNode via interpreter."""

    def test_first_frame_passes(self):
        node = TimestampSamplerNode("ts_sample", interval_ns=100_000_000)
        interp = SimpleInterpreter()
        data = make_flow_data(timestamp_ns=0)

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1

    def test_timestamp_based_sampling(self):
        node = TimestampSamplerNode("ts_sample", interval_ns=100_000_000)
        interp = SimpleInterpreter()

        assert len(interp.interpret(node, make_flow_data(timestamp_ns=0))) == 1
        assert len(interp.interpret(node, make_flow_data(timestamp_ns=50_000_000))) == 0
        assert len(interp.interpret(node, make_flow_data(timestamp_ns=100_000_000))) == 1
        assert len(interp.interpret(node, make_flow_data(timestamp_ns=120_000_000))) == 0


# =============================================================================
# BranchNode Tests (via interpreter)
# =============================================================================


class TestBranchNode:
    """Tests for BranchNode via interpreter."""

    def test_branch_true(self):
        node = BranchNode("branch", condition=lambda d: True, if_true="path_a", if_false="path_b")
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1
        assert outputs[0].path_id == "path_a"

    def test_branch_false(self):
        node = BranchNode("branch", condition=lambda d: False, if_true="path_a", if_false="path_b")
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1
        assert outputs[0].path_id == "path_b"

    def test_branch_with_data_condition(self):
        node = BranchNode(
            "branch",
            condition=lambda d: len(d.observations) > 0,
            if_true="has_obs",
            if_false="no_obs",
        )
        interp = SimpleInterpreter()

        obs = Observation(source="test", frame_id=1, t_ns=1000)
        data_with_obs = make_flow_data(observations=[obs])
        data_without_obs = make_flow_data(observations=[])

        assert interp.interpret(node, data_with_obs)[0].path_id == "has_obs"
        assert interp.interpret(node, data_without_obs)[0].path_id == "no_obs"


class TestFanOutNode:
    """Tests for FanOutNode via interpreter."""

    def test_fanout_multiple_paths(self):
        node = FanOutNode("fanout", paths=["a", "b", "c"])
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 3
        path_ids = {o.path_id for o in outputs}
        assert path_ids == {"a", "b", "c"}

    def test_fanout_data_cloned(self):
        node = FanOutNode("fanout", paths=["a", "b"])
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert outputs[0] is not outputs[1]
        assert outputs[0].frame is outputs[1].frame

    def test_fanout_empty_paths_raises(self):
        with pytest.raises(ValueError):
            FanOutNode("fanout", paths=[])


class TestMultiBranchNode:
    """Tests for MultiBranchNode via interpreter."""

    def test_first_match(self):
        node = MultiBranchNode(
            "multi",
            branches=[
                (lambda d: d.metadata.get("type") == "a", "path_a"),
                (lambda d: d.metadata.get("type") == "b", "path_b"),
            ],
            default="default",
        )
        interp = SimpleInterpreter()

        data_a = make_flow_data()
        data_a = data_a.clone(metadata={"type": "a"})
        outputs = interp.interpret(node, data_a)
        assert len(outputs) == 1
        assert outputs[0].path_id == "path_a"

    def test_default_path(self):
        node = MultiBranchNode(
            "multi",
            branches=[(lambda d: False, "never")],
            default="default",
        )
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1
        assert outputs[0].path_id == "default"

    def test_no_match_no_default(self):
        node = MultiBranchNode(
            "multi",
            branches=[(lambda d: False, "never")],
            default=None,
        )
        interp = SimpleInterpreter()
        data = make_flow_data()

        outputs = interp.interpret(node, data)
        assert len(outputs) == 0


# =============================================================================
# JoinNode Tests (via interpreter - buffers are in interpreter state)
# =============================================================================


class TestJoinNode:
    """Tests for JoinNode via interpreter."""

    def test_join_all_mode(self):
        node = JoinNode("join", input_paths=["a", "b"], mode="all", window_ns=100_000_000, output_path_id="merged")
        interp = SimpleInterpreter()

        data_a = make_flow_data(path_id="a", timestamp_ns=0)
        data_b = make_flow_data(path_id="b", timestamp_ns=0)

        outputs_a = interp.interpret(node, data_a)
        assert len(outputs_a) == 0

        outputs_b = interp.interpret(node, data_b)
        assert len(outputs_b) == 1
        assert outputs_b[0].path_id == "merged"

    def test_join_any_mode(self):
        node = JoinNode("join", input_paths=["a", "b"], mode="any", output_path_id="merged")
        interp = SimpleInterpreter()

        data_a = make_flow_data(path_id="a", timestamp_ns=0)
        outputs = interp.interpret(node, data_a)
        assert len(outputs) == 1

    def test_join_merges_observations(self):
        node = JoinNode("join", input_paths=["a", "b"], mode="all", merge_observations=True)
        interp = SimpleInterpreter()

        obs_a = Observation(source="ext_a", frame_id=1, t_ns=1000)
        obs_b = Observation(source="ext_b", frame_id=1, t_ns=1000)

        data_a = make_flow_data(path_id="a", timestamp_ns=0, observations=[obs_a])
        data_b = make_flow_data(path_id="b", timestamp_ns=0, observations=[obs_b])

        interp.interpret(node, data_a)
        outputs = interp.interpret(node, data_b)

        assert len(outputs) == 1
        assert len(outputs[0].observations) == 2

    def test_join_passthrough_unknown_path(self):
        node = JoinNode("join", input_paths=["a", "b"], mode="all")
        interp = SimpleInterpreter()

        data_c = make_flow_data(path_id="c", timestamp_ns=0)
        outputs = interp.interpret(node, data_c)

        assert len(outputs) == 1
        assert outputs[0].path_id == "c"

    def test_flush(self):
        node = JoinNode("join", input_paths=["a", "b"], mode="all")
        interp = SimpleInterpreter()

        data_a = make_flow_data(path_id="a", timestamp_ns=0)
        interp.interpret(node, data_a)

        outputs = interp.flush_node(node)
        assert len(outputs) == 1


class TestCascadeFusionNode:
    """Tests for CascadeFusionNode via interpreter."""

    def test_applies_fusion_function(self):
        def add_score(data: FlowData) -> FlowData:
            new_meta = dict(data.metadata)
            new_meta["total_score"] = sum(r.trigger_score for r in data.results)
            return data.clone(metadata=new_meta)

        node = CascadeFusionNode("cascade", fusion_fn=add_score)
        interp = SimpleInterpreter()

        result1 = Observation(
            source="trigger1",
            frame_id=1,
            t_ns=1000,
            signals={"should_trigger": False, "trigger_score": 0.3},
        )
        result2 = Observation(
            source="trigger2",
            frame_id=1,
            t_ns=1000,
            signals={"should_trigger": False, "trigger_score": 0.5},
        )
        data = make_flow_data()
        data = data.clone(results=[result1, result2])

        outputs = interp.interpret(node, data)
        assert len(outputs) == 1
        assert outputs[0].metadata["total_score"] == 0.8


class TestCollectorNode:
    """Tests for CollectorNode via interpreter."""

    def test_collect_by_batch_size(self):
        node = CollectorNode("collect", batch_size=3)
        interp = SimpleInterpreter()

        assert len(interp.interpret(node, make_flow_data())) == 0
        assert len(interp.interpret(node, make_flow_data())) == 0

        outputs = interp.interpret(node, make_flow_data())
        assert len(outputs) == 1
        assert outputs[0].metadata["_batch_size"] == 3

    def test_flush_partial(self):
        node = CollectorNode("collect", batch_size=5, emit_partial=True)
        interp = SimpleInterpreter()

        interp.interpret(node, make_flow_data())
        interp.interpret(node, make_flow_data())

        outputs = interp.flush_node(node)
        assert len(outputs) == 1
        assert outputs[0].metadata["_batch_size"] == 2


# =============================================================================
# PathNode Tests (via interpreter)
# =============================================================================


class TestPathNode:
    """Tests for PathNode via interpreter."""

    def test_with_existing_path(self):
        ext = CountingModule("ext1", return_value=0.7)
        path = Path(name="test_path", extractors=[ext])
        node = PathNode(path=path)
        assert node.name == "test_path"
        assert node.path is path

    def test_with_components(self):
        ext = CountingModule("ext1")
        trigger = ThresholdTriggerModule()
        node = PathNode(name="my_path", extractors=[ext], fusion=trigger)
        assert node.name == "my_path"

    def test_interpret_adds_observations(self):
        ext = CountingModule("ext1", return_value=0.7)
        path = Path(name="test", extractors=[ext])
        node = PathNode(path=path, run_fusion=False)
        interp = SimpleInterpreter()

        frame = make_frame()
        data = make_flow_data(frame=frame)

        with node:
            outputs = interp.interpret(node, data)

        assert len(outputs) == 1
        assert len(outputs[0].observations) == 1
        assert outputs[0].observations[0].source == "ext1"

    def test_interpret_runs_fusion(self):
        ext = CountingModule("ext1", return_value=0.7)
        trigger = ThresholdTriggerModule(threshold=0.5, depends_on="ext1")
        path = Path(name="test", extractors=[ext], fusion=trigger)
        node = PathNode(path=path, run_fusion=True)
        interp = SimpleInterpreter()

        frame = make_frame()
        data = make_flow_data(frame=frame)

        with node:
            outputs = interp.interpret(node, data)

        assert len(outputs) == 1
        assert len(outputs[0].results) == 1
        assert outputs[0].results[0].should_trigger  # 0.7 > 0.5

    def test_updates_path_id(self):
        ext = CountingModule("ext1")
        path = Path(name="human", extractors=[ext])
        node = PathNode(path=path, run_fusion=False)
        interp = SimpleInterpreter()

        frame = make_frame()
        data = make_flow_data(frame=frame, path_id="original")

        with node:
            outputs = interp.interpret(node, data)

        assert outputs[0].path_id == "human"

    def test_requires_name_or_path(self):
        with pytest.raises(ValueError):
            PathNode()
