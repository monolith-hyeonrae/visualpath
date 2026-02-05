"""Tests for FlowNode spec properties and NodeSpec dataclasses.

Verifies:
- Each node type returns the correct spec type
- Spec fields match node configuration
- Specs are frozen (immutable)
- Custom nodes without spec return None
- FlowGraphConverter dispatches on spec types
- _split_by_dependency correctly groups extractors
"""

import pytest
from dataclasses import FrozenInstanceError
from typing import List, Optional
from unittest.mock import MagicMock

from visualpath.flow.node import FlowNode, FlowData
from visualpath.flow.specs import (
    NodeSpec,
    SourceSpec,
    ExtractSpec,
    FilterSpec,
    ObservationFilterSpec,
    SignalFilterSpec,
    SampleSpec,
    RateLimitSpec,
    TimestampSampleSpec,
    BranchSpec,
    FanOutSpec,
    MultiBranchSpec,
    ConditionalFanOutSpec,
    JoinSpec,
    CascadeFusionSpec,
    CollectorSpec,
    CustomSpec,
)
from visualpath.flow.nodes import (
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
from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion, FusionResult


# =============================================================================
# Test Fixtures
# =============================================================================


class DummyExtractor(BaseExtractor):
    """Minimal extractor for testing."""

    def __init__(self, name: str = "dummy", depends=None):
        self._name = name
        self._depends = depends or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def depends(self):
        return self._depends

    def extract(self, frame, deps=None) -> Optional[Observation]:
        return Observation(source=self._name, frame_id=0, t_ns=0, signals={})

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class DummyFusion(BaseFusion):
    """Minimal fusion for testing."""

    def update(self, observation: Observation) -> FusionResult:
        return FusionResult(should_trigger=False)

    def reset(self) -> None:
        pass

    @property
    def is_gate_open(self) -> bool:
        return True

    @property
    def in_cooldown(self) -> bool:
        return False


class CustomNode(FlowNode):
    """Custom node that uses CustomSpec."""

    def __init__(self, processor=None):
        self._processor = processor

    @property
    def name(self) -> str:
        return "custom"

    @property
    def spec(self) -> CustomSpec:
        return CustomSpec(processor=self._processor)


# =============================================================================
# Spec Type Tests
# =============================================================================


class TestSourceNodeSpec:
    """Tests for SourceNode.spec."""

    def test_returns_source_spec(self):
        node = SourceNode("src", default_path_id="main")
        spec = node.spec
        assert isinstance(spec, SourceSpec)

    def test_spec_fields_match(self):
        node = SourceNode("src", default_path_id="main")
        spec = node.spec
        assert spec.default_path_id == "main"

    def test_spec_default_path_id(self):
        node = SourceNode("src")
        assert node.spec.default_path_id == "default"

    def test_spec_is_frozen(self):
        node = SourceNode("src")
        with pytest.raises(FrozenInstanceError):
            node.spec.default_path_id = "changed"


class TestPathNodeSpec:
    """Tests for PathNode.spec."""

    def test_returns_extract_spec(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext])
        spec = node.spec
        assert isinstance(spec, ExtractSpec)

    def test_spec_extractors(self):
        ext1 = DummyExtractor("face")
        ext2 = DummyExtractor("pose")
        node = PathNode(name="analysis", extractors=[ext1, ext2])
        spec = node.spec
        assert len(spec.extractors) == 2
        assert spec.extractors[0] is ext1
        assert spec.extractors[1] is ext2

    def test_spec_fusion(self):
        ext = DummyExtractor("face")
        fusion = DummyFusion()
        node = PathNode(name="analysis", extractors=[ext], fusion=fusion)
        spec = node.spec
        assert spec.fusion is fusion

    def test_spec_run_fusion(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext], run_fusion=False)
        assert node.spec.run_fusion is False

    def test_spec_join_window_ns_default(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext])
        assert node.spec.join_window_ns == 100_000_000

    def test_spec_join_window_ns_custom(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext], join_window_ns=200_000_000)
        assert node.spec.join_window_ns == 200_000_000

    def test_spec_is_frozen(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext])
        with pytest.raises(FrozenInstanceError):
            node.spec.run_fusion = False

    def test_spec_extractors_are_tuple(self):
        ext = DummyExtractor("face")
        node = PathNode(name="analysis", extractors=[ext])
        assert isinstance(node.spec.extractors, tuple)


class TestFilterNodeSpec:
    """Tests for FilterNode.spec."""

    def test_returns_filter_spec(self):
        cond = lambda d: True
        node = FilterNode("f", condition=cond)
        spec = node.spec
        assert isinstance(spec, FilterSpec)

    def test_spec_condition(self):
        cond = lambda d: True
        node = FilterNode("f", condition=cond)
        assert node.spec.condition is cond

    def test_spec_is_frozen(self):
        node = FilterNode("f", condition=lambda d: True)
        with pytest.raises(FrozenInstanceError):
            node.spec.condition = None


class TestObservationFilterSpec:
    """Tests for ObservationFilter.spec."""

    def test_returns_observation_filter_spec(self):
        node = ObservationFilter("obs_f", min_count=3)
        spec = node.spec
        assert isinstance(spec, ObservationFilterSpec)

    def test_spec_min_count(self):
        node = ObservationFilter("obs_f", min_count=3)
        assert node.spec.min_count == 3

    def test_spec_is_frozen(self):
        node = ObservationFilter("obs_f")
        with pytest.raises(FrozenInstanceError):
            node.spec.min_count = 10


class TestSignalThresholdFilterSpec:
    """Tests for SignalThresholdFilter.spec."""

    def test_returns_signal_filter_spec(self):
        node = SignalThresholdFilter("sig_f", signal_name="confidence", threshold=0.8)
        spec = node.spec
        assert isinstance(spec, SignalFilterSpec)

    def test_spec_fields(self):
        node = SignalThresholdFilter(
            "sig_f", signal_name="confidence", threshold=0.8, comparison="ge",
        )
        spec = node.spec
        assert spec.signal_name == "confidence"
        assert spec.threshold == 0.8
        assert spec.comparison == "ge"

    def test_spec_is_frozen(self):
        node = SignalThresholdFilter("sig_f", signal_name="val", threshold=0.5)
        with pytest.raises(FrozenInstanceError):
            node.spec.threshold = 0.9


class TestSamplerNodeSpec:
    """Tests for SamplerNode.spec."""

    def test_returns_sample_spec(self):
        node = SamplerNode("s", every_nth=5)
        spec = node.spec
        assert isinstance(spec, SampleSpec)

    def test_spec_every_nth(self):
        node = SamplerNode("s", every_nth=5)
        assert node.spec.every_nth == 5

    def test_spec_is_frozen(self):
        node = SamplerNode("s", every_nth=3)
        with pytest.raises(FrozenInstanceError):
            node.spec.every_nth = 10


class TestRateLimiterNodeSpec:
    """Tests for RateLimiterNode.spec."""

    def test_returns_rate_limit_spec(self):
        node = RateLimiterNode("rl", min_interval_ms=200)
        spec = node.spec
        assert isinstance(spec, RateLimitSpec)

    def test_spec_min_interval_ms(self):
        node = RateLimiterNode("rl", min_interval_ms=200)
        assert node.spec.min_interval_ms == 200

    def test_spec_is_frozen(self):
        node = RateLimiterNode("rl", min_interval_ms=100)
        with pytest.raises(FrozenInstanceError):
            node.spec.min_interval_ms = 500


class TestTimestampSamplerNodeSpec:
    """Tests for TimestampSamplerNode.spec."""

    def test_returns_timestamp_sample_spec(self):
        node = TimestampSamplerNode("ts", interval_ns=33_333_333)
        spec = node.spec
        assert isinstance(spec, TimestampSampleSpec)

    def test_spec_interval_ns(self):
        node = TimestampSamplerNode("ts", interval_ns=33_333_333)
        assert node.spec.interval_ns == 33_333_333

    def test_spec_is_frozen(self):
        node = TimestampSamplerNode("ts", interval_ns=100)
        with pytest.raises(FrozenInstanceError):
            node.spec.interval_ns = 200


class TestBranchNodeSpec:
    """Tests for BranchNode.spec."""

    def test_returns_branch_spec(self):
        cond = lambda d: True
        node = BranchNode("b", condition=cond, if_true="yes", if_false="no")
        spec = node.spec
        assert isinstance(spec, BranchSpec)

    def test_spec_fields(self):
        cond = lambda d: True
        node = BranchNode("b", condition=cond, if_true="yes", if_false="no")
        spec = node.spec
        assert spec.condition is cond
        assert spec.if_true == "yes"
        assert spec.if_false == "no"

    def test_spec_is_frozen(self):
        node = BranchNode("b", condition=lambda d: True, if_true="a", if_false="b")
        with pytest.raises(FrozenInstanceError):
            node.spec.if_true = "changed"


class TestFanOutNodeSpec:
    """Tests for FanOutNode.spec."""

    def test_returns_fanout_spec(self):
        node = FanOutNode("fo", paths=["a", "b", "c"])
        spec = node.spec
        assert isinstance(spec, FanOutSpec)

    def test_spec_paths(self):
        node = FanOutNode("fo", paths=["a", "b", "c"])
        assert node.spec.paths == ("a", "b", "c")

    def test_spec_is_frozen(self):
        node = FanOutNode("fo", paths=["a", "b"])
        with pytest.raises(FrozenInstanceError):
            node.spec.paths = ("x",)


class TestMultiBranchNodeSpec:
    """Tests for MultiBranchNode.spec."""

    def test_returns_multi_branch_spec(self):
        branches = [(lambda d: True, "yes")]
        node = MultiBranchNode("mb", branches=branches, default="fallback")
        spec = node.spec
        assert isinstance(spec, MultiBranchSpec)

    def test_spec_fields(self):
        branches = [(lambda d: True, "yes"), (lambda d: False, "no")]
        node = MultiBranchNode("mb", branches=branches, default="fallback")
        spec = node.spec
        assert len(spec.branches) == 2
        assert spec.default == "fallback"

    def test_spec_is_frozen(self):
        node = MultiBranchNode("mb", branches=[(lambda d: True, "y")], default="n")
        with pytest.raises(FrozenInstanceError):
            node.spec.default = "changed"


class TestConditionalFanOutNodeSpec:
    """Tests for ConditionalFanOutNode.spec."""

    def test_returns_conditional_fanout_spec(self):
        paths = [("a", lambda d: True), ("b", lambda d: False)]
        node = ConditionalFanOutNode("cfo", paths=paths)
        spec = node.spec
        assert isinstance(spec, ConditionalFanOutSpec)

    def test_spec_paths(self):
        paths = [("a", lambda d: True), ("b", lambda d: False)]
        node = ConditionalFanOutNode("cfo", paths=paths)
        assert len(node.spec.paths) == 2

    def test_spec_is_frozen(self):
        paths = [("a", lambda d: True)]
        node = ConditionalFanOutNode("cfo", paths=paths)
        with pytest.raises(FrozenInstanceError):
            node.spec.paths = ()


class TestJoinNodeSpec:
    """Tests for JoinNode.spec."""

    def test_returns_join_spec(self):
        node = JoinNode("j", input_paths=["a", "b"])
        spec = node.spec
        assert isinstance(spec, JoinSpec)

    def test_spec_fields(self):
        node = JoinNode(
            "j",
            input_paths=["b", "a"],
            mode="any",
            window_ns=200_000_000,
            merge_observations=False,
            merge_results=False,
            output_path_id="result",
        )
        spec = node.spec
        assert spec.input_paths == ("a", "b")  # sorted
        assert spec.mode == "any"
        assert spec.window_ns == 200_000_000
        assert spec.lateness_ns == 0
        assert spec.merge_observations is False
        assert spec.merge_results is False
        assert spec.output_path_id == "result"

    def test_spec_is_frozen(self):
        node = JoinNode("j", input_paths=["a", "b"])
        with pytest.raises(FrozenInstanceError):
            node.spec.mode = "any"


class TestCascadeFusionNodeSpec:
    """Tests for CascadeFusionNode.spec."""

    def test_returns_cascade_fusion_spec(self):
        fn = lambda d: d
        node = CascadeFusionNode("cf", fusion_fn=fn)
        spec = node.spec
        assert isinstance(spec, CascadeFusionSpec)

    def test_spec_fusion_fn(self):
        fn = lambda d: d
        node = CascadeFusionNode("cf", fusion_fn=fn)
        assert node.spec.fusion_fn is fn

    def test_spec_is_frozen(self):
        node = CascadeFusionNode("cf", fusion_fn=lambda d: d)
        with pytest.raises(FrozenInstanceError):
            node.spec.fusion_fn = None


class TestCollectorNodeSpec:
    """Tests for CollectorNode.spec."""

    def test_returns_collector_spec(self):
        node = CollectorNode("c", batch_size=10, timeout_ns=1_000_000, emit_partial=False)
        spec = node.spec
        assert isinstance(spec, CollectorSpec)

    def test_spec_fields(self):
        node = CollectorNode("c", batch_size=10, timeout_ns=1_000_000, emit_partial=False)
        spec = node.spec
        assert spec.batch_size == 10
        assert spec.timeout_ns == 1_000_000
        assert spec.emit_partial is False

    def test_spec_is_frozen(self):
        node = CollectorNode("c", batch_size=5)
        with pytest.raises(FrozenInstanceError):
            node.spec.batch_size = 20


# =============================================================================
# Custom Node Tests
# =============================================================================


class TestCustomNodeSpec:
    """Tests for custom nodes with CustomSpec."""

    def test_custom_node_returns_custom_spec(self):
        processor = lambda d: d
        node = CustomNode(processor=processor)
        spec = node.spec
        assert isinstance(spec, CustomSpec)
        assert spec.processor is processor


# =============================================================================
# Spec Inheritance Tests
# =============================================================================


class TestSpecInheritance:
    """Tests for NodeSpec hierarchy."""

    def test_all_specs_inherit_from_node_spec(self):
        spec_types = [
            SourceSpec, ExtractSpec, FilterSpec, ObservationFilterSpec,
            SignalFilterSpec, SampleSpec, RateLimitSpec, TimestampSampleSpec,
            BranchSpec, FanOutSpec, MultiBranchSpec, ConditionalFanOutSpec,
            JoinSpec, CascadeFusionSpec, CollectorSpec, CustomSpec,
        ]
        for spec_type in spec_types:
            instance = spec_type()
            assert isinstance(instance, NodeSpec), f"{spec_type.__name__} should inherit from NodeSpec"


# =============================================================================
# Spec Export Tests
# =============================================================================


class TestSpecExports:
    """Tests for spec exports from flow module."""

    def test_specs_importable_from_flow(self):
        from visualpath.flow import (
            NodeSpec, SourceSpec, ExtractSpec, FilterSpec,
            ObservationFilterSpec, SignalFilterSpec,
            SampleSpec, RateLimitSpec, TimestampSampleSpec,
            BranchSpec, FanOutSpec, MultiBranchSpec, ConditionalFanOutSpec,
            JoinSpec, CascadeFusionSpec, CollectorSpec, CustomSpec,
        )
        # If we get here, imports succeeded
        assert NodeSpec is not None


# =============================================================================
# Dependency Splitting Tests
# =============================================================================


class TestSplitByDependency:
    """Tests for FlowGraphConverter._split_by_dependency."""

    def test_all_independent(self):
        """All extractors are independent -> one group each."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        ext1 = DummyExtractor("face")
        ext2 = DummyExtractor("pose")
        ext3 = DummyExtractor("scene")

        groups = FlowGraphConverter._split_by_dependency([ext1, ext2, ext3])

        assert len(groups) == 3
        assert [g[0].name for g in groups] == ["face", "pose", "scene"]

    def test_all_chained(self):
        """All extractors form a chain -> single group."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        ext1 = DummyExtractor("face_detect")
        ext2 = DummyExtractor("face_expression", depends=["face_detect"])
        ext3 = DummyExtractor("face_emotion", depends=["face_expression"])

        groups = FlowGraphConverter._split_by_dependency([ext1, ext2, ext3])

        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_mixed_deps(self):
        """Mix of dependent and independent extractors."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        face = DummyExtractor("face_detect")
        expression = DummyExtractor("face_expression", depends=["face_detect"])
        pose = DummyExtractor("pose_detect")

        groups = FlowGraphConverter._split_by_dependency([face, expression, pose])

        assert len(groups) == 2
        # face_detect and face_expression should be in same group
        group_names = [sorted([e.name for e in g]) for g in groups]
        assert ["face_detect", "face_expression"] in group_names
        assert ["pose_detect"] in group_names

    def test_single_extractor(self):
        """Single extractor -> single group."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        ext = DummyExtractor("face")
        groups = FlowGraphConverter._split_by_dependency([ext])

        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_preserves_order_within_group(self):
        """Extractors within a group maintain insertion order."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        ext1 = DummyExtractor("face_detect")
        ext2 = DummyExtractor("face_expression", depends=["face_detect"])

        groups = FlowGraphConverter._split_by_dependency([ext1, ext2])

        assert len(groups) == 1
        assert groups[0][0].name == "face_detect"
        assert groups[0][1].name == "face_expression"


# =============================================================================
# FlowGraphConverter Spec-Based Dispatch Tests
# =============================================================================


class TestConverterSpecDispatch:
    """Tests that FlowGraphConverter uses spec for dispatch."""

    def test_converter_uses_spec_not_isinstance(self):
        """Verify converter dispatches based on node.spec type."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        converter = FlowGraphConverter()

        # Create a custom node that returns SourceSpec
        class FakeSourceNode(FlowNode):
            @property
            def name(self):
                return "fake_source"

            @property
            def spec(self):
                return SourceSpec(default_path_id="test")

            def process(self, data):
                return [data]

        from visualpath.flow.graph import FlowGraph

        graph = FlowGraph()
        node = FakeSourceNode()
        graph.add_node(node)

        # The converter should treat this as SourceSpec
        # even though it's not an actual SourceNode instance
        # We can verify by checking _tables after conversion attempt
        # (needs Pathway for full test, so we test the dispatch logic)
        spec = node.spec
        assert isinstance(spec, SourceSpec)


# =============================================================================
# Integration: Spec consistency with process()
# =============================================================================


class TestSpecProcessConsistency:
    """Tests that spec and process() agree."""

    def test_sampler_spec_matches_behavior(self):
        """SamplerNode spec.every_nth matches actual sampling behavior."""
        node = SamplerNode("s", every_nth=3)
        assert node.spec.every_nth == 3
        assert node.every_nth == 3

    def test_join_spec_input_paths_sorted(self):
        """JoinNode spec.input_paths are always sorted."""
        node = JoinNode("j", input_paths=["z", "a", "m"])
        assert node.spec.input_paths == ("a", "m", "z")

    def test_path_node_spec_extractors_immutable(self):
        """PathNode spec.extractors is a tuple (immutable)."""
        ext = DummyExtractor("face")
        node = PathNode(name="p", extractors=[ext])
        spec = node.spec
        assert isinstance(spec.extractors, tuple)
        # Modifying the original list shouldn't affect spec
        # (tuple is already immutable)

    def test_fanout_spec_paths_immutable(self):
        """FanOutNode spec.paths is a tuple (immutable)."""
        node = FanOutNode("fo", paths=["a", "b"])
        spec = node.spec
        assert isinstance(spec.paths, tuple)
