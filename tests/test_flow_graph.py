"""Tests for FlowGraph, GraphExecutor, and FlowGraphBuilder."""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import (
    Module,
    Observation,
)
from visualpath.flow import (
    FlowData,
    FlowGraph,
    Edge,
    GraphExecutor,
    FlowGraphBuilder,
    SourceNode,
    PathNode,
    SamplerNode,
    FilterNode,
    BranchNode,
    FanOutNode,
    JoinNode,
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


# =============================================================================
# FlowGraph Tests
# =============================================================================


class TestFlowGraph:
    """Tests for FlowGraph class."""

    def test_empty_graph(self):
        """Test creating empty graph."""
        graph = FlowGraph()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.entry_node is None

    def test_add_node(self):
        """Test adding nodes."""
        graph = FlowGraph()
        source = SourceNode("source")

        graph.add_node(source)

        assert "source" in graph.nodes
        assert graph.entry_node == "source"  # First node becomes entry

    def test_add_duplicate_node_raises(self):
        """Test that duplicate node names raise error."""
        graph = FlowGraph()
        source1 = SourceNode("source")
        source2 = SourceNode("source")

        graph.add_node(source1)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(source2)

    def test_add_edge(self):
        """Test adding edges."""
        graph = FlowGraph()
        source = SourceNode("source")
        sampler = SamplerNode("sampler")

        graph.add_node(source)
        graph.add_node(sampler)
        graph.add_edge("source", "sampler")

        assert len(graph.edges) == 1
        assert graph.edges[0].source == "source"
        assert graph.edges[0].target == "sampler"

    def test_add_edge_invalid_source(self):
        """Test that invalid edge source raises error."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))

        with pytest.raises(ValueError, match="not in graph"):
            graph.add_edge("nonexistent", "source")

    def test_add_edge_invalid_target(self):
        """Test that invalid edge target raises error."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))

        with pytest.raises(ValueError, match="not in graph"):
            graph.add_edge("source", "nonexistent")

    def test_get_outgoing_edges(self):
        """Test getting outgoing edges."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        edges = graph.get_outgoing_edges("a")

        assert len(edges) == 2
        targets = {e.target for e in edges}
        assert targets == {"b", "c"}

    def test_get_successors(self):
        """Test getting successor nodes."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        graph.add_edge("a", "c", path_filter="special")

        # Without path filter - only unfiltered edge
        successors_default = graph.get_successors("a", path_id="default")
        assert "b" in successors_default
        assert "c" not in successors_default

        # With matching path filter
        successors_special = graph.get_successors("a", path_id="special")
        assert "b" in successors_special
        assert "c" in successors_special

    def test_get_terminal_nodes(self):
        """Test getting terminal nodes."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        # c has no outgoing edges

        terminals = graph.get_terminal_nodes()

        assert "b" in terminals
        assert "c" in terminals
        assert "a" not in terminals

    def test_validate_no_entry(self):
        """Test validation fails without entry node."""
        graph = FlowGraph()

        with pytest.raises(ValueError, match="No entry node"):
            graph.validate()

    def test_validate_cycle_detection(self):
        """Test validation detects cycles."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")  # Creates cycle

        with pytest.raises(ValueError, match="Cycle detected"):
            graph.validate()

    def test_validate_unreachable(self):
        """Test validation detects unreachable nodes."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        # c is not connected

        with pytest.raises(ValueError, match="Unreachable"):
            graph.validate()

    def test_validate_success(self):
        """Test successful validation."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        graph.validate()  # Should not raise

    def test_topological_order(self):
        """Test topological ordering."""
        graph = FlowGraph()
        graph.add_node(SourceNode("a"))
        graph.add_node(SamplerNode("b"))
        graph.add_node(SamplerNode("c"))
        graph.add_node(SamplerNode("d"))
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        graph.add_edge("c", "d")

        order = graph.topological_order()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_on_trigger(self):
        """Test trigger callback registration."""
        graph = FlowGraph()
        triggered = []

        graph.on_trigger(lambda d: triggered.append(d))

        # Create data with triggering result
        result = Observation(
            source="trigger",
            frame_id=1,
            t_ns=1000000,
            signals={"should_trigger": True, "trigger_score": 0.9},
        )
        data = FlowData(results=[result])

        graph.fire_triggers(data)

        assert len(triggered) == 1

    def test_context_manager(self):
        """Test graph as context manager."""
        ext = CountingModule("ext1")
        path = Path(name="test", extractors=[ext])
        path_node = PathNode(path=path)

        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(path_node)
        graph.add_edge("source", "test")

        assert not ext._initialized

        with graph:
            assert ext._initialized

        assert ext._cleaned_up


# =============================================================================
# GraphExecutor Tests
# =============================================================================


class TestGraphExecutor:
    """Tests for GraphExecutor class."""

    def test_simple_pipeline(self):
        """Test executing a simple pipeline."""
        ext = CountingModule("ext1", return_value=0.7)
        path = Path(name="test", extractors=[ext])
        path_node = PathNode(path=path, run_fusion=False)

        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(path_node)
        graph.add_edge("source", "test")

        executor = GraphExecutor(graph)
        frame = make_frame()

        with executor:
            results = executor.process(frame)

        assert len(results) == 1
        assert len(results[0].observations) == 1

    def test_sampler_in_pipeline(self):
        """Test sampler node in pipeline."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(SamplerNode("sampler", every_nth=3))
        graph.add_edge("source", "sampler")

        executor = GraphExecutor(graph)
        results = []

        with executor:
            for i in range(9):
                frame = make_frame(frame_id=i)
                r = executor.process(frame)
                results.extend(r)

        # Only every 3rd frame passes
        assert len(results) == 3

    def test_branch_in_pipeline(self):
        """Test branching in pipeline."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(BranchNode(
            "branch",
            condition=lambda d: d.metadata.get("go_a", False),
            if_true="a",
            if_false="b",
        ))
        graph.add_node(SamplerNode("path_a"))
        graph.add_node(SamplerNode("path_b"))
        graph.add_edge("source", "branch")
        graph.add_edge("branch", "path_a", path_filter="a")
        graph.add_edge("branch", "path_b", path_filter="b")

        executor = GraphExecutor(graph)

        with executor:
            # Process frame that goes to path_a
            frame = make_frame()
            data = FlowData(frame=frame, metadata={"go_a": True})
            results = executor.process_data(data)

            assert len(results) == 1
            assert results[0].path_id == "a"

    def test_trigger_callback(self):
        """Test trigger callback is called."""
        ext = CountingModule("ext1", return_value=0.7)
        trigger = ThresholdTriggerModule(threshold=0.5, depends_on="ext1")
        path = Path(name="test", extractors=[ext], fusion=trigger)
        path_node = PathNode(path=path, run_fusion=True)

        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(path_node)
        graph.add_edge("source", "test")

        triggered = []
        executor = GraphExecutor(graph, on_trigger=lambda d: triggered.append(d))

        with executor:
            frame = make_frame()
            executor.process(frame)

        assert len(triggered) == 1
        assert triggered[0].results[0].should_trigger

    def test_not_initialized_raises(self):
        """Test that processing without initialization raises."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        executor = GraphExecutor(graph)

        with pytest.raises(RuntimeError, match="not initialized"):
            executor.process(make_frame())

    def test_batch_processing(self):
        """Test batch frame processing."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        executor = GraphExecutor(graph)

        frames = [make_frame(frame_id=i) for i in range(5)]

        with executor:
            results = executor.process_batch(frames)

        assert len(results) == 5


# =============================================================================
# FlowGraphBuilder Tests
# =============================================================================


class TestFlowGraphBuilder:
    """Tests for FlowGraphBuilder class."""

    def test_simple_build(self):
        """Test building a simple graph."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .sample(every_nth=2)
            .build())

        assert "frames" in graph.nodes
        assert "sampler_2" in graph.nodes
        assert graph.entry_node == "frames"

    def test_fluent_api(self):
        """Test fluent API chaining."""
        builder = FlowGraphBuilder()

        result = (builder
            .source("source")
            .sample(every_nth=3)
            .filter(condition=lambda d: True))

        assert result is builder

    def test_build_with_extractors(self):
        """Test building with extractors."""
        ext = CountingModule("ext1")

        graph = (FlowGraphBuilder()
            .register_extractor("ext1", ext)
            .source("frames")
            .path("test", extractors=["ext1"])
            .build())

        assert "test" in graph.nodes

    def test_build_with_branch(self):
        """Test building with branching."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .branch(
                condition=lambda d: len(d.observations) > 0,
                if_true="has_obs",
                if_false="no_obs",
            )
            .build())

        assert "branch_has_obs_no_obs" in graph.nodes

    def test_build_with_fanout(self):
        """Test building with fanout."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .fanout(paths=["a", "b", "c"])
            .build())

        assert "fanout_a_b_c" in graph.nodes

    def test_build_with_join(self):
        """Test building with join."""
        ext_a = CountingModule("ext_a")
        ext_b = CountingModule("ext_b")

        graph = (FlowGraphBuilder()
            .source("frames")
            .fanout(paths=["a", "b"])
            .path("a", extractors=[ext_a])
            .path("b", extractors=[ext_b])
            .join(["a", "b"])
            .build())

        assert "join_a_b" in graph.nodes

    def test_build_with_trigger(self):
        """Test building with trigger callback."""
        triggered = []

        graph = (FlowGraphBuilder()
            .source("frames")
            .on_trigger(lambda d: triggered.append(d))
            .build())

        # Verify callback was registered
        result = Observation(
            source="trigger",
            frame_id=1,
            t_ns=1000000,
            signals={"should_trigger": True, "trigger_score": 0.9},
        )
        data = FlowData(results=[result])
        graph.fire_triggers(data)

        assert len(triggered) == 1

    def test_edge_method(self):
        """Test adding custom edges."""
        # Create graph without auto-edges by not using add_node
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(SamplerNode("a"))
        graph.add_node(SamplerNode("b"))

        builder = FlowGraphBuilder()
        builder._graph = graph
        builder.edge("source", "a")
        builder.edge("source", "b")

        edges = graph.get_outgoing_edges("source")
        assert len(edges) == 2

    def test_rate_limit(self):
        """Test rate limit builder method."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .rate_limit(min_interval_ms=100)
            .build())

        assert "rate_limit_100ms" in graph.nodes

    def test_filter_observations(self):
        """Test observation filter builder method."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .filter_observations(min_count=2)
            .build())

        assert "observation_filter" in graph.nodes

    def test_filter_signal(self):
        """Test signal filter builder method."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .filter_signal(signal_name="score", threshold=0.5)
            .build())

        assert "signal_filter_score" in graph.nodes

    def test_cascade_fusion(self):
        """Test cascade fusion builder method."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .cascade_fusion(fusion_fn=lambda d: d)
            .build())

        # Check node exists
        assert any("cascade" in name for name in graph.nodes)

    def test_collect(self):
        """Test collector builder method."""
        graph = (FlowGraphBuilder()
            .source("frames")
            .collect(batch_size=10)
            .build())

        # Check node exists
        assert any("collector" in name for name in graph.nodes)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFlowIntegration:
    """Integration tests for the flow system."""

    def test_complete_pipeline(self):
        """Test a complete pipeline with sampling, extraction, and fusion."""
        ext = CountingModule("ext1", return_value=0.7)
        trigger = ThresholdTriggerModule(threshold=0.5, depends_on="ext1")

        triggered = []

        graph = (FlowGraphBuilder()
            .source("frames")
            .sample(every_nth=2)
            .path("main", extractors=[ext], fusion=trigger)
            .on_trigger(lambda d: triggered.append(d))
            .build())

        executor = GraphExecutor(graph)

        with executor:
            for i in range(6):
                frame = make_frame(frame_id=i)
                executor.process(frame)

        # Only every 2nd frame triggers
        assert len(triggered) == 3
        assert ext._extract_count == 3

    def test_branching_pipeline(self):
        """Test pipeline with conditional branching."""
        ext_human = CountingModule("human_ext")
        ext_scene = CountingModule("scene_ext")

        frame_count = [0]

        def has_face(data: FlowData) -> bool:
            frame_count[0] += 1
            return frame_count[0] % 2 == 0  # Every other frame "has face"

        graph = (FlowGraphBuilder()
            .source("frames")
            .branch(
                condition=has_face,
                if_true="human",
                if_false="scene",
            )
            .path("human", extractors=[ext_human])
            .path("scene", extractors=[ext_scene])
            .build())

        executor = GraphExecutor(graph)

        with executor:
            for i in range(4):
                frame = make_frame(frame_id=i)
                executor.process(frame)

        # 2 frames go to human, 2 to scene
        assert ext_human._extract_count == 2
        assert ext_scene._extract_count == 2

    def test_from_pipeline_basic(self):
        """Test FlowGraph.from_pipeline() creates correct graph."""
        ext = CountingModule("ext1", return_value=0.7)

        graph = FlowGraph.from_pipeline([ext])

        assert "source" in graph.nodes
        assert "pipeline" in graph.nodes
        assert graph.entry_node == "source"
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "source"
        assert graph.edges[0].target == "pipeline"

    def test_from_pipeline_with_fusion(self):
        """Test FlowGraph.from_pipeline() with fusion."""
        ext = CountingModule("ext1", return_value=0.7)
        trigger = ThresholdTriggerModule(threshold=0.5, depends_on="ext1")

        graph = FlowGraph.from_pipeline([ext], fusion=trigger)

        assert "source" in graph.nodes
        assert "pipeline" in graph.nodes

    def test_from_pipeline_with_on_trigger(self):
        """Test FlowGraph.from_pipeline() with on_trigger callback."""
        ext = CountingModule("ext1", return_value=0.7)
        triggered = []

        graph = FlowGraph.from_pipeline(
            [ext],
            on_trigger=lambda d: triggered.append(d),
        )

        # Fire a trigger to verify callback was registered
        result = Observation(
            source="trigger",
            frame_id=1,
            t_ns=1000000,
            signals={"should_trigger": True, "trigger_score": 0.9},
        )
        data = FlowData(results=[result])
        graph.fire_triggers(data)

        assert len(triggered) == 1

    def test_from_pipeline_multiple_extractors(self):
        """Test FlowGraph.from_pipeline() with multiple extractors."""
        ext1 = CountingModule("ext1", return_value=0.3)
        ext2 = CountingModule("ext2", return_value=0.7)

        graph = FlowGraph.from_pipeline([ext1, ext2])

        assert "source" in graph.nodes
        assert "pipeline" in graph.nodes
        assert len(graph.edges) == 1

    def test_parallel_paths_with_join(self):
        """Test parallel processing paths that join."""
        ext_a = CountingModule("ext_a", return_value=0.3)
        ext_b = CountingModule("ext_b", return_value=0.5)

        graph = (FlowGraphBuilder()
            .source("frames")
            .fanout(paths=["a", "b"])
            .path("a", extractors=[ext_a])
            .path("b", extractors=[ext_b])
            .join(["a", "b"], mode="all")
            .build())

        executor = GraphExecutor(graph)

        with executor:
            frame = make_frame()
            results = executor.process(frame)

        # Both extractors should run
        assert ext_a._extract_count == 1
        assert ext_b._extract_count == 1

        # Should have merged observations
        assert len(results) == 1
        assert len(results[0].observations) == 2


# =============================================================================
# Visualization Tests
# =============================================================================


class TestFlowGraphVisualization:
    """Tests for FlowGraph visualization methods."""

    def test_to_dot_simple(self):
        """Test DOT output for a simple pipeline."""
        ext = CountingModule("test")
        graph = FlowGraph.from_pipeline([ext])
        dot = graph.to_dot()

        assert 'digraph' in dot
        assert '"source"' in dot
        assert '"pipeline"' in dot
        assert '"source" -> "pipeline"' in dot

    def test_to_dot_with_branch(self):
        """Test DOT output with branching and path filters."""
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))
        graph.add_node(BranchNode("branch", condition=lambda d: True, if_true="yes", if_false="no"))
        graph.add_node(FilterNode("filter_a", condition=lambda d: True))
        graph.add_edge("source", "branch")
        graph.add_edge("branch", "filter_a", path_filter="yes")

        dot = graph.to_dot(title="Test")
        assert 'digraph "Test"' in dot
        assert '"yes"' in dot  # path_filter label on edge
        # BranchNode should use triangle shape
        assert 'triangle' in dot

    def test_to_dot_node_styles(self):
        """Test that different node types get different shapes."""
        graph = FlowGraph()
        graph.add_node(SourceNode("src"))
        graph.add_node(SamplerNode("samp"))
        graph.add_node(JoinNode("join", input_paths=["a"]))
        graph.add_edge("src", "samp")
        graph.add_edge("samp", "join")

        dot = graph.to_dot()
        assert 'circle' in dot    # SourceNode
        assert 'diamond' in dot   # SamplerNode
        assert 'invtriangle' in dot  # JoinNode

    def test_print_ascii(self):
        """Test ASCII output for a simple pipeline."""
        ext = CountingModule("test")
        graph = FlowGraph.from_pipeline([ext])
        ascii_repr = graph.print_ascii()

        assert "source" in ascii_repr
        assert "pipeline" in ascii_repr
        assert "SourceNode" in ascii_repr
        assert "PathNode" in ascii_repr


# =============================================================================
# Unified Modules API Tests
# =============================================================================


class AnalyzerModule(Module):
    """Test module that produces Observation."""

    depends: List[str] = []

    def __init__(self, name: str = "analyzer", value: float = 0.5):
        self._name = name
        self._value = value
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Observation:
        self._count += 1
        return Observation(
            source=self._name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value, "count": self._count},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class TriggerModuleNew(Module):
    """Test module that produces trigger Observation."""

    def __init__(self, name: str = "trigger", threshold: float = 0.3, depends_on: str = None):
        self._name = name
        self._threshold = threshold
        self._depends_on = depends_on
        self.depends = [depends_on] if depends_on else []
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trigger(self) -> bool:
        return True

    def process(self, frame, deps=None) -> Observation:
        self._count += 1
        obs = None
        if deps:
            if self._depends_on and self._depends_on in deps:
                obs = deps[self._depends_on]
            else:
                for v in deps.values():
                    if hasattr(v, 'signals'):
                        obs = v
                        break

        if not obs:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": False},
            )

        value = obs.signals.get("value", 0)
        if value > self._threshold:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": True, "trigger_score": value},
            )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"should_trigger": False},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class TestUnifiedModulesAPI:
    """Test unified modules API for PathNode and FlowGraphBuilder."""

    def test_pathnode_with_modules(self):
        """Test PathNode accepts modules parameter."""
        analyzer = AnalyzerModule("test_analyzer")
        trigger = TriggerModuleNew("test_trigger", depends_on="test_analyzer")

        node = PathNode(name="test", modules=[analyzer, trigger])

        assert node.name == "test"
        assert node.modules == (analyzer, trigger)
        assert node.path is None  # Not using legacy API

    def test_pathnode_modules_generates_modulespec(self):
        """Test PathNode with modules generates ModuleSpec."""
        from visualpath.flow.specs import ModuleSpec

        analyzer = AnalyzerModule("analyzer")
        node = PathNode(name="analysis", modules=[analyzer], parallel=True)

        spec = node.spec
        assert isinstance(spec, ModuleSpec)
        assert spec.modules == (analyzer,)
        assert spec.parallel is True

    def test_pathnode_legacy_generates_extractspec(self):
        """Test PathNode with extractors generates ExtractSpec (legacy)."""
        from visualpath.flow.specs import ExtractSpec

        ext = CountingModule("ext")
        node = PathNode(name="legacy", extractors=[ext])

        spec = node.spec
        assert isinstance(spec, ExtractSpec)
        assert spec.extractors == (ext,)

    def test_pathnode_auto_name_from_module(self):
        """Test PathNode auto-generates name from first module."""
        analyzer = AnalyzerModule("my_analyzer")
        node = PathNode(modules=[analyzer])

        assert node.name == "path_my_analyzer"

    def test_builder_path_with_modules(self):
        """Test FlowGraphBuilder.path() with modules parameter."""
        analyzer = AnalyzerModule("analyzer")
        trigger = TriggerModuleNew("trigger", depends_on="analyzer")

        graph = (FlowGraphBuilder()
            .source("frames")
            .path("analysis", modules=[analyzer, trigger])
            .build())

        assert "frames" in graph.nodes
        assert "analysis" in graph.nodes
        assert graph.nodes["analysis"].modules == (analyzer, trigger)

    def test_builder_register_and_use_module(self):
        """Test module registration and name-based reference."""
        analyzer = AnalyzerModule("analyzer")

        graph = (FlowGraphBuilder()
            .register_module("my_analyzer", analyzer)
            .source("frames")
            .path("analysis", modules=["my_analyzer"])
            .build())

        assert graph.nodes["analysis"].modules == (analyzer,)

    def test_builder_unknown_module_raises(self):
        """Test that unknown module name raises ValueError."""
        builder = FlowGraphBuilder().source("frames")

        with pytest.raises(ValueError, match="Unknown module"):
            builder.path("analysis", modules=["nonexistent"])

    def test_executor_with_modules(self):
        """Test GraphExecutor runs modules correctly."""
        analyzer = AnalyzerModule("analyzer", value=0.5)
        trigger = TriggerModuleNew("trigger", threshold=0.3, depends_on="analyzer")

        graph = (FlowGraphBuilder()
            .source("frames")
            .path("analysis", modules=[analyzer, trigger])
            .build())

        triggers_received = []
        graph.on_trigger(lambda data: triggers_received.append(data))

        executor = GraphExecutor(graph)

        frame = MockFrame(frame_id=0, t_src_ns=0, data=np.zeros((10, 10, 3)))

        with executor:
            executor.process(frame)

        assert analyzer._count == 1
        assert trigger._count == 1
        assert len(triggers_received) == 1  # value 0.5 > threshold 0.3

    def test_executor_modules_no_trigger(self):
        """Test executor when trigger condition is not met."""
        analyzer = AnalyzerModule("analyzer", value=0.1)  # Below threshold
        trigger = TriggerModuleNew("trigger", threshold=0.3, depends_on="analyzer")

        graph = (FlowGraphBuilder()
            .source("frames")
            .path("analysis", modules=[analyzer, trigger])
            .build())

        triggers_received = []
        graph.on_trigger(lambda data: triggers_received.append(data))

        executor = GraphExecutor(graph)
        frame = MockFrame(frame_id=0, t_src_ns=0, data=np.zeros((10, 10, 3)))

        with executor:
            executor.process(frame)

        assert analyzer._count == 1
        assert trigger._count == 1
        assert len(triggers_received) == 0  # No trigger

    def test_pathnode_initialize_cleanup_modules(self):
        """Test PathNode initializes and cleans up modules."""
        init_called = []
        cleanup_called = []

        class TrackingModule(Module):
            depends = []

            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            def process(self, frame, deps=None):
                return None

            def initialize(self):
                init_called.append(self._name)

            def cleanup(self):
                cleanup_called.append(self._name)

        mod1 = TrackingModule("mod1")
        mod2 = TrackingModule("mod2")
        node = PathNode(name="test", modules=[mod1, mod2])

        node.initialize()
        assert "mod1" in init_called
        assert "mod2" in init_called

        node.cleanup()
        assert "mod1" in cleanup_called
        assert "mod2" in cleanup_called

    def test_parallel_option(self):
        """Test parallel option is passed to ModuleSpec."""
        from visualpath.flow.specs import ModuleSpec

        analyzer = AnalyzerModule("analyzer")
        node = PathNode(
            name="parallel_analysis",
            modules=[analyzer],
            parallel=True,
            join_window_ns=200_000_000,
        )

        spec = node.spec
        assert isinstance(spec, ModuleSpec)
        assert spec.parallel is True
        assert spec.join_window_ns == 200_000_000
