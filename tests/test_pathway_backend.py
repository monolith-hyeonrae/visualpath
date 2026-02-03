"""Tests for Pathway backend integration.

These tests verify the Pathway execution backend functionality.
Tests are skipped if Pathway is not installed.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from unittest.mock import MagicMock, patch

from visualpath.core import BaseExtractor, Observation, BaseFusion, FusionResult
from visualpath.backends.base import ExecutionBackend
from visualpath.backends.simple import SimpleBackend


# Check if Pathway is available
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


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

    def extract(self, frame, deps=None) -> Optional[Observation]:
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
        self._trigger_count = 0

    def update(self, observation: Observation) -> FusionResult:
        self._update_count += 1
        value = observation.signals.get("value", 0)
        if value > self._threshold:
            self._trigger_count += 1
            # Create a mock trigger
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=observation.t_ns,
                pre_sec=2.0,
                post_sec=2.0,
                label="threshold_exceeded",
                score=value,
            )
            return FusionResult(
                should_trigger=True,
                trigger=trigger,
                score=value,
                reason="threshold_exceeded",
                observations_used=1,
            )
        return FusionResult(should_trigger=False)

    def reset(self) -> None:
        self._update_count = 0
        self._trigger_count = 0

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


def make_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    """Create a list of mock frames."""
    return [
        make_frame(frame_id=i, t_ns=i * interval_ns)
        for i in range(count)
    ]


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
        backend = SimpleBackend()
        assert isinstance(backend, ExecutionBackend)
        assert hasattr(backend, "run")
        assert hasattr(backend, "run_graph")
        assert hasattr(backend, "name")


# =============================================================================
# SimpleBackend Tests
# =============================================================================


class TestSimpleBackend:
    """Tests for SimpleBackend."""

    def test_run_single_extractor(self):
        """Test running with a single extractor."""
        backend = SimpleBackend()
        extractor = CountingExtractor("test", return_value=0.3)
        frames = make_frames(5)

        triggers = backend.run(iter(frames), [extractor])

        assert extractor._extract_count == 5
        assert extractor._initialized
        assert extractor._cleaned_up
        assert len(triggers) == 0  # No fusion

    def test_run_with_fusion(self):
        """Test running with fusion that triggers."""
        backend = SimpleBackend()
        extractor = CountingExtractor("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(5)

        triggers = backend.run(iter(frames), [extractor], fusion)

        assert extractor._extract_count == 5
        assert fusion._update_count == 5
        assert len(triggers) == 5  # All frames trigger

    def test_run_with_callback(self):
        """Test running with trigger callback."""
        backend = SimpleBackend()
        extractor = CountingExtractor("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        callback_triggers = []
        triggers = backend.run(
            iter(frames),
            [extractor],
            fusion,
            on_trigger=lambda t: callback_triggers.append(t),
        )

        assert len(callback_triggers) == 3
        assert len(triggers) == 3

    def test_run_multiple_extractors(self):
        """Test running with multiple extractors."""
        backend = SimpleBackend()
        ext1 = CountingExtractor("ext1", return_value=0.3)
        ext2 = CountingExtractor("ext2", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        triggers = backend.run(iter(frames), [ext1, ext2], fusion)

        assert ext1._extract_count == 3
        assert ext2._extract_count == 3
        # Fusion receives observations from both extractors
        assert fusion._update_count == 6

    def test_run_graph(self):
        """Test run_graph with simple graph."""
        from visualpath.flow import FlowGraph, SourceNode

        backend = SimpleBackend()
        graph = FlowGraph()
        graph.add_node(SourceNode("source"))

        frames = make_frames(3)
        results = backend.run_graph(iter(frames), graph)

        assert len(results) == 3


# =============================================================================
# PathwayBackend Tests (require Pathway)
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayBackend:
    """Tests for PathwayBackend."""

    def test_import(self):
        """Test PathwayBackend can be imported."""
        from visualpath.backends.pathway import PathwayBackend
        assert PathwayBackend is not None

    def test_instantiation(self):
        """Test PathwayBackend can be instantiated."""
        from visualpath.backends.pathway import PathwayBackend
        backend = PathwayBackend()
        assert backend.name == "pathway"

    def test_configuration(self):
        """Test PathwayBackend configuration options."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(
            window_ns=200_000_000,
            allowed_lateness_ns=100_000_000,
            autocommit_ms=50,
        )

        assert backend._window_ns == 200_000_000
        assert backend._allowed_lateness_ns == 100_000_000
        assert backend._autocommit_ms == 50

    def test_implements_interface(self):
        """Test PathwayBackend implements ExecutionBackend."""
        from visualpath.backends.pathway import PathwayBackend
        backend = PathwayBackend()
        assert isinstance(backend, ExecutionBackend)


# =============================================================================
# Pathway Schema Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestSchemas:
    """Tests for Pathway schemas using PyObjectWrapper."""

    def test_frame_schema(self):
        """Test FrameSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import FrameSchema

        columns = FrameSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "frame" in columns

    def test_observation_schema(self):
        """Test ObservationSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import ObservationSchema

        columns = ObservationSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "source" in columns
        assert "observation" in columns

    def test_trigger_schema(self):
        """Test TriggerSchema uses PyObjectWrapper."""
        from visualpath.backends.pathway.connector import TriggerSchema

        columns = TriggerSchema.column_names()
        assert "frame_id" in columns
        assert "t_ns" in columns
        assert "trigger" in columns


# =============================================================================
# VideoConnectorSubject Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestVideoConnectorSubject:
    """Tests for VideoConnectorSubject."""

    def test_import(self):
        """Test VideoConnectorSubject can be imported."""
        from visualpath.backends.pathway.connector import VideoConnectorSubject
        assert VideoConnectorSubject is not None

    def test_instantiation(self):
        """Test VideoConnectorSubject can be instantiated."""
        from visualpath.backends.pathway.connector import VideoConnectorSubject

        frames = make_frames(3)
        subject = VideoConnectorSubject(iter(frames))
        assert subject is not None


# =============================================================================
# Pathway End-to-End Execution Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayExecution:
    """Tests that verify actual Pathway engine execution."""

    def test_run_single_extractor(self):
        """Test PathwayBackend.run() with a single extractor through Pathway engine."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.3)
        frames = make_frames(5)

        triggers = backend.run(iter(frames), [extractor])

        # No fusion = no triggers
        assert len(triggers) == 0
        # Extractor should have been called for each frame
        assert extractor._extract_count == 5
        assert extractor._initialized
        assert extractor._cleaned_up

    def test_run_with_fusion_triggers(self):
        """Test PathwayBackend.run() with fusion that fires triggers."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(5)

        triggers = backend.run(iter(frames), [extractor], fusion)

        # All frames should trigger (value 0.7 > threshold 0.5)
        assert len(triggers) == 5
        assert extractor._extract_count == 5
        assert fusion._update_count == 5

    def test_run_with_fusion_no_trigger(self):
        """Test PathwayBackend.run() when fusion doesn't fire."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.3)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        triggers = backend.run(iter(frames), [extractor], fusion)

        # value 0.3 < threshold 0.5 → no triggers
        assert len(triggers) == 0
        assert extractor._extract_count == 3
        assert fusion._update_count == 3

    def test_run_multiple_extractors(self):
        """Test PathwayBackend.run() with multiple extractors."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        ext1 = CountingExtractor("ext1", return_value=0.3)
        ext2 = CountingExtractor("ext2", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        triggers = backend.run(iter(frames), [ext1, ext2], fusion)

        assert ext1._extract_count == 3
        assert ext2._extract_count == 3
        # Fusion receives observations from both extractors
        assert fusion._update_count == 6
        # Only ext2's observations (0.7) trigger, not ext1's (0.3)
        assert len(triggers) == 3

    def test_run_with_callback(self):
        """Test PathwayBackend.run() with on_trigger callback."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        callback_triggers = []
        triggers = backend.run(
            iter(frames),
            [extractor],
            fusion,
            on_trigger=lambda t: callback_triggers.append(t),
        )

        assert len(callback_triggers) == 3
        assert len(triggers) == 3

    def test_run_initializes_and_cleans_up(self):
        """Test that run() properly initializes and cleans up extractors."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)
        frames = make_frames(2)

        backend.run(iter(frames), [ext])

        assert ext._initialized
        assert ext._cleaned_up

    def test_run_cleanup_on_error(self):
        """Test that cleanup happens even if extraction errors."""
        from visualpath.backends.pathway import PathwayBackend

        class ErrorExtractor(BaseExtractor):
            _name = "error"
            _initialized = False
            _cleaned_up = False

            @property
            def name(self):
                return self._name

            def extract(self, frame, deps=None):
                raise RuntimeError("intentional")

            def initialize(self):
                self._initialized = True

            def cleanup(self):
                self._cleaned_up = True

        backend = PathwayBackend(autocommit_ms=10)
        ext = ErrorExtractor()
        frames = make_frames(2)

        # Should not raise - errors are caught inside UDF
        triggers = backend.run(iter(frames), [ext])
        assert ext._initialized
        assert ext._cleaned_up
        assert len(triggers) == 0


# =============================================================================
# Pathway vs Simple Backend Comparison
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestBackendComparison:
    """Tests comparing SimpleBackend and PathwayBackend produce same results."""

    def test_same_trigger_count(self):
        """Test both backends produce the same number of triggers."""
        from visualpath.backends.pathway import PathwayBackend

        # Simple backend
        simple = SimpleBackend()
        ext_s = CountingExtractor("test", return_value=0.7)
        fusion_s = ThresholdFusion(threshold=0.5)
        simple_triggers = simple.run(iter(make_frames(5)), [ext_s], fusion_s)

        # Pathway backend
        pathway = PathwayBackend(autocommit_ms=10)
        ext_p = CountingExtractor("test", return_value=0.7)
        fusion_p = ThresholdFusion(threshold=0.5)
        pathway_triggers = pathway.run(iter(make_frames(5)), [ext_p], fusion_p)

        assert len(simple_triggers) == len(pathway_triggers)

    def test_same_extract_count(self):
        """Test both backends call extractors the same number of times."""
        from visualpath.backends.pathway import PathwayBackend

        ext_s = CountingExtractor("test", return_value=0.5)
        SimpleBackend().run(iter(make_frames(10)), [ext_s])

        ext_p = CountingExtractor("test", return_value=0.5)
        PathwayBackend(autocommit_ms=10).run(iter(make_frames(10)), [ext_p])

        assert ext_s._extract_count == ext_p._extract_count

    def test_no_trigger_consistency(self):
        """Test both backends agree on no-trigger case."""
        from visualpath.backends.pathway import PathwayBackend

        ext_s = CountingExtractor("test", return_value=0.2)
        fusion_s = ThresholdFusion(threshold=0.5)
        simple_triggers = SimpleBackend().run(
            iter(make_frames(5)), [ext_s], fusion_s
        )

        ext_p = CountingExtractor("test", return_value=0.2)
        fusion_p = ThresholdFusion(threshold=0.5)
        pathway_triggers = PathwayBackend(autocommit_ms=10).run(
            iter(make_frames(5)), [ext_p], fusion_p
        )

        assert len(simple_triggers) == 0
        assert len(pathway_triggers) == 0


# =============================================================================
# Pathway Operator Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayOperators:
    """Tests for Pathway operators."""

    def test_create_extractor_udf(self):
        """Test creating extractor UDF."""
        from visualpath.backends.pathway.operators import create_extractor_udf

        extractor = CountingExtractor("test", return_value=0.5)
        udf = create_extractor_udf(extractor)

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 1
        assert results[0].source == "test"
        assert results[0].observation is not None

    def test_create_multi_extractor_udf(self):
        """Test creating multi-extractor UDF."""
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        ext1 = CountingExtractor("ext1", return_value=0.3)
        ext2 = CountingExtractor("ext2", return_value=0.7)
        udf = create_multi_extractor_udf([ext1, ext2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2
        sources = {r.source for r in results}
        assert sources == {"ext1", "ext2"}

    def test_create_fusion_state_handler(self):
        """Test creating fusion state handler."""
        from visualpath.backends.pathway.operators import create_fusion_state_handler

        fusion = ThresholdFusion(threshold=0.5)
        init_fn, update_fn = create_fusion_state_handler(fusion)

        # Initialize state
        state = init_fn()
        assert state.fusion is fusion
        assert len(state.last_observations) == 0

        # Create observation above threshold
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1_000_000,
            signals={"value": 0.7},
        )

        # Update state
        new_state, trigger = update_fn(state, obs)
        assert trigger is not None
        assert new_state.last_observations["test"] == obs

    def test_apply_extractors_pathway_table(self):
        """Test apply_extractors creates a valid Pathway table pipeline."""
        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject,
            FrameSchema,
        )
        from visualpath.backends.pathway.operators import apply_extractors

        ext1 = CountingExtractor("ext1", return_value=0.5)
        ext2 = CountingExtractor("ext2", return_value=0.8)
        frames = make_frames(3)

        subject = VideoConnectorSubject(iter(frames))
        frames_table = pw.io.python.read(
            subject, schema=FrameSchema, autocommit_duration_ms=10,
        )

        obs_table = apply_extractors(frames_table, [ext1, ext2])

        collected = []
        pw.io.subscribe(
            obs_table,
            on_change=lambda key, row, time, is_addition: (
                collected.append(row) if is_addition else None
            ),
        )
        pw.run()

        assert len(collected) == 3
        assert ext1._extract_count == 3
        assert ext2._extract_count == 3


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestFlowGraphConverter:
    """Tests for FlowGraphConverter."""

    def test_import(self):
        """Test FlowGraphConverter can be imported."""
        from visualpath.backends.pathway.converter import FlowGraphConverter
        assert FlowGraphConverter is not None

    def test_instantiation(self):
        """Test FlowGraphConverter can be instantiated."""
        from visualpath.backends.pathway.converter import FlowGraphConverter

        converter = FlowGraphConverter(
            window_ns=100_000_000,
            allowed_lateness_ns=50_000_000,
        )
        assert converter._window_ns == 100_000_000
        assert converter._allowed_lateness_ns == 50_000_000


# =============================================================================
# API Integration Tests
# =============================================================================


class TestAPIBackendParameter:
    """Tests for backend parameter in api.py."""

    def test_get_backend_simple(self):
        """Test _get_backend returns SimpleBackend for 'simple'."""
        from visualpath.api import _get_backend

        backend = _get_backend("simple")
        assert isinstance(backend, SimpleBackend)

    def test_get_backend_unknown(self):
        """Test _get_backend raises for unknown backend."""
        from visualpath.api import _get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            _get_backend("unknown")

    @pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
    def test_get_backend_pathway(self):
        """Test _get_backend returns PathwayBackend for 'pathway'."""
        from visualpath.api import _get_backend
        from visualpath.backends.pathway import PathwayBackend

        backend = _get_backend("pathway")
        assert isinstance(backend, PathwayBackend)


# =============================================================================
# Deps Support Tests
# =============================================================================


class UpstreamExtractor(BaseExtractor):
    """Extractor that produces observations used by dependent extractors."""

    def __init__(self, name: str = "upstream"):
        self._name = name
        self._extract_count = 0

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame, deps=None) -> Optional[Observation]:
        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"upstream_value": 42, "call_count": self._extract_count},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class DependentExtractor(BaseExtractor):
    """Extractor that depends on upstream and records received deps."""

    depends = ["upstream"]

    def __init__(self, name: str = "dependent"):
        self._name = name
        self._extract_count = 0
        self.received_deps: List[Optional[Dict]] = []

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame, deps=None) -> Optional[Observation]:
        self._extract_count += 1
        self.received_deps.append(deps)
        upstream_value = None
        if deps and "upstream" in deps:
            upstream_value = deps["upstream"].signals.get("upstream_value")
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "received_upstream": upstream_value is not None,
                "upstream_value": upstream_value,
            },
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class TestMultiExtractorUDFWithDeps:
    """Tests for create_multi_extractor_udf with deps accumulation."""

    def test_deps_passed_to_dependent_extractor(self):
        """Test that UDF passes deps from upstream to dependent extractor."""
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        upstream = UpstreamExtractor()
        dependent = DependentExtractor()
        udf = create_multi_extractor_udf([upstream, dependent])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2
        # Dependent should have received upstream's observation
        assert dependent.received_deps[-1] is not None
        assert "upstream" in dependent.received_deps[-1]
        assert dependent.received_deps[-1]["upstream"].signals["upstream_value"] == 42

    def test_deps_not_passed_when_no_depends(self):
        """Test that extractors without depends don't get deps."""
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        ext1 = CountingExtractor("ext1")
        ext2 = CountingExtractor("ext2")
        udf = create_multi_extractor_udf([ext1, ext2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 2

    def test_deps_accumulate_across_chain(self):
        """Test deps accumulate for multi-level dependency chains."""
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        class Level2Extractor(BaseExtractor):
            depends = ["dependent"]

            def __init__(self):
                self.received_deps = []

            @property
            def name(self):
                return "level2"

            def extract(self, frame, deps=None):
                self.received_deps.append(deps)
                has_dep = deps is not None and "dependent" in deps
                return Observation(
                    source="level2",
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    signals={"has_dependent_dep": has_dep},
                )

            def initialize(self):
                pass

            def cleanup(self):
                pass

        upstream = UpstreamExtractor()
        dependent = DependentExtractor()
        level2 = Level2Extractor()
        udf = create_multi_extractor_udf([upstream, dependent, level2])

        frame = make_frame()
        results = udf(frame)

        assert len(results) == 3
        # Level2 should have received dependent's observation
        assert level2.received_deps[-1] is not None
        assert "dependent" in level2.received_deps[-1]


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayDepsExecution:
    """Tests for dependency resolution through Pathway engine."""

    def test_deps_through_pathway(self):
        """Test deps work when running through actual Pathway engine."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        upstream = UpstreamExtractor()
        dependent = DependentExtractor()
        frames = make_frames(3)

        backend.run(iter(frames), [upstream, dependent])

        assert upstream._extract_count == 3
        assert dependent._extract_count == 3
        # All calls should have received deps
        for dep in dependent.received_deps:
            assert dep is not None
            assert "upstream" in dep


class TestSequentialExecutorWithDeps:
    """Tests for SequentialExecutor with deps accumulation."""

    def test_deps_accumulation(self):
        """Test that SequentialExecutor accumulates deps."""
        from visualpath.backends.simple.executor import SequentialExecutor

        executor = SequentialExecutor()
        upstream = UpstreamExtractor()
        dependent = DependentExtractor()

        frame = make_frame()
        results = executor.execute(frame, [upstream, dependent])

        assert len(results) == 2
        assert all(r.success for r in results)
        # Dependent should have received upstream's observation
        dep_obs = results[1].observation
        assert dep_obs.signals["received_upstream"] is True
        assert dep_obs.signals["upstream_value"] == 42


class TestThreadPoolExecutorWithDeps:
    """Tests for ThreadPoolExecutor with deps layer separation."""

    def test_layer_separation(self):
        """Test that dependent extractors run after their deps."""
        from visualpath.backends.simple.executor import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)
        upstream = UpstreamExtractor()
        dependent = DependentExtractor()

        frame = make_frame()
        results = executor.execute(frame, [upstream, dependent])

        assert len(results) == 2
        # Find the dependent result
        dep_result = next(r for r in results if r.extractor_name == "dependent")
        assert dep_result.success
        assert dep_result.observation.signals["received_upstream"] is True
        assert dep_result.observation.signals["upstream_value"] == 42

        executor.shutdown()

    def test_no_deps_runs_parallel(self):
        """Test that extractors without deps run in parallel."""
        from visualpath.backends.simple.executor import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=4)
        ext1 = CountingExtractor("ext1")
        ext2 = CountingExtractor("ext2")

        frame = make_frame()
        results = executor.execute(frame, [ext1, ext2])

        assert len(results) == 2
        assert all(r.success for r in results)

        executor.shutdown()


class TestVenvWorkerDepsSerialization:
    """Tests for VenvWorker deps serialization/deserialization roundtrip."""

    def test_serialize_observation_for_deps(self):
        """Test observation serialization for deps."""
        from visualpath.process.launcher import _serialize_observation_for_deps

        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000000,
            signals={"value": 42, "name": "test"},
            metadata={"key": "val"},
        )

        serialized = _serialize_observation_for_deps(obs)

        assert serialized is not None
        assert serialized["source"] == "test"
        assert serialized["frame_id"] == 1
        assert serialized["t_ns"] == 1000000
        assert serialized["signals"]["value"] == 42

    def test_serialize_none_observation(self):
        """Test serializing None observation."""
        from visualpath.process.launcher import _serialize_observation_for_deps

        assert _serialize_observation_for_deps(None) is None

    def test_deserialize_observation_in_worker(self):
        """Test observation deserialization in worker."""
        from visualpath.process.worker import _deserialize_observation_in_worker

        data = {
            "source": "test",
            "frame_id": 1,
            "t_ns": 1000000,
            "signals": {"value": 42},
            "metadata": {"key": "val"},
            "timing": None,
        }

        obs = _deserialize_observation_in_worker(data)

        assert obs.source == "test"
        assert obs.frame_id == 1
        assert obs.t_ns == 1000000
        assert obs.signals["value"] == 42

    def test_roundtrip_serialization(self):
        """Test serialize -> deserialize roundtrip preserves data."""
        from visualpath.process.launcher import _serialize_observation_for_deps
        from visualpath.process.worker import _deserialize_observation_in_worker

        original = Observation(
            source="face_detect",
            frame_id=5,
            t_ns=5000000,
            signals={"face_count": 2, "confidence": 0.95},
            metadata={"backend": "insightface"},
        )

        serialized = _serialize_observation_for_deps(original)
        restored = _deserialize_observation_in_worker(serialized)

        assert restored.source == original.source
        assert restored.frame_id == original.frame_id
        assert restored.t_ns == original.t_ns
        assert restored.signals == original.signals
        assert restored.metadata == original.metadata


# =============================================================================
# PathwayStats Unit Tests
# =============================================================================


class TestPathwayStats:
    """Tests for PathwayStats dataclass."""

    def test_initial_values(self):
        """Test PathwayStats starts with zero counters."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.frames_ingested == 0
        assert stats.frames_extracted == 0
        assert stats.extractions_completed == 0
        assert stats.extractions_failed == 0
        assert stats.triggers_fired == 0
        assert stats.observations_output == 0
        assert stats.total_extraction_ms == 0.0
        assert stats.per_extractor_time_ms == {}
        assert stats.pipeline_start_ns == 0
        assert stats.pipeline_end_ns == 0

    def test_record_ingestion(self):
        """Test recording frame ingestion."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_ingestion()
        stats.record_ingestion()
        assert stats.frames_ingested == 3

    def test_record_extraction_success(self):
        """Test recording successful extractions."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_extraction("face", 10.0, success=True)
        stats.record_extraction("pose", 20.0, success=True)
        assert stats.extractions_completed == 2
        assert stats.extractions_failed == 0
        assert stats.total_extraction_ms == 30.0

    def test_record_extraction_failure(self):
        """Test recording failed extractions."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_extraction("face", 5.0, success=False)
        assert stats.extractions_completed == 0
        assert stats.extractions_failed == 1
        assert stats.total_extraction_ms == 5.0

    def test_per_extractor_ema(self):
        """Test EMA calculation for per-extractor times."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        # First call sets the initial value
        stats.record_extraction("face", 10.0)
        assert stats.per_extractor_time_ms["face"] == 10.0

        # Second call applies EMA: 0.3 * 20 + 0.7 * 10 = 13.0
        stats.record_extraction("face", 20.0)
        assert abs(stats.per_extractor_time_ms["face"] - 13.0) < 0.01

    def test_record_trigger(self):
        """Test recording triggers."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_trigger()
        stats.record_trigger()
        assert stats.triggers_fired == 2

    def test_record_observation_output(self):
        """Test recording observation output."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_observation_output()
        assert stats.observations_output == 1

    def test_record_frame_extracted(self):
        """Test recording frame extraction completion."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_frame_extracted()
        stats.record_frame_extracted()
        assert stats.frames_extracted == 2

    def test_avg_extraction_ms(self):
        """Test average extraction time."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_extraction("a", 10.0)
        stats.record_extraction("b", 20.0)
        assert abs(stats.avg_extraction_ms - 15.0) < 0.01

    def test_avg_extraction_ms_empty(self):
        """Test average extraction time when empty."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.avg_extraction_ms == 0.0

    def test_p95_extraction_ms(self):
        """Test p95 extraction time."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        # Add 20 values: 1..20
        for i in range(1, 21):
            stats.record_extraction("ext", float(i))
        # p95 of 1..20 → index 19 (0-based), value 19 or 20
        p95 = stats.p95_extraction_ms
        assert p95 >= 19.0

    def test_p95_empty(self):
        """Test p95 when no data."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.p95_extraction_ms == 0.0

    def test_pipeline_duration(self):
        """Test pipeline duration calculation."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.mark_pipeline_start()
        # Simulate some time passing
        import time
        time.sleep(0.01)
        stats.mark_pipeline_end()
        assert stats.pipeline_duration_sec > 0.0

    def test_pipeline_duration_not_started(self):
        """Test pipeline duration when not started."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.pipeline_duration_sec == 0.0

    def test_throughput_fps(self):
        """Test throughput FPS calculation."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.mark_pipeline_start()
        for _ in range(10):
            stats.record_frame_extracted()
        import time
        time.sleep(0.01)
        stats.mark_pipeline_end()
        assert stats.throughput_fps > 0.0

    def test_throughput_fps_no_duration(self):
        """Test throughput when no duration recorded."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        assert stats.throughput_fps == 0.0

    def test_to_dict(self):
        """Test to_dict contains all expected keys."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_extraction("face", 10.0)
        stats.record_trigger()

        d = stats.to_dict()
        assert d["frames_ingested"] == 1
        assert d["extractions_completed"] == 1
        assert d["triggers_fired"] == 1
        assert d["total_extraction_ms"] == 10.0
        assert "face" in d["per_extractor_time_ms"]
        assert "throughput_fps" in d
        assert "avg_extraction_ms" in d
        assert "p95_extraction_ms" in d
        assert "pipeline_duration_sec" in d

    def test_reset(self):
        """Test that reset clears all counters."""
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        stats.record_ingestion()
        stats.record_extraction("face", 10.0)
        stats.record_trigger()
        stats.record_observation_output()
        stats.record_frame_extracted()
        stats.mark_pipeline_start()
        stats.mark_pipeline_end()

        stats.reset()

        assert stats.frames_ingested == 0
        assert stats.frames_extracted == 0
        assert stats.extractions_completed == 0
        assert stats.extractions_failed == 0
        assert stats.triggers_fired == 0
        assert stats.observations_output == 0
        assert stats.total_extraction_ms == 0.0
        assert stats.per_extractor_time_ms == {}
        assert stats.pipeline_start_ns == 0
        assert stats.pipeline_end_ns == 0

    def test_thread_safety(self):
        """Test that concurrent access does not corrupt counters."""
        import threading
        from visualpath.backends.pathway.stats import PathwayStats

        stats = PathwayStats()
        n_threads = 4
        n_ops = 1000

        def worker():
            for _ in range(n_ops):
                stats.record_ingestion()
                stats.record_extraction("ext", 1.0)
                stats.record_frame_extracted()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.frames_ingested == n_threads * n_ops
        assert stats.extractions_completed == n_threads * n_ops
        assert stats.frames_extracted == n_threads * n_ops


# =============================================================================
# PathwayBackend Stats Integration Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayBackendStats:
    """Tests for PathwayBackend.get_stats() integration."""

    def test_get_stats_initial(self):
        """Test get_stats returns zeroed stats before run."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend()
        s = backend.get_stats()
        assert s["frames_ingested"] == 0
        assert s["frames_extracted"] == 0

    def test_get_stats_after_run(self):
        """Test get_stats after running pipeline."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.5)
        frames = make_frames(5)

        backend.run(iter(frames), [extractor])

        s = backend.get_stats()
        assert s["frames_ingested"] == 5
        assert s["frames_extracted"] == 5
        assert s["extractions_completed"] == 5
        assert s["extractions_failed"] == 0
        assert s["observations_output"] == 5
        assert s["pipeline_duration_sec"] > 0
        assert s["throughput_fps"] > 0

    def test_get_stats_with_fusion_triggers(self):
        """Test get_stats tracks triggers."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        extractor = CountingExtractor("test", return_value=0.7)
        fusion = ThresholdFusion(threshold=0.5)
        frames = make_frames(3)

        backend.run(iter(frames), [extractor], fusion)

        s = backend.get_stats()
        assert s["triggers_fired"] == 3

    def test_get_stats_per_extractor_time(self):
        """Test per-extractor time tracking."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        ext1 = CountingExtractor("ext1", return_value=0.3)
        ext2 = CountingExtractor("ext2", return_value=0.7)
        frames = make_frames(3)

        backend.run(iter(frames), [ext1, ext2])

        s = backend.get_stats()
        assert "ext1" in s["per_extractor_time_ms"]
        assert "ext2" in s["per_extractor_time_ms"]

    def test_stats_reset_between_runs(self):
        """Test that stats reset between consecutive runs."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)

        # First run
        backend.run(iter(make_frames(5)), [ext])
        s1 = backend.get_stats()
        assert s1["frames_ingested"] == 5

        # Second run - stats should be fresh
        ext2 = CountingExtractor("test", return_value=0.5)
        backend.run(iter(make_frames(3)), [ext2])
        s2 = backend.get_stats()
        assert s2["frames_ingested"] == 3

    def test_get_stats_timing_fields(self):
        """Test timing-related stats fields."""
        from visualpath.backends.pathway import PathwayBackend

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)
        frames = make_frames(5)

        backend.run(iter(frames), [ext])

        s = backend.get_stats()
        assert s["total_extraction_ms"] > 0
        assert s["avg_extraction_ms"] > 0
        assert s["p95_extraction_ms"] >= 0


# =============================================================================
# PathwayBackend ObservabilityHub Tests
# =============================================================================


@pytest.mark.skipif(not PATHWAY_AVAILABLE, reason="Pathway not installed")
class TestPathwayObservabilityHub:
    """Tests for ObservabilityHub integration in PathwayBackend."""

    def setup_method(self):
        """Reset the ObservabilityHub before each test."""
        from visualpath.observability import ObservabilityHub
        ObservabilityHub.reset_instance()

    def teardown_method(self):
        """Reset the ObservabilityHub after each test."""
        from visualpath.observability import ObservabilityHub
        ObservabilityHub.reset_instance()

    def test_session_records_emitted(self):
        """Test SessionStartRecord and SessionEndRecord are emitted."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.MINIMAL, sinks=[sink])

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)
        backend.run(iter(make_frames(3)), [ext])

        records = sink.get_records()
        types = [r.record_type for r in records]
        assert "session_start" in types
        assert "session_end" in types

        # Verify session_start content
        start_rec = next(r for r in records if r.record_type == "session_start")
        assert start_rec.session_id != ""
        assert "test" in start_rec.extractors
        assert start_rec.config["backend"] == "pathway"

        # Verify session_end content
        end_rec = next(r for r in records if r.record_type == "session_end")
        assert end_rec.session_id == start_rec.session_id
        assert end_rec.total_frames == 3
        assert end_rec.duration_sec > 0

    def test_timing_records_emitted_at_normal(self):
        """Test TimingRecords are emitted at NORMAL level."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.NORMAL, sinks=[sink])

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)
        backend.run(iter(make_frames(3)), [ext])

        records = sink.get_records()
        types = [r.record_type for r in records]
        assert "timing" in types

        timing_recs = [r for r in records if r.record_type == "timing"]
        assert len(timing_recs) == 3  # One per frame
        assert all(r.component == "pathway_udf" for r in timing_recs)

    def test_no_records_when_off(self):
        """Test no records emitted when hub is OFF."""
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        from visualpath.backends.pathway import PathwayBackend

        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.OFF)
        hub.add_sink(sink)

        backend = PathwayBackend(autocommit_ms=10)
        ext = CountingExtractor("test", return_value=0.5)
        backend.run(iter(make_frames(3)), [ext])

        assert len(sink) == 0
