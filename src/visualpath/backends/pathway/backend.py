"""Pathway streaming execution backend.

PathwayBackend provides a Pathway-based execution strategy for video
analysis pipelines, enabling:
- Event-time windows with watermarks
- Late arrival handling via allowed_lateness
- Built-in backpressure management
- Interval joins for multi-path synchronization

Example:
    >>> from visualpath.backends.pathway import PathwayBackend
    >>> from visualpath.flow.graph import FlowGraph
    >>>
    >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
    >>> backend = PathwayBackend()
    >>> result = backend.execute(frames, graph)
"""

import time as _time_mod
import uuid
from typing import Iterator, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend, PipelineResult
from visualpath.backends.pathway.stats import PathwayStats

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.flow.graph import FlowGraph

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


def _check_pathway() -> None:
    """Check if Pathway is available."""
    if not PATHWAY_AVAILABLE:
        raise ImportError(
            "Pathway is not installed. Install with: pip install visualpath[pathway]"
        )


class PathwayBackend(ExecutionBackend):
    """Pathway streaming execution backend.

    PathwayBackend uses Pathway's Rust-based streaming engine to process
    video frames through a FlowGraph. This provides several advantages
    over SimpleBackend:

    1. **Event-time Windows**: Uses source timestamps (t_src_ns) rather
       than processing time for accurate synchronization.

    2. **Watermarks**: Handles out-of-order frames using watermarks,
       enabling proper window semantics.

    3. **Late Arrival**: Configurable allowed_lateness lets late frames
       be processed instead of dropped.

    4. **Backpressure**: Built-in backpressure prevents queue overflow
       when extractors have varying speeds.

    5. **Interval Joins**: Temporal interval_join for synchronizing
       observations from multiple extractors.

    Configuration:
        - window_ns: Window size for joins (default: 100ms)
        - allowed_lateness_ns: Late arrival tolerance (default: 50ms)
        - autocommit_ms: Commit interval for output (default: 100ms)

    Example:
        >>> backend = PathwayBackend(
        ...     window_ns=100_000_000,  # 100ms
        ...     allowed_lateness_ns=50_000_000,  # 50ms
        ... )
        >>> graph = FlowGraph.from_pipeline([face_ext], fusion=expr_fusion)
        >>> result = backend.execute(frames, graph)
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,  # 100ms
        allowed_lateness_ns: int = 50_000_000,  # 50ms
        autocommit_ms: int = 100,
    ) -> None:
        """Initialize the PathwayBackend.

        Args:
            window_ns: Window size for temporal joins (nanoseconds).
            allowed_lateness_ns: Allowed late arrival time (nanoseconds).
            autocommit_ms: Commit interval for outputs (milliseconds).
        """
        _check_pathway()

        self._window_ns = window_ns
        self._allowed_lateness_ns = allowed_lateness_ns
        self._autocommit_ms = autocommit_ms
        self._initialized = False
        self._stats = PathwayStats()

    @property
    def name(self) -> str:
        """Backend identifier name."""
        return "pathway"

    def initialize(self) -> None:
        """Initialize backend resources."""
        _check_pathway()
        self._initialized = True

    def cleanup(self) -> None:
        """Clean up backend resources."""
        self._initialized = False

    def __enter__(self) -> "PathwayBackend":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def get_stats(self) -> dict:
        """Get pipeline execution statistics.

        Returns:
            Dictionary with current stats snapshot.
        """
        return self._stats.to_dict()

    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
    ) -> PipelineResult:
        """Execute a FlowGraph-based pipeline using Pathway streaming.

        This method:
        1. Creates an _InstrumentedConnectorSubject from the frame iterator
        2. Converts the FlowGraph to a Pathway dataflow via FlowGraphConverter
        3. Subscribes to output for trigger collection
        4. Runs the Pathway engine until all frames are processed

        Args:
            frames: Iterator of Frame objects (consumed lazily by Pathway).
            graph: FlowGraph defining the processing pipeline.

        Returns:
            PipelineResult with triggers, frame_count, and stats.
        """
        _check_pathway()

        from visualpath.backends.pathway.connector import FrameSchema
        from visualpath.backends.pathway.converter import FlowGraphConverter

        # Reset stats for this run
        self._stats.reset()
        stats = self._stats

        triggers = []
        frame_count_holder = [0]

        # Emit SessionStartRecord via ObservabilityHub
        hub = self._get_hub()
        session_id = uuid.uuid4().hex[:12]
        if hub.enabled:
            from visualpath.observability.records import SessionStartRecord
            hub.emit(SessionStartRecord(
                session_id=session_id,
                extractors=[],  # Graph-based: extractors embedded in nodes
                config={
                    "backend": "pathway",
                    "window_ns": self._window_ns,
                    "allowed_lateness_ns": self._allowed_lateness_ns,
                    "autocommit_ms": self._autocommit_ms,
                },
            ))

        # Collect fusion from graph's PathNodes for subscribe callback
        fusion = self._find_fusion(graph)

        # Initialize graph nodes (extractors, fusions)
        graph.initialize()

        try:
            # 1. Create frame source via instrumented ConnectorSubject
            subject = _InstrumentedConnectorSubject(frames, stats, frame_count_holder)
            frames_table = pw.io.python.read(
                subject,
                schema=FrameSchema,
                autocommit_duration_ms=self._autocommit_ms,
            )

            # 2. Convert FlowGraph to Pathway dataflow
            converter = FlowGraphConverter(
                window_ns=self._window_ns,
                allowed_lateness_ns=self._allowed_lateness_ns,
            )
            output_table = converter.convert(graph, frames_table)

            # 3. Subscribe — collect triggers from output
            def on_output(key, row, time, is_addition):
                if not is_addition:
                    return
                stats.record_observation_output()

                t0 = _time_mod.perf_counter()

                # Extract results from output row if available
                results_wrapper = row.get("results")
                if results_wrapper is not None:
                    result_list = (
                        results_wrapper.value
                        if hasattr(results_wrapper, "value")
                        else results_wrapper
                    )
                    if isinstance(result_list, list):
                        frame_id_val = 0
                        for r in result_list:
                            # Record extraction stats
                            if hasattr(r, "observation") and r.observation is not None:
                                stats.record_frame_extracted()
                                elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
                                stats.record_extraction(
                                    r.source, elapsed_ms, success=True,
                                )
                                frame_id_val = r.frame_id
                                # Run fusion on observation if available
                                if fusion is not None:
                                    fusion_result = fusion.update(r.observation)
                                    if fusion_result.should_trigger and fusion_result.trigger is not None:
                                        stats.record_trigger()
                                        triggers.append(fusion_result.trigger)
                                        # Fire graph trigger callbacks
                                        from visualpath.flow.node import FlowData
                                        trigger_data = FlowData(results=[fusion_result])
                                        graph.fire_triggers(trigger_data)
                            elif hasattr(r, "source"):
                                # Extraction failed (no observation)
                                stats.record_extraction(
                                    r.source, 0.0, success=False,
                                )

                            # Also check for direct trigger results
                            if hasattr(r, "should_trigger") and r.should_trigger:
                                if hasattr(r, "trigger") and r.trigger is not None:
                                    stats.record_trigger()
                                    triggers.append(r.trigger)

                        # Emit timing record via ObservabilityHub
                        if hub.enabled:
                            from visualpath.observability.records import TimingRecord
                            elapsed_ms = (_time_mod.perf_counter() - t0) * 1000
                            hub.emit(TimingRecord(
                                frame_id=frame_id_val,
                                component="pathway_udf",
                                processing_ms=elapsed_ms,
                            ))

            pw.io.subscribe(output_table, on_change=on_output)

            # 4. Run the Pathway engine
            stats.mark_pipeline_start()
            pw.run()
            stats.mark_pipeline_end()

        finally:
            graph.cleanup()

        # Emit SessionEndRecord
        if hub.enabled:
            from visualpath.observability.records import SessionEndRecord
            hub.emit(SessionEndRecord(
                session_id=session_id,
                duration_sec=stats.pipeline_duration_sec,
                total_frames=frame_count_holder[0],
                total_triggers=stats.triggers_fired,
                avg_fps=stats.throughput_fps,
            ))

        return PipelineResult(
            triggers=triggers,
            frame_count=frame_count_holder[0],
            stats=stats.to_dict(),
        )

    @staticmethod
    def _find_fusion(graph: "FlowGraph"):
        """Find fusion module from graph nodes via spec.

        Returns the first fusion found, or None.
        """
        from visualpath.flow.specs import ExtractSpec

        for node in graph.nodes.values():
            spec = node.spec
            if isinstance(spec, ExtractSpec) and spec.fusion is not None:
                return spec.fusion
        return None

    @staticmethod
    def _get_hub():
        """Get the ObservabilityHub singleton."""
        from visualpath.observability import ObservabilityHub
        return ObservabilityHub.get_instance()


if PATHWAY_AVAILABLE:
    class _InstrumentedConnectorSubject(pw.io.python.ConnectorSubject):
        """ConnectorSubject that counts ingested frames."""

        def __init__(self, frames: Iterator, stats: PathwayStats, frame_count_holder: list) -> None:
            super().__init__()
            self._frames = frames
            self._stats = stats
            self._frame_count = frame_count_holder

        def run(self) -> None:
            for frame in self._frames:
                self._frame_count[0] += 1
                self._stats.record_ingestion()
                self.next(
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    frame=pw.PyObjectWrapper(frame),
                )
            self.close()


__all__ = ["PathwayBackend", "PATHWAY_AVAILABLE"]
