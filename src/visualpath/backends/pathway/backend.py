"""Pathway streaming execution backend.

PathwayBackend provides a Pathway-based execution strategy for video
analysis pipelines, enabling:
- Event-time windows with watermarks
- Late arrival handling via allowed_lateness
- Built-in backpressure management
- Interval joins for multi-path synchronization

Example:
    >>> from visualpath.backends.pathway import PathwayBackend
    >>>
    >>> backend = PathwayBackend()
    >>> triggers = backend.run(frames, extractors=[face_ext])
"""

import time
import uuid
from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend
from visualpath.backends.pathway.stats import PathwayStats

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.core.extractor import BaseExtractor, Observation
    from visualpath.core.fusion import BaseFusion
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData

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
    video frames. This provides several advantages over SimpleBackend:

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
        >>> triggers = backend.run(
        ...     frames=video.stream(),
        ...     extractors=[face_ext, pose_ext],
        ...     fusion=expression_fusion,
        ... )

    Note:
        Pathway runs in a separate thread and processes frames
        asynchronously. The run() method blocks until all frames
        are processed.
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

    def run(
        self,
        frames: Iterator["Frame"],
        extractors: List["BaseExtractor"],
        fusion: Optional["BaseFusion"] = None,
        on_trigger: Optional[Callable[["Trigger"], None]] = None,
        on_frame_result: Optional[Callable] = None,
    ) -> List["Trigger"]:
        """Run the pipeline using Pathway streaming.

        This method:
        1. Creates a VideoConnectorSubject from the frame iterator
        2. Builds a Pathway dataflow: frame -> @pw.udf extractors -> observations
        3. If fusion is provided, applies fusion in the subscribe callback
        4. Runs the Pathway Rust engine until all frames are processed

        All Frame and Observation objects are wrapped in pw.PyObjectWrapper
        for transport through the Pathway engine.

        Args:
            frames: Iterator of Frame objects.
            extractors: List of extractors to run.
            fusion: Optional fusion module.
            on_trigger: Optional trigger callback.
            on_frame_result: Optional per-frame callback receiving
                (frame, observations_list, fusion_result_or_None).
                When provided, the original frame is passed through
                the dataflow so callers can access it in the output.

        Returns:
            List of triggers that fired.
        """
        _check_pathway()

        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject,
            FrameSchema,
        )
        from visualpath.backends.pathway.operators import (
            create_multi_extractor_udf,
        )

        # Reset stats for this run
        self._stats.reset()

        triggers: List["Trigger"] = []
        stats = self._stats

        # Emit SessionStartRecord via ObservabilityHub
        hub = self._get_hub()
        session_id = uuid.uuid4().hex[:12]
        if hub.enabled:
            from visualpath.observability.records import SessionStartRecord
            hub.emit(SessionStartRecord(
                session_id=session_id,
                extractors=[ext.name for ext in extractors],
                config={
                    "backend": "pathway",
                    "window_ns": self._window_ns,
                    "allowed_lateness_ns": self._allowed_lateness_ns,
                    "autocommit_ms": self._autocommit_ms,
                },
            ))

        # Initialize extractors
        for ext in extractors:
            ext.initialize()

        try:
            # 1. Create frame source via instrumented ConnectorSubject
            subject = _InstrumentedConnectorSubject(frames, stats)
            frames_table = pw.io.python.read(
                subject,
                schema=FrameSchema,
                autocommit_duration_ms=self._autocommit_ms,
            )

            # 2. Build extractor UDF with per-extractor timing
            raw_udf = create_multi_extractor_udf(extractors)

            @pw.udf
            def extract_all(frame_wrapped: pw.PyObjectWrapper) -> pw.PyObjectWrapper:
                """Run all extractors on a frame through Pathway engine."""
                frame = frame_wrapped.value
                t0 = time.perf_counter()
                results = raw_udf(frame)
                elapsed_total_ms = (time.perf_counter() - t0) * 1000

                # Record per-extractor stats
                for r in results:
                    stats.record_extraction(
                        r.source,
                        elapsed_total_ms / max(len(results), 1),
                        success=(r.observation is not None),
                    )
                stats.record_frame_extracted()

                # Emit per-frame TimingRecord if hub enabled
                if hub.enabled:
                    from visualpath.observability.records import TimingRecord
                    hub.emit(TimingRecord(
                        frame_id=frame.frame_id,
                        component="pathway_udf",
                        processing_ms=elapsed_total_ms,
                        is_slow=elapsed_total_ms > 50.0,
                    ))

                return pw.PyObjectWrapper(results)

            # 3. Apply extractors as Pathway UDF
            # Include frame passthrough when on_frame_result callback is provided
            if on_frame_result is not None:
                obs_table = frames_table.select(
                    frame_id=pw.this.frame_id,
                    t_ns=pw.this.t_ns,
                    frame=pw.this.frame,
                    observations=extract_all(pw.this.frame),
                )
            else:
                obs_table = frames_table.select(
                    frame_id=pw.this.frame_id,
                    t_ns=pw.this.t_ns,
                    observations=extract_all(pw.this.frame),
                )

            # 4. Subscribe to output - apply fusion in callback
            if fusion is not None:
                def on_change(key, row, time, is_addition):
                    if not is_addition:
                        return
                    stats.record_observation_output()
                    obs_results = row["observations"].value
                    obs_list = [
                        r.observation for r in obs_results
                        if r.observation is not None
                    ]
                    last_fusion_result = None
                    for obs in obs_list:
                        result = fusion.update(obs)
                        last_fusion_result = result
                        if result.should_trigger and result.trigger:
                            stats.record_trigger()
                            triggers.append(result.trigger)
                            if on_trigger:
                                on_trigger(result.trigger)
                    if on_frame_result is not None:
                        frame = row["frame"].value
                        on_frame_result(frame, obs_list, last_fusion_result)

                pw.io.subscribe(obs_table, on_change=on_change)
            else:
                def on_change_noop(key, row, time, is_addition):
                    if not is_addition:
                        return
                    stats.record_observation_output()
                    if on_frame_result is not None:
                        frame = row["frame"].value
                        obs_results = row["observations"].value
                        obs_list = [
                            r.observation for r in obs_results
                            if r.observation is not None
                        ]
                        on_frame_result(frame, obs_list, None)

                pw.io.subscribe(obs_table, on_change=on_change_noop)

            # 5. Run the Pathway Rust engine
            stats.mark_pipeline_start()
            pw.run()
            stats.mark_pipeline_end()

        finally:
            for ext in extractors:
                ext.cleanup()

        # Emit SessionEndRecord
        if hub.enabled:
            from visualpath.observability.records import SessionEndRecord
            hub.emit(SessionEndRecord(
                session_id=session_id,
                duration_sec=stats.pipeline_duration_sec,
                total_frames=stats.frames_extracted,
                total_triggers=stats.triggers_fired,
                avg_fps=stats.throughput_fps,
            ))

        return triggers

    def run_graph(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        on_trigger: Optional[Callable[["FlowData"], None]] = None,
    ) -> List["FlowData"]:
        """Run the pipeline using a FlowGraph with Pathway.

        This method converts the FlowGraph to a Pathway dataflow
        and executes it using the Pathway engine.

        Args:
            frames: Iterator of Frame objects.
            graph: FlowGraph defining the pipeline.
            on_trigger: Optional trigger callback.

        Returns:
            List of FlowData that reached terminal nodes.
        """
        _check_pathway()

        from visualpath.backends.pathway.connector import (
            VideoConnectorSubject,
            FrameSchema,
        )
        from visualpath.backends.pathway.converter import FlowGraphConverter

        # Reset stats for this run
        self._stats.reset()
        stats = self._stats

        results: List["FlowData"] = []

        # Initialize graph nodes
        graph.initialize()

        try:
            # Create frame source
            subject = _InstrumentedConnectorSubject(frames, stats)
            frames_table = pw.io.python.read(
                subject,
                schema=FrameSchema,
                autocommit_duration_ms=self._autocommit_ms,
            )

            # Convert FlowGraph to Pathway dataflow
            converter = FlowGraphConverter(
                window_ns=self._window_ns,
                allowed_lateness_ns=self._allowed_lateness_ns,
            )
            output_table = converter.convert(graph, frames_table)

            # Register output callback
            def on_output(key, row, time, is_addition):
                if not is_addition:
                    return
                from visualpath.flow.node import FlowData
                frame_wrapper = row.get("frame")
                frame_val = frame_wrapper.value if (
                    frame_wrapper is not None
                    and hasattr(frame_wrapper, "value")
                ) else None
                data = FlowData(
                    frame=frame_val,
                    timestamp_ns=row.get("t_ns", 0),
                )
                results.append(data)
                if on_trigger is not None:
                    results_wrapper = row.get("results")
                    if results_wrapper is not None:
                        result_list = (
                            results_wrapper.value
                            if hasattr(results_wrapper, "value")
                            else results_wrapper
                        )
                        should_fire = any(
                            r.should_trigger for r in result_list
                        )
                        if should_fire:
                            on_trigger(data)

            pw.io.subscribe(output_table, on_change=on_output)

            # Run the Pathway engine
            stats.mark_pipeline_start()
            pw.run()
            stats.mark_pipeline_end()

        finally:
            graph.cleanup()

        return results

    @staticmethod
    def _get_hub():
        """Get the ObservabilityHub singleton."""
        from visualpath.observability import ObservabilityHub
        return ObservabilityHub.get_instance()


if PATHWAY_AVAILABLE:
    class _InstrumentedConnectorSubject(pw.io.python.ConnectorSubject):
        """ConnectorSubject that counts ingested frames."""

        def __init__(self, frames: Iterator, stats: PathwayStats) -> None:
            super().__init__()
            self._frames = frames
            self._stats = stats

        def run(self) -> None:
            for frame in self._frames:
                self._stats.record_ingestion()
                self.next(
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    frame=pw.PyObjectWrapper(frame),
                )
            self.close()


__all__ = ["PathwayBackend", "PATHWAY_AVAILABLE"]
