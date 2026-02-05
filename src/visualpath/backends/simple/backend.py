"""SimpleBackend implementation using GraphExecutor.

SimpleBackend executes FlowGraph pipelines using the GraphExecutor,
processing frames sequentially through the DAG.
"""

from typing import Iterator, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend, PipelineResult

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.flow.graph import FlowGraph


class SimpleBackend(ExecutionBackend):
    """GraphExecutor-based sequential execution backend.

    SimpleBackend processes frames sequentially through a FlowGraph using
    GraphExecutor. It is the default backend for local video processing
    and development/debugging.

    For complex scheduling, sampling, or branching, construct a FlowGraph
    with the appropriate nodes (SamplerNode, RateLimiterNode, JoinNode, etc.).

    Examples:
        >>> from visualpath.flow.graph import FlowGraph
        >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
        >>> backend = SimpleBackend()
        >>> result = backend.execute(frames, graph)
        >>> print(result.triggers)

        >>> # With FlowGraphBuilder for complex pipelines
        >>> from visualpath.flow import FlowGraphBuilder
        >>> graph = (FlowGraphBuilder()
        ...     .source("frames")
        ...     .sample(every_nth=3)
        ...     .path("main", extractors=[face_ext], fusion=smile_fusion)
        ...     .build())
        >>> result = backend.execute(frames, graph)
    """

    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
    ) -> PipelineResult:
        """Execute a FlowGraph-based pipeline.

        Processing flow:
        1. Initialize all graph nodes
        2. Process each frame through GraphExecutor
        3. Collect triggers from FlowData results
        4. Clean up all graph nodes

        Args:
            frames: Iterator of Frame objects (not materialized to list).
            graph: FlowGraph defining the pipeline.

        Returns:
            PipelineResult with triggers and frame count.
        """
        from visualpath.flow.executor import GraphExecutor

        triggers = []
        frame_count = 0

        # Register trigger collector on graph
        def collect_trigger(data):
            for result in data.results:
                # result is now an Observation with trigger info
                # Use the should_trigger property and trigger from metadata
                if result.should_trigger and result.trigger:
                    triggers.append(result.trigger)

        graph.on_trigger(collect_trigger)

        executor = GraphExecutor(graph)

        with executor:
            for frame in frames:
                frame_count += 1
                executor.process(frame)

        return PipelineResult(
            triggers=triggers,
            frame_count=frame_count,
        )


__all__ = ["SimpleBackend"]
