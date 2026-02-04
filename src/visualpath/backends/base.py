"""Base execution backend interface.

ExecutionBackend defines the abstract interface for pipeline execution backends.
Different backends can provide different execution strategies (simple sequential,
Pathway streaming, etc.) while maintaining the same API.

Example:
    >>> from visualpath.backends import ExecutionBackend
    >>>
    >>> class MyBackend(ExecutionBackend):
    ...     def execute(self, frames, graph):
    ...         # Custom execution logic
    ...         ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.core.extractor import BaseExtractor
    from visualpath.core.fusion import BaseFusion
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData


@dataclass
class PipelineResult:
    """Result from executing a pipeline via ExecutionBackend.execute().

    Attributes:
        triggers: List of triggers that fired during processing.
        frame_count: Total frames processed.
        stats: Optional backend-specific statistics.
    """

    triggers: List["Trigger"] = field(default_factory=list)
    frame_count: int = 0
    stats: dict = field(default_factory=dict)


class ExecutionBackend(ABC):
    """Abstract base class for pipeline execution backends.

    ExecutionBackend provides a unified interface for running video analysis
    pipelines. The primary method is ``execute(frames, graph)`` which takes
    a FlowGraph defining the pipeline structure.

    Available backends:
    - SimpleBackend: Sequential processing via GraphExecutor (default)
    - PathwayBackend: Pathway streaming engine

    Example:
        >>> from visualpath.flow.graph import FlowGraph
        >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
        >>> backend = SimpleBackend()
        >>> result = backend.execute(frames, graph)
        >>> print(result.triggers)
    """

    @property
    def name(self) -> str:
        """Backend identifier name."""
        return self.__class__.__name__

    @abstractmethod
    def execute(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
    ) -> PipelineResult:
        """Execute a FlowGraph-based pipeline.

        This is the primary execution method. All backends must implement it.

        Args:
            frames: Iterator of Frame objects from video source.
                Must not be materialized into a list.
            graph: FlowGraph defining the processing pipeline.

        Returns:
            PipelineResult with triggers, frame_count, and optional stats.
        """
        ...

    # ------------------------------------------------------------------
    # Backward-compatible shims
    # ------------------------------------------------------------------

    def run(
        self,
        frames: Iterator["Frame"],
        extractors: List["BaseExtractor"],
        fusion: Optional["BaseFusion"] = None,
        on_trigger: Optional[Callable[["Trigger"], None]] = None,
    ) -> List["Trigger"]:
        """Run a simple pipeline (backward-compatible shim).

        Converts the extractor/fusion arguments to a FlowGraph and
        delegates to ``execute()``.

        Args:
            frames: Iterator of Frame objects.
            extractors: List of extractors to run on each frame.
            fusion: Optional fusion module for trigger decisions.
            on_trigger: Optional callback invoked when a trigger fires.

        Returns:
            List of all triggers that fired during processing.
        """
        from visualpath.flow.graph import FlowGraph

        graph = FlowGraph.from_pipeline(extractors, fusion)
        if on_trigger:
            original_callback = on_trigger

            def _trigger_callback(data: "FlowData") -> None:
                for result in data.results:
                    if result.should_trigger and result.trigger:
                        original_callback(result.trigger)

            graph.on_trigger(_trigger_callback)

        result = self.execute(frames, graph)
        return result.triggers

    def run_graph(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        on_trigger: Optional[Callable[["FlowData"], None]] = None,
    ) -> List["FlowData"]:
        """Run a FlowGraph pipeline (backward-compatible shim).

        Delegates to ``execute()`` and returns an empty list to match
        the previous return type.

        Args:
            frames: Iterator of Frame objects.
            graph: FlowGraph defining the processing pipeline.
            on_trigger: Optional callback for trigger events.

        Returns:
            Empty list (for backward compatibility).
        """
        if on_trigger:
            graph.on_trigger(on_trigger)
        self.execute(frames, graph)
        return []


__all__ = ["ExecutionBackend", "PipelineResult"]
