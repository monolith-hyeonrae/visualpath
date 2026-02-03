"""Base execution backend interface.

ExecutionBackend defines the abstract interface for pipeline execution backends.
Different backends can provide different execution strategies (simple sequential,
Pathway streaming, etc.) while maintaining the same API.

Example:
    >>> from visualpath.backends import ExecutionBackend
    >>>
    >>> class MyBackend(ExecutionBackend):
    ...     def run(self, frames, extractors, fusion=None, on_trigger=None):
    ...         # Custom execution logic
    ...         ...
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.core.extractor import BaseExtractor
    from visualpath.core.fusion import BaseFusion
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData


class ExecutionBackend(ABC):
    """Abstract base class for pipeline execution backends.

    ExecutionBackend provides a unified interface for running video analysis
    pipelines. Implementations handle the details of frame processing,
    extractor execution, synchronization, and fusion.

    Available backends:
    - SimpleBackend: Sequential processing (default)
    - PathwayBackend: Pathway streaming engine

    Example:
        >>> backend = SimpleBackend()
        >>> triggers = backend.run(
        ...     frames=video.stream(),
        ...     extractors=[face_ext, pose_ext],
        ...     fusion=smile_fusion,
        ... )
    """

    @property
    def name(self) -> str:
        """Backend identifier name."""
        return self.__class__.__name__

    @abstractmethod
    def run(
        self,
        frames: Iterator["Frame"],
        extractors: List["BaseExtractor"],
        fusion: Optional["BaseFusion"] = None,
        on_trigger: Optional[Callable[["Trigger"], None]] = None,
    ) -> List["Trigger"]:
        """Run the pipeline with extractors and optional fusion.

        This is the primary method for simple pipeline execution.

        Args:
            frames: Iterator of Frame objects from video source.
            extractors: List of extractors to run on each frame.
            fusion: Optional fusion module for trigger decisions.
            on_trigger: Optional callback invoked when a trigger fires.

        Returns:
            List of all triggers that fired during processing.

        Example:
            >>> triggers = backend.run(
            ...     frames=video.stream(fps=10),
            ...     extractors=[face_ext],
            ...     fusion=smile_fusion,
            ...     on_trigger=lambda t: print(f"Trigger: {t.label}"),
            ... )
        """
        ...

    @abstractmethod
    def run_graph(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        on_trigger: Optional[Callable[["FlowData"], None]] = None,
    ) -> List["FlowData"]:
        """Run the pipeline using a FlowGraph.

        This method supports complex DAG-based pipelines with branching,
        joining, and multi-path processing.

        Args:
            frames: Iterator of Frame objects from video source.
            graph: FlowGraph defining the processing pipeline.
            on_trigger: Optional callback invoked when data reaches
                terminal nodes with should_trigger=True.

        Returns:
            List of FlowData that reached terminal nodes.

        Example:
            >>> graph = FlowGraphBuilder()
            ...     .source("frames")
            ...     .fanout(["face", "pose"])
            ...     .path("face", extractors=[face_ext])
            ...     .path("pose", extractors=[pose_ext])
            ...     .join(["face", "pose"])
            ...     .build()
            >>>
            >>> results = backend.run_graph(frames, graph)
        """
        ...


__all__ = ["ExecutionBackend"]
