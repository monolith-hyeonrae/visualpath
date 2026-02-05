"""Base execution backend interface.

ExecutionBackend defines the single abstract interface for pipeline execution.
Backends are spec interpreters — they read the FlowGraph (AST) and execute it.

Each backend provides its own interpretation of NodeSpecs:
- SimpleBackend: uses SimpleInterpreter for synchronous execution
- PathwayBackend: converts specs to Pathway operators for streaming
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.flow.graph import FlowGraph


@dataclass
class PipelineResult:
    """Result from executing a pipeline.

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

    One method: ``execute(frames, graph) -> PipelineResult``.

    Example:
        >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
        >>> result = SimpleBackend().execute(frames, graph)
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

        Args:
            frames: Iterator of Frame objects from video source.
            graph: FlowGraph defining the processing pipeline.

        Returns:
            PipelineResult with triggers, frame_count, and optional stats.
        """
        ...


__all__ = ["ExecutionBackend", "PipelineResult"]
