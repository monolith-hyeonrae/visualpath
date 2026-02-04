"""Backend protocols and execution backends for visualpath.

This module provides:
- ML backend protocols (DetectionBackend)
- Execution backends (ExecutionBackend, SimpleBackend, PathwayBackend)

Execution backends control how the pipeline processes frames:
- SimpleBackend: GraphExecutor-based sequential processing
- PathwayBackend: Pathway streaming engine with backpressure

Example:
    >>> from visualpath.backends import SimpleBackend
    >>> from visualpath.flow.graph import FlowGraph
    >>>
    >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
    >>> backend = SimpleBackend()
    >>> result = backend.execute(frames, graph)
"""

from visualpath.backends.protocols import (
    DetectionBackend,
    DetectionResult,
)
from visualpath.backends.base import ExecutionBackend, PipelineResult
from visualpath.backends.simple import SimpleBackend

__all__ = [
    # ML backend protocols
    "DetectionBackend",
    "DetectionResult",
    # Execution backends
    "ExecutionBackend",
    "PipelineResult",
    "SimpleBackend",
]

# Conditionally export PathwayBackend if available
try:
    from visualpath.backends.pathway import PathwayBackend
    __all__.append("PathwayBackend")
except ImportError:
    pass  # Pathway not installed
