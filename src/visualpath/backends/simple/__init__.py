"""Simple execution backend using GraphExecutor.

The simple backend executes FlowGraph pipelines using the GraphExecutor,
processing frames sequentially through the DAG.

Example:
    >>> from visualpath.backends.simple import SimpleBackend
    >>> from visualpath.flow.graph import FlowGraph
    >>>
    >>> graph = FlowGraph.from_pipeline([face_ext], fusion=smile_fusion)
    >>> backend = SimpleBackend()
    >>> result = backend.execute(frames, graph)
"""

from visualpath.backends.simple.backend import SimpleBackend

__all__ = [
    "SimpleBackend",
]
