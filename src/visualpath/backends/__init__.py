"""Backend protocols and execution backends for visualpath.

This module provides:
- ML backend protocols (DetectionBackend)
- Execution backends (ExecutionBackend, SimpleBackend, PathwayBackend)

Execution backends control how the pipeline processes frames:
- SimpleBackend: Modular backend with composable components
- PathwayBackend: Pathway streaming engine with backpressure

SimpleBackend Components:
- Scheduler: Frame selection/dropping strategy
- Executor: Extractor execution strategy
- Synchronizer: Observation alignment
- Buffer: Backpressure management

Example:
    >>> from visualpath.backends import SimpleBackend
    >>>
    >>> # Default backend
    >>> backend = SimpleBackend()
    >>> triggers = backend.run(frames, extractors=[face_ext])
    >>>
    >>> # Customized backend
    >>> from visualpath.backends.simple import (
    ...     ThreadPoolExecutor,
    ...     TimeWindowSync,
    ... )
    >>> backend = SimpleBackend(
    ...     executor=ThreadPoolExecutor(max_workers=4),
    ...     synchronizer=TimeWindowSync(window_ns=100_000_000),
    ... )
"""

from visualpath.backends.protocols import (
    DetectionBackend,
    DetectionResult,
)
from visualpath.backends.base import ExecutionBackend
from visualpath.backends.simple import (
    SimpleBackend,
    create_default_backend,
    create_parallel_backend,
    create_realtime_backend,
    create_batch_backend,
)

__all__ = [
    # ML backend protocols
    "DetectionBackend",
    "DetectionResult",
    # Execution backends
    "ExecutionBackend",
    "SimpleBackend",
    # Factory functions
    "create_default_backend",
    "create_parallel_backend",
    "create_realtime_backend",
    "create_batch_backend",
]

# Conditionally export PathwayBackend if available
try:
    from visualpath.backends.pathway import PathwayBackend
    __all__.append("PathwayBackend")
except ImportError:
    pass  # Pathway not installed
