"""Simple execution backend with composable components.

The simple backend provides a modular approach to pipeline execution,
allowing independent customization of:

- **Scheduler**: Frame selection and dropping strategy
- **Executor**: Extractor execution strategy (sequential/parallel)
- **Synchronizer**: Observation alignment strategy
- **Buffer**: Backpressure management

Example:
    >>> from visualpath.backends.simple import (
    ...     SimpleBackend,
    ...     AdaptiveRateScheduler,
    ...     ThreadPoolExecutor,
    ...     TimeWindowSync,
    ... )
    >>>
    >>> backend = SimpleBackend(
    ...     scheduler=AdaptiveRateScheduler(target_fps=10),
    ...     executor=ThreadPoolExecutor(max_workers=4),
    ...     synchronizer=TimeWindowSync(window_ns=100_000_000),
    ... )

Factory functions for common configurations:
    >>> from visualpath.backends.simple import (
    ...     create_default_backend,
    ...     create_parallel_backend,
    ...     create_realtime_backend,
    ...     create_batch_backend,
    ... )
"""

# Backend
from visualpath.backends.simple.backend import (
    SimpleBackend,
    create_default_backend,
    create_parallel_backend,
    create_realtime_backend,
    create_batch_backend,
)

# Schedulers
from visualpath.backends.simple.scheduler import (
    SchedulerStats,
    FrameScheduler,
    PassThroughScheduler,
    SkipOldestScheduler,
    SkipIntermediateScheduler,
    KeyframeScheduler,
    AdaptiveRateScheduler,
)

# Synchronizers
from visualpath.backends.simple.synchronizer import (
    SyncStats,
    SyncWindow,
    Synchronizer,
    NoSyncSynchronizer,
    TimeWindowSync,
    WatermarkSync,
    BarrierSync,
)

# Buffers
from visualpath.backends.simple.buffer import (
    OverflowPolicy,
    BufferStats,
    BackpressureBuffer,
    UnboundedBuffer,
    BoundedBuffer,
    SlidingWindowBuffer,
    PriorityBuffer,
    BufferOverflowError,
)

# Executors
from visualpath.backends.simple.executor import (
    ExecutorStats,
    ExtractionResult,
    ExtractorExecutor,
    SequentialExecutor,
    ThreadPoolExecutor,
    TimeoutExecutor,
    AdaptiveExecutor,
)

__all__ = [
    # Backend
    "SimpleBackend",
    "create_default_backend",
    "create_parallel_backend",
    "create_realtime_backend",
    "create_batch_backend",
    # Schedulers
    "SchedulerStats",
    "FrameScheduler",
    "PassThroughScheduler",
    "SkipOldestScheduler",
    "SkipIntermediateScheduler",
    "KeyframeScheduler",
    "AdaptiveRateScheduler",
    # Synchronizers
    "SyncStats",
    "SyncWindow",
    "Synchronizer",
    "NoSyncSynchronizer",
    "TimeWindowSync",
    "WatermarkSync",
    "BarrierSync",
    # Buffers
    "OverflowPolicy",
    "BufferStats",
    "BackpressureBuffer",
    "UnboundedBuffer",
    "BoundedBuffer",
    "SlidingWindowBuffer",
    "PriorityBuffer",
    "BufferOverflowError",
    # Executors
    "ExecutorStats",
    "ExtractionResult",
    "ExtractorExecutor",
    "SequentialExecutor",
    "ThreadPoolExecutor",
    "TimeoutExecutor",
    "AdaptiveExecutor",
]
