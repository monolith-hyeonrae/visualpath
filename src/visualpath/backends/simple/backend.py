"""SimpleBackend implementation with composable components.

SimpleBackend provides a modular execution backend where each aspect
of pipeline execution can be customized independently:
- Scheduler: Frame selection/dropping strategy
- Synchronizer: Observation alignment strategy
- Buffer: Backpressure management
- Executor: Parallel execution strategy
"""

from typing import Callable, Iterator, List, Optional, TYPE_CHECKING

from visualpath.backends.base import ExecutionBackend
from visualpath.backends.simple.scheduler import (
    FrameScheduler,
    PassThroughScheduler,
)
from visualpath.backends.simple.synchronizer import (
    Synchronizer,
    NoSyncSynchronizer,
    TimeWindowSync,
)
from visualpath.backends.simple.executor import (
    ExtractorExecutor,
    SequentialExecutor,
    ThreadPoolExecutor,
)

if TYPE_CHECKING:
    from visualbase import Frame, Trigger
    from visualpath.core.extractor import BaseExtractor, Observation
    from visualpath.core.fusion import BaseFusion
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowData


class SimpleBackend(ExecutionBackend):
    """Modular sequential/parallel execution backend.

    SimpleBackend composes several components to create a flexible
    pipeline execution strategy:

    - **Scheduler**: Controls which frames to process (dropping policy)
    - **Executor**: Controls how extractors run (sequential/parallel)
    - **Synchronizer**: Aligns observations for fusion
    - **Buffer**: Manages backpressure (optional)

    Each component can be customized independently, allowing fine-grained
    control over pipeline behavior.

    Examples:
        >>> # Default: simple sequential processing
        >>> backend = SimpleBackend()

        >>> # Parallel extractors with timeout
        >>> backend = SimpleBackend(
        ...     executor=TimeoutExecutor(timeout_ms=100),
        ... )

        >>> # Frame dropping when slow
        >>> backend = SimpleBackend(
        ...     scheduler=KeyframeScheduler(every_n=3),
        ... )

        >>> # Full customization
        >>> backend = SimpleBackend(
        ...     scheduler=AdaptiveRateScheduler(target_fps=10),
        ...     executor=ThreadPoolExecutor(max_workers=4),
        ...     synchronizer=WatermarkSync(window_ns=100_000_000),
        ... )
    """

    def __init__(
        self,
        scheduler: Optional[FrameScheduler] = None,
        executor: Optional[ExtractorExecutor] = None,
        synchronizer: Optional[Synchronizer] = None,
    ) -> None:
        """Initialize the backend with components.

        Args:
            scheduler: Frame scheduling strategy (default: PassThrough).
            executor: Extractor execution strategy (default: Sequential).
            synchronizer: Observation synchronization (default: NoSync).
        """
        self._scheduler = scheduler or PassThroughScheduler()
        self._executor = executor or SequentialExecutor()
        self._synchronizer = synchronizer or NoSyncSynchronizer()
        self._initialized = False

    @property
    def scheduler(self) -> FrameScheduler:
        """Get the frame scheduler."""
        return self._scheduler

    @property
    def executor(self) -> ExtractorExecutor:
        """Get the extractor executor."""
        return self._executor

    @property
    def synchronizer(self) -> Synchronizer:
        """Get the observation synchronizer."""
        return self._synchronizer

    def initialize(self) -> None:
        """Initialize backend resources."""
        self._initialized = True

    def cleanup(self) -> None:
        """Clean up backend resources."""
        self._executor.shutdown()
        self._initialized = False

    def __enter__(self) -> "SimpleBackend":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def run(
        self,
        frames: Iterator["Frame"],
        extractors: List["BaseExtractor"],
        fusion: Optional["BaseFusion"] = None,
        on_trigger: Optional[Callable[["Trigger"], None]] = None,
    ) -> List["Trigger"]:
        """Run the pipeline with composable components.

        Processing flow:
        1. Scheduler selects which frames to process
        2. Executor runs extractors (sequential or parallel)
        3. Synchronizer aligns observations
        4. Fusion decides on triggers

        Args:
            frames: Iterator of Frame objects.
            extractors: List of extractors to run.
            fusion: Optional fusion module.
            on_trigger: Optional trigger callback.

        Returns:
            List of triggers that fired.
        """
        triggers: List["Trigger"] = []

        try:
            # Initialize extractors
            for ext in extractors:
                ext.initialize()

            # Process frames through scheduler
            for frame in self._scheduler.schedule(frames):
                # Execute extractors
                results = self._executor.execute(frame, extractors)

                # Collect successful observations
                observations: List["Observation"] = []
                for result in results:
                    if result.success and result.observation is not None:
                        observations.append(result.observation)

                # Run fusion if provided
                if fusion and observations:
                    # Add observations to synchronizer
                    for obs in observations:
                        windows = self._synchronizer.add(obs)

                        # Process completed windows
                        for window in windows:
                            for synced_obs in window.all_observations():
                                fusion_result = fusion.update(synced_obs)
                                if fusion_result.should_trigger and fusion_result.trigger:
                                    triggers.append(fusion_result.trigger)
                                    if on_trigger:
                                        on_trigger(fusion_result.trigger)

            # Flush remaining synchronized observations
            if fusion:
                for window in self._synchronizer.flush():
                    for obs in window.all_observations():
                        fusion_result = fusion.update(obs)
                        if fusion_result.should_trigger and fusion_result.trigger:
                            triggers.append(fusion_result.trigger)
                            if on_trigger:
                                on_trigger(fusion_result.trigger)

        finally:
            # Cleanup extractors
            for ext in extractors:
                ext.cleanup()

        return triggers

    def run_graph(
        self,
        frames: Iterator["Frame"],
        graph: "FlowGraph",
        on_trigger: Optional[Callable[["FlowData"], None]] = None,
    ) -> List["FlowData"]:
        """Run the pipeline using a FlowGraph.

        Uses GraphExecutor for DAG-based processing, with frames
        filtered through the scheduler.

        Args:
            frames: Iterator of Frame objects.
            graph: FlowGraph defining the pipeline.
            on_trigger: Optional trigger callback.

        Returns:
            List of FlowData that reached terminal nodes.
        """
        from visualpath.flow.executor import GraphExecutor

        if on_trigger is not None:
            graph.on_trigger(on_trigger)

        executor = GraphExecutor(graph)
        all_results: List["FlowData"] = []

        with executor:
            # Apply scheduler to frames
            for frame in self._scheduler.schedule(frames):
                results = executor.process(frame)
                all_results.extend(results)

        return all_results

    def get_stats(self) -> dict:
        """Get combined statistics from all components.

        Returns:
            Dictionary with stats from scheduler, executor, synchronizer.
        """
        return {
            "scheduler": {
                "frames_received": self._scheduler.stats.frames_received,
                "frames_processed": self._scheduler.stats.frames_processed,
                "frames_dropped": self._scheduler.stats.frames_dropped,
                "drop_rate": self._scheduler.stats.drop_rate,
            },
            "executor": {
                "frames_processed": self._executor.stats.frames_processed,
                "extractions_completed": self._executor.stats.extractions_completed,
                "extractions_failed": self._executor.stats.extractions_failed,
                "extractions_timeout": self._executor.stats.extractions_timeout,
                "total_time_ms": self._executor.stats.total_time_ms,
                "per_extractor_time_ms": self._executor.stats.per_extractor_time_ms,
            },
            "synchronizer": {
                "windows_completed": self._synchronizer.stats.windows_completed,
                "observations_synced": self._synchronizer.stats.observations_synced,
                "observations_dropped": self._synchronizer.stats.observations_dropped,
                "avg_window_fill": self._synchronizer.stats.avg_window_fill,
            },
        }


# Factory functions for common configurations


def create_default_backend() -> SimpleBackend:
    """Create a default SimpleBackend with basic configuration.

    Returns:
        SimpleBackend with sequential execution, no dropping.
    """
    return SimpleBackend()


def create_parallel_backend(
    max_workers: int = 4,
    timeout_ms: Optional[float] = None,
) -> SimpleBackend:
    """Create a SimpleBackend optimized for parallel extraction.

    Args:
        max_workers: Number of parallel workers.
        timeout_ms: Optional timeout per extractor.

    Returns:
        SimpleBackend with parallel execution.
    """
    from visualpath.backends.simple.executor import TimeoutExecutor

    if timeout_ms is not None:
        executor = TimeoutExecutor(timeout_ms=timeout_ms, max_workers=max_workers)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)

    return SimpleBackend(executor=executor)


def create_realtime_backend(
    target_fps: float = 10.0,
    window_ns: int = 100_000_000,
    max_workers: int = 4,
) -> SimpleBackend:
    """Create a SimpleBackend optimized for real-time processing.

    Args:
        target_fps: Target output frame rate.
        window_ns: Synchronization window size.
        max_workers: Number of parallel workers.

    Returns:
        SimpleBackend with adaptive scheduling and windowed sync.
    """
    from visualpath.backends.simple.scheduler import AdaptiveRateScheduler
    from visualpath.backends.simple.executor import TimeoutExecutor

    return SimpleBackend(
        scheduler=AdaptiveRateScheduler(target_fps=target_fps),
        executor=TimeoutExecutor(timeout_ms=1000 / target_fps, max_workers=max_workers),
        synchronizer=TimeWindowSync(window_ns=window_ns),
    )


def create_batch_backend(
    max_workers: int = 8,
    window_ns: int = 200_000_000,
) -> SimpleBackend:
    """Create a SimpleBackend optimized for offline batch processing.

    Args:
        max_workers: Number of parallel workers.
        window_ns: Synchronization window size.

    Returns:
        SimpleBackend with high parallelism, no dropping.
    """
    return SimpleBackend(
        scheduler=PassThroughScheduler(),
        executor=ThreadPoolExecutor(max_workers=max_workers),
        synchronizer=TimeWindowSync(window_ns=window_ns, allowed_lateness_ns=window_ns),
    )


__all__ = [
    "SimpleBackend",
    "create_default_backend",
    "create_parallel_backend",
    "create_realtime_backend",
    "create_batch_backend",
]
