"""Extractor execution strategies for SimpleBackend.

Executors control how extractors are invoked - sequentially,
in parallel threads, or asynchronously.

Available strategies:
- SequentialExecutor: Run extractors one at a time
- ThreadPoolExecutor: Run extractors in thread pool
- TimeoutExecutor: Run with timeout, skip slow extractors
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import as_completed, Future, TimeoutError
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.extractor import BaseExtractor, Observation


@dataclass
class ExecutorStats:
    """Statistics from extractor execution.

    Attributes:
        frames_processed: Total frames processed.
        extractions_completed: Successful extractions.
        extractions_failed: Failed extractions.
        extractions_timeout: Extractions that timed out.
        total_time_ms: Total execution time.
        per_extractor_time_ms: Time per extractor.
    """
    frames_processed: int = 0
    extractions_completed: int = 0
    extractions_failed: int = 0
    extractions_timeout: int = 0
    total_time_ms: float = 0.0
    per_extractor_time_ms: Dict[str, float] = field(default_factory=dict)

    def record_extraction(self, extractor_name: str, time_ms: float) -> None:
        """Record extraction timing."""
        current = self.per_extractor_time_ms.get(extractor_name, 0.0)
        # Running average
        count = self.extractions_completed
        if count > 0:
            self.per_extractor_time_ms[extractor_name] = (
                current * (count - 1) + time_ms
            ) / count
        else:
            self.per_extractor_time_ms[extractor_name] = time_ms


@dataclass
class ExtractionResult:
    """Result from extractor execution.

    Attributes:
        extractor_name: Name of the extractor.
        observation: The observation, or None if failed/timeout.
        success: Whether extraction succeeded.
        error: Error message if failed.
        time_ms: Execution time in milliseconds.
    """
    extractor_name: str
    observation: Optional["Observation"]
    success: bool = True
    error: Optional[str] = None
    time_ms: float = 0.0


class ExtractorExecutor(ABC):
    """Abstract base class for extractor execution strategies.

    Executors run multiple extractors on a frame and collect
    the results. Different strategies offer different trade-offs
    between throughput, latency, and resource usage.

    Example:
        >>> executor = ThreadPoolExecutor(max_workers=4)
        >>> results = executor.execute(frame, [face_ext, pose_ext])
        >>> observations = [r.observation for r in results if r.success]
    """

    @abstractmethod
    def execute(
        self,
        frame: "Frame",
        extractors: List["BaseExtractor"],
    ) -> List[ExtractionResult]:
        """Execute extractors on a frame.

        Args:
            frame: Frame to process.
            extractors: List of extractors to run.

        Returns:
            List of extraction results.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown executor and release resources."""
        ...

    @property
    @abstractmethod
    def stats(self) -> ExecutorStats:
        """Get execution statistics."""
        ...


class SequentialExecutor(ExtractorExecutor):
    """Executor that runs extractors sequentially.

    Simple and predictable - runs each extractor one at a time.
    Good for debugging and when extractors share resources.

    Example:
        >>> executor = SequentialExecutor()
        >>> results = executor.execute(frame, extractors)
    """

    def __init__(self) -> None:
        self._stats = ExecutorStats()

    def execute(
        self,
        frame: "Frame",
        extractors: List["BaseExtractor"],
    ) -> List[ExtractionResult]:
        """Execute extractors sequentially with deps accumulation."""
        results = []
        deps: Dict[str, "Observation"] = {}
        frame_start = time.perf_counter()

        for extractor in extractors:
            start = time.perf_counter()
            try:
                extractor_deps = None
                if extractor.depends:
                    extractor_deps = {
                        name: deps[name]
                        for name in extractor.depends
                        if name in deps
                    }
                try:
                    observation = extractor.extract(frame, extractor_deps)
                except TypeError:
                    observation = extractor.extract(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000

                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=observation,
                    success=True,
                    time_ms=elapsed_ms,
                ))
                if observation is not None:
                    deps[extractor.name] = observation
                self._stats.extractions_completed += 1
                self._stats.record_extraction(extractor.name, elapsed_ms)

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=None,
                    success=False,
                    error=str(e),
                    time_ms=elapsed_ms,
                ))
                self._stats.extractions_failed += 1

        self._stats.frames_processed += 1
        self._stats.total_time_ms += (time.perf_counter() - frame_start) * 1000

        return results

    def shutdown(self) -> None:
        """Nothing to shutdown."""
        pass

    @property
    def stats(self) -> ExecutorStats:
        return self._stats


class ThreadPoolExecutor(ExtractorExecutor):
    """Executor that runs extractors in parallel threads.

    Uses a thread pool to run extractors concurrently. Good for
    I/O-bound extractors or when extractors can run independently.

    Example:
        >>> executor = ThreadPoolExecutor(max_workers=4)
        >>> results = executor.execute(frame, extractors)
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the executor.

        Args:
            max_workers: Maximum number of worker threads.
        """
        self._max_workers = max_workers
        self._pool = _ThreadPoolExecutor(max_workers=max_workers)
        self._stats = ExecutorStats()

    def _build_layers(
        self,
        extractors: List["BaseExtractor"],
    ) -> List[List["BaseExtractor"]]:
        """Build dependency layers for extractors.

        Layer 0: extractors with no depends
        Layer 1: extractors depending on layer 0 results
        etc.

        Returns:
            List of layers, each layer is a list of extractors.
        """
        layers: List[List["BaseExtractor"]] = []
        resolved: set = set()

        remaining = list(extractors)
        while remaining:
            layer = []
            still_remaining = []
            for ext in remaining:
                if not ext.depends or all(d in resolved for d in ext.depends):
                    layer.append(ext)
                else:
                    still_remaining.append(ext)
            if not layer:
                # Circular or unresolvable deps - put all remaining in one layer
                layer = still_remaining
                still_remaining = []
            layers.append(layer)
            for ext in layer:
                resolved.add(ext.name)
            remaining = still_remaining

        return layers

    def execute(
        self,
        frame: "Frame",
        extractors: List["BaseExtractor"],
    ) -> List[ExtractionResult]:
        """Execute extractors in parallel with dependency layers."""
        if not extractors:
            return []

        results = []
        deps: Dict[str, "Observation"] = {}
        frame_start = time.perf_counter()

        # Check if any extractor has depends
        has_deps = any(ext.depends for ext in extractors)

        if not has_deps:
            # No dependencies - run all in parallel (original behavior)
            future_to_extractor: Dict[Future, "BaseExtractor"] = {}
            start_times: Dict[str, float] = {}

            for extractor in extractors:
                start_times[extractor.name] = time.perf_counter()
                future = self._pool.submit(extractor.extract, frame)
                future_to_extractor[future] = extractor

            for future in as_completed(future_to_extractor):
                extractor = future_to_extractor[future]
                elapsed_ms = (time.perf_counter() - start_times[extractor.name]) * 1000

                try:
                    observation = future.result()
                    results.append(ExtractionResult(
                        extractor_name=extractor.name,
                        observation=observation,
                        success=True,
                        time_ms=elapsed_ms,
                    ))
                    self._stats.extractions_completed += 1
                    self._stats.record_extraction(extractor.name, elapsed_ms)
                except Exception as e:
                    results.append(ExtractionResult(
                        extractor_name=extractor.name,
                        observation=None,
                        success=False,
                        error=str(e),
                        time_ms=elapsed_ms,
                    ))
                    self._stats.extractions_failed += 1
        else:
            # Dependencies exist - run layer by layer
            layers = self._build_layers(extractors)
            for layer in layers:
                future_to_extractor = {}
                start_times = {}

                for extractor in layer:
                    extractor_deps = None
                    if extractor.depends:
                        extractor_deps = {
                            name: deps[name]
                            for name in extractor.depends
                            if name in deps
                        }
                    start_times[extractor.name] = time.perf_counter()
                    future = self._pool.submit(
                        self._extract_with_deps, extractor, frame, extractor_deps,
                    )
                    future_to_extractor[future] = extractor

                for future in as_completed(future_to_extractor):
                    extractor = future_to_extractor[future]
                    elapsed_ms = (time.perf_counter() - start_times[extractor.name]) * 1000

                    try:
                        observation = future.result()
                        results.append(ExtractionResult(
                            extractor_name=extractor.name,
                            observation=observation,
                            success=True,
                            time_ms=elapsed_ms,
                        ))
                        if observation is not None:
                            deps[extractor.name] = observation
                        self._stats.extractions_completed += 1
                        self._stats.record_extraction(extractor.name, elapsed_ms)
                    except Exception as e:
                        results.append(ExtractionResult(
                            extractor_name=extractor.name,
                            observation=None,
                            success=False,
                            error=str(e),
                            time_ms=elapsed_ms,
                        ))
                        self._stats.extractions_failed += 1

        self._stats.frames_processed += 1
        self._stats.total_time_ms += (time.perf_counter() - frame_start) * 1000

        return results

    @staticmethod
    def _extract_with_deps(extractor, frame, deps):
        """Call extractor.extract with deps and TypeError fallback."""
        try:
            return extractor.extract(frame, deps)
        except TypeError:
            return extractor.extract(frame)

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=True)

    @property
    def stats(self) -> ExecutorStats:
        return self._stats


class TimeoutExecutor(ExtractorExecutor):
    """Executor with per-extractor timeout.

    Runs extractors in parallel but enforces a timeout. Slow
    extractors are skipped to maintain throughput.

    Example:
        >>> executor = TimeoutExecutor(
        ...     timeout_ms=100,
        ...     max_workers=4,
        ... )
        >>> results = executor.execute(frame, extractors)
        >>> # Slow extractors will have success=False, error="timeout"
    """

    def __init__(
        self,
        timeout_ms: float = 100.0,
        max_workers: int = 4,
    ) -> None:
        """Initialize the executor.

        Args:
            timeout_ms: Timeout per extractor in milliseconds.
            max_workers: Maximum number of worker threads.
        """
        self._timeout_ms = timeout_ms
        self._timeout_sec = timeout_ms / 1000
        self._max_workers = max_workers
        self._pool = _ThreadPoolExecutor(max_workers=max_workers)
        self._stats = ExecutorStats()

    @staticmethod
    def _extract_with_deps(extractor, frame, deps):
        """Call extractor.extract with deps and TypeError fallback."""
        try:
            return extractor.extract(frame, deps)
        except TypeError:
            return extractor.extract(frame)

    def execute(
        self,
        frame: "Frame",
        extractors: List["BaseExtractor"],
    ) -> List[ExtractionResult]:
        """Execute extractors with timeout and deps support.

        Extractors with dependencies run sequentially; independent
        extractors run in parallel with timeout.
        """
        if not extractors:
            return []

        results = []
        deps: Dict[str, "Observation"] = {}
        frame_start = time.perf_counter()

        # Split into independent and dependent extractors
        independent = [e for e in extractors if not e.depends]
        dependent = [e for e in extractors if e.depends]

        # Run independent extractors in parallel with timeout
        if independent:
            future_to_extractor: Dict[Future, "BaseExtractor"] = {}
            start_times: Dict[str, float] = {}

            for extractor in independent:
                start_times[extractor.name] = time.perf_counter()
                future = self._pool.submit(extractor.extract, frame)
                future_to_extractor[future] = extractor

            try:
                for future in as_completed(future_to_extractor, timeout=self._timeout_sec):
                    extractor = future_to_extractor[future]
                    elapsed_ms = (time.perf_counter() - start_times[extractor.name]) * 1000

                    try:
                        observation = future.result(timeout=0)
                        results.append(ExtractionResult(
                            extractor_name=extractor.name,
                            observation=observation,
                            success=True,
                            time_ms=elapsed_ms,
                        ))
                        if observation is not None:
                            deps[extractor.name] = observation
                        self._stats.extractions_completed += 1
                        self._stats.record_extraction(extractor.name, elapsed_ms)
                    except Exception as e:
                        results.append(ExtractionResult(
                            extractor_name=extractor.name,
                            observation=None,
                            success=False,
                            error=str(e),
                            time_ms=elapsed_ms,
                        ))
                        self._stats.extractions_failed += 1
            except TimeoutError:
                pass

            # Handle remaining futures that didn't complete
            for future, extractor in future_to_extractor.items():
                if extractor.name not in [r.extractor_name for r in results]:
                    elapsed_ms = (time.perf_counter() - start_times[extractor.name]) * 1000
                    results.append(ExtractionResult(
                        extractor_name=extractor.name,
                        observation=None,
                        success=False,
                        error="timeout",
                        time_ms=elapsed_ms,
                    ))
                    self._stats.extractions_timeout += 1
                    future.cancel()

        # Run dependent extractors sequentially (need deps from above)
        for extractor in dependent:
            start = time.perf_counter()
            try:
                extractor_deps = {
                    name: deps[name]
                    for name in extractor.depends
                    if name in deps
                }
                observation = self._extract_with_deps(extractor, frame, extractor_deps)
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=observation,
                    success=True,
                    time_ms=elapsed_ms,
                ))
                if observation is not None:
                    deps[extractor.name] = observation
                self._stats.extractions_completed += 1
                self._stats.record_extraction(extractor.name, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=None,
                    success=False,
                    error=str(e),
                    time_ms=elapsed_ms,
                ))
                self._stats.extractions_failed += 1

        self._stats.frames_processed += 1
        self._stats.total_time_ms += (time.perf_counter() - frame_start) * 1000

        return results

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def stats(self) -> ExecutorStats:
        return self._stats

    @property
    def timeout_ms(self) -> float:
        """Current timeout setting."""
        return self._timeout_ms


class AdaptiveExecutor(ExtractorExecutor):
    """Executor that adapts strategy based on extractor performance.

    Monitors extractor performance and adjusts execution strategy:
    - Fast extractors run sequentially (less overhead)
    - Slow extractors run in parallel

    Example:
        >>> executor = AdaptiveExecutor()
        >>> # Automatically optimizes execution strategy
    """

    def __init__(
        self,
        parallel_threshold_ms: float = 20.0,
        max_workers: int = 4,
    ) -> None:
        """Initialize the executor.

        Args:
            parallel_threshold_ms: Extractors slower than this run in parallel.
            max_workers: Maximum workers for parallel execution.
        """
        self._parallel_threshold_ms = parallel_threshold_ms
        self._max_workers = max_workers
        self._pool: Optional[_ThreadPoolExecutor] = None
        self._stats = ExecutorStats()
        self._extractor_avg_times: Dict[str, float] = {}

    def _get_pool(self) -> _ThreadPoolExecutor:
        """Lazily create thread pool."""
        if self._pool is None:
            self._pool = _ThreadPoolExecutor(max_workers=self._max_workers)
        return self._pool

    def _should_parallelize(self, extractor: "BaseExtractor") -> bool:
        """Check if extractor should run in parallel."""
        avg_time = self._extractor_avg_times.get(extractor.name)
        if avg_time is None:
            return True  # Unknown - try parallel first
        return avg_time > self._parallel_threshold_ms

    @staticmethod
    def _extract_with_deps(extractor, frame, deps):
        """Call extractor.extract with deps and TypeError fallback."""
        try:
            return extractor.extract(frame, deps)
        except TypeError:
            return extractor.extract(frame)

    def execute(
        self,
        frame: "Frame",
        extractors: List["BaseExtractor"],
    ) -> List[ExtractionResult]:
        """Execute extractors with adaptive strategy and deps support.

        Extractors with dependencies always run sequentially after
        their dependencies complete.
        """
        if not extractors:
            return []

        results = []
        deps: Dict[str, "Observation"] = {}
        frame_start = time.perf_counter()

        # Separate dependent extractors (must run sequentially after deps)
        independent = [e for e in extractors if not e.depends]
        dependent = [e for e in extractors if e.depends]

        # Partition independent extractors by speed
        fast_extractors = []
        slow_extractors = []
        for ext in independent:
            if self._should_parallelize(ext):
                slow_extractors.append(ext)
            else:
                fast_extractors.append(ext)

        # Run fast extractors sequentially
        for extractor in fast_extractors:
            start = time.perf_counter()
            try:
                observation = extractor.extract(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=observation,
                    success=True,
                    time_ms=elapsed_ms,
                ))
                if observation is not None:
                    deps[extractor.name] = observation
                self._stats.extractions_completed += 1
                self._update_avg_time(extractor.name, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=None,
                    success=False,
                    error=str(e),
                    time_ms=elapsed_ms,
                ))
                self._stats.extractions_failed += 1

        # Run slow extractors in parallel
        if slow_extractors:
            pool = self._get_pool()
            future_to_extractor: Dict[Future, "BaseExtractor"] = {}
            start_times: Dict[str, float] = {}

            for extractor in slow_extractors:
                start_times[extractor.name] = time.perf_counter()
                future = pool.submit(extractor.extract, frame)
                future_to_extractor[future] = extractor

            for future in as_completed(future_to_extractor):
                extractor = future_to_extractor[future]
                elapsed_ms = (time.perf_counter() - start_times[extractor.name]) * 1000

                try:
                    observation = future.result()
                    results.append(ExtractionResult(
                        extractor_name=extractor.name,
                        observation=observation,
                        success=True,
                        time_ms=elapsed_ms,
                    ))
                    if observation is not None:
                        deps[extractor.name] = observation
                    self._stats.extractions_completed += 1
                    self._update_avg_time(extractor.name, elapsed_ms)
                except Exception as e:
                    results.append(ExtractionResult(
                        extractor_name=extractor.name,
                        observation=None,
                        success=False,
                        error=str(e),
                        time_ms=elapsed_ms,
                    ))
                    self._stats.extractions_failed += 1

        # Run dependent extractors sequentially
        for extractor in dependent:
            start = time.perf_counter()
            try:
                extractor_deps = {
                    name: deps[name]
                    for name in extractor.depends
                    if name in deps
                }
                observation = self._extract_with_deps(extractor, frame, extractor_deps)
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=observation,
                    success=True,
                    time_ms=elapsed_ms,
                ))
                if observation is not None:
                    deps[extractor.name] = observation
                self._stats.extractions_completed += 1
                self._update_avg_time(extractor.name, elapsed_ms)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(ExtractionResult(
                    extractor_name=extractor.name,
                    observation=None,
                    success=False,
                    error=str(e),
                    time_ms=elapsed_ms,
                ))
                self._stats.extractions_failed += 1

        self._stats.frames_processed += 1
        self._stats.total_time_ms += (time.perf_counter() - frame_start) * 1000

        return results

    def _update_avg_time(self, name: str, time_ms: float) -> None:
        """Update running average time for extractor."""
        current = self._extractor_avg_times.get(name, time_ms)
        # Exponential moving average
        self._extractor_avg_times[name] = current * 0.9 + time_ms * 0.1
        self._stats.record_extraction(name, time_ms)

    def shutdown(self) -> None:
        """Shutdown the thread pool if created."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    @property
    def stats(self) -> ExecutorStats:
        return self._stats


__all__ = [
    "ExecutorStats",
    "ExtractionResult",
    "ExtractorExecutor",
    "SequentialExecutor",
    "ThreadPoolExecutor",
    "TimeoutExecutor",
    "AdaptiveExecutor",
]
