"""ExtractorOrchestrator - Thread-parallel extractor execution.

Provides thread-based parallelism for running multiple extractors
simultaneously on the same frame. Useful for Library mode where
extractors run in the same process.

Architecture:
    Frame ──→ [Thread Pool] ──→ Observations
                  │
                  ├── Extractor1 (thread 1)
                  ├── Extractor2 (thread 2)
                  └── Extractor3 (thread 3)

Example:
    >>> from visualpath.process import ExtractorOrchestrator
    >>>
    >>> orchestrator = ExtractorOrchestrator(
    ...     extractors=[ext1, ext2, ext3],
    ...     max_workers=3,
    ... )
    >>> orchestrator.initialize()
    >>> for frame in frames:
    ...     observations = orchestrator.extract_all(frame)
    ...     for obs in observations:
    ...         process(obs)
    >>> orchestrator.cleanup()
"""

import time
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError

from visualbase import Frame

from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.observability import ObservabilityHub

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


class ExtractorOrchestrator:
    """Orchestrates parallel execution of multiple extractors.

    Manages a pool of extractors and runs them in parallel using threads.
    Collects observations and provides unified access to results.

    Args:
        extractors: List of extractors to run in parallel.
        max_workers: Maximum number of worker threads. Default: len(extractors).
        timeout: Timeout in seconds for each extraction. Default: 5.0.
        observability_hub: Optional custom observability hub (uses global if None).

    Thread Safety:
        - Each extractor runs in its own thread
        - Observations are collected thread-safely
        - Initialize/cleanup are called from the main thread
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        max_workers: Optional[int] = None,
        timeout: float = 5.0,
        observability_hub: Optional[ObservabilityHub] = None,
    ):
        if not extractors:
            raise ValueError("At least one extractor is required")

        self._extractors = extractors
        self._max_workers = max_workers or len(extractors)
        self._timeout = timeout
        self._hub = observability_hub or _hub

        self._executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False

        # Stats
        self._frames_processed = 0
        self._total_observations = 0
        self._timeouts = 0
        self._errors = 0
        self._total_time_ns = 0

    def initialize(self) -> None:
        """Initialize all extractors and create thread pool.

        Must be called before extract_all().
        """
        if self._initialized:
            return

        # Initialize all extractors
        for ext in self._extractors:
            try:
                ext.initialize()
                logger.debug(f"Initialized extractor: {ext.name}")
            except Exception as e:
                logger.error(f"Failed to initialize {ext.name}: {e}")
                raise

        # Create thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="extractor_",
        )

        self._initialized = True
        logger.info(
            f"ExtractorOrchestrator initialized with {len(self._extractors)} extractors, "
            f"{self._max_workers} workers"
        )

    def extract_all(self, frame: Frame) -> List[Observation]:
        """Run all extractors on a frame in parallel.

        Args:
            frame: Frame to process.

        Returns:
            List of observations from all extractors.
            May be fewer than number of extractors if some return None.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized or self._executor is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        start_ns = time.perf_counter_ns()
        observations: List[Observation] = []
        timed_out_extractors: List[str] = []

        # Submit all extractors
        futures: Dict[Future, BaseExtractor] = {}
        for ext in self._extractors:
            future = self._executor.submit(self._safe_extract, ext, frame)
            futures[future] = ext

        # Collect results with timeout
        try:
            for future in as_completed(futures, timeout=self._timeout):
                ext = futures[future]
                try:
                    obs = future.result()
                    if obs is not None:
                        observations.append(obs)
                        self._total_observations += 1
                except TimeoutError:
                    logger.warning(f"Extractor {ext.name} timed out")
                    self._timeouts += 1
                    timed_out_extractors.append(ext.name)
                except Exception as e:
                    logger.error(f"Extractor {ext.name} error: {e}")
                    self._errors += 1
        except TimeoutError:
            # Some futures didn't complete in time
            for future, ext in futures.items():
                if not future.done():
                    timed_out_extractors.append(ext.name)
                    self._timeouts += 1
                    logger.warning(f"Extractor {ext.name} timed out")

        self._frames_processed += 1
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._total_time_ns += elapsed_ns

        # Emit observability records
        if self._hub.enabled:
            processing_ms = elapsed_ns / 1_000_000
            self._emit_timing(frame.frame_id, processing_ms)
            if timed_out_extractors:
                self._emit_timeout(frame.frame_id, timed_out_extractors)

        return observations

    def _emit_timing(self, frame_id: int, processing_ms: float) -> None:
        """Emit timing record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="timing",
            frame_id=frame_id,
            data={
                "component": "orchestrator",
                "processing_ms": processing_ms,
                "threshold_ms": self._timeout * 1000,
                "is_slow": processing_ms > self._timeout * 1000,
            },
        ))

    def _emit_timeout(self, frame_id: int, timed_out: List[str]) -> None:
        """Emit timeout/frame drop record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="frame_drop",
            frame_id=frame_id,
            data={
                "dropped_frame_ids": [frame_id],
                "reason": f"timeout:{','.join(timed_out)}",
            },
        ))

    def _safe_extract(
        self, extractor: BaseExtractor, frame: Frame
    ) -> Optional[Observation]:
        """Safely run extraction with error handling.

        Args:
            extractor: Extractor to run.
            frame: Frame to process.

        Returns:
            Observation or None on error.
        """
        try:
            return extractor.extract(frame)
        except Exception as e:
            logger.error(f"Extract error in {extractor.name}: {e}")
            self._errors += 1
            return None

    def extract_sequential(self, frame: Frame) -> List[Observation]:
        """Run all extractors sequentially (no parallelism).

        Useful for debugging or when parallelism is not needed.

        Args:
            frame: Frame to process.

        Returns:
            List of observations from all extractors.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        start_ns = time.perf_counter_ns()
        observations: List[Observation] = []

        for ext in self._extractors:
            try:
                obs = ext.extract(frame)
                if obs is not None:
                    observations.append(obs)
                    self._total_observations += 1
            except Exception as e:
                logger.error(f"Extract error in {ext.name}: {e}")
                self._errors += 1

        self._frames_processed += 1
        self._total_time_ns += time.perf_counter_ns() - start_ns

        return observations

    def cleanup(self) -> None:
        """Clean up all extractors and shutdown thread pool."""
        if not self._initialized:
            return

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Cleanup extractors
        for ext in self._extractors:
            try:
                ext.cleanup()
                logger.debug(f"Cleaned up extractor: {ext.name}")
            except Exception as e:
                logger.error(f"Failed to cleanup {ext.name}: {e}")

        self._initialized = False
        logger.info("ExtractorOrchestrator shut down")

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Dict with processing statistics.
        """
        avg_time_ms = (
            (self._total_time_ns / self._frames_processed / 1_000_000)
            if self._frames_processed > 0
            else 0
        )
        return {
            "frames_processed": self._frames_processed,
            "total_observations": self._total_observations,
            "timeouts": self._timeouts,
            "errors": self._errors,
            "avg_time_ms": avg_time_ms,
            "extractors": [ext.name for ext in self._extractors],
            "max_workers": self._max_workers,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized

    @property
    def extractor_names(self) -> List[str]:
        """Get names of all extractors."""
        return [ext.name for ext in self._extractors]

    def __enter__(self) -> "ExtractorOrchestrator":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
