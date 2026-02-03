"""Statistics collection for PathwayBackend.

PathwayStats provides thread-safe counters and timing metrics for
monitoring Pathway pipeline execution. No Pathway dependency required.

Example:
    >>> stats = PathwayStats()
    >>> stats.record_ingestion()
    >>> stats.record_extraction("face", 12.5, success=True)
    >>> print(stats.throughput_fps)
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathwayStats:
    """Thread-safe statistics for PathwayBackend pipeline execution.

    All public mutation methods acquire an internal lock, making this
    class safe for concurrent use from Pathway's connector thread,
    UDF threads, and subscribe callbacks.
    """

    # Frame counters
    frames_ingested: int = 0
    frames_extracted: int = 0

    # Extraction counters
    extractions_completed: int = 0
    extractions_failed: int = 0

    # Trigger / observation output counters
    triggers_fired: int = 0
    observations_output: int = 0

    # Timing (milliseconds)
    total_extraction_ms: float = 0.0

    # Per-extractor EMA times (name -> ema_ms)
    per_extractor_time_ms: Dict[str, float] = field(default_factory=dict)

    # Raw extraction times for percentile calculation
    _extraction_times: List[float] = field(default_factory=list, repr=False)

    # Per-extractor raw times for EMA
    _per_extractor_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int), repr=False,
    )

    # Pipeline wall-clock (nanoseconds, monotonic)
    pipeline_start_ns: int = 0
    pipeline_end_ns: int = 0

    # Internal lock
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # EMA smoothing factor
    _ema_alpha: float = field(default=0.3, repr=False)

    # --- Mutation methods ------------------------------------------------

    def record_ingestion(self) -> None:
        """Record a frame ingested via ConnectorSubject."""
        with self._lock:
            self.frames_ingested += 1

    def record_extraction(
        self,
        extractor_name: str,
        elapsed_ms: float,
        success: bool = True,
    ) -> None:
        """Record a single extractor invocation.

        Args:
            extractor_name: Name of the extractor.
            elapsed_ms: Wall-clock time in milliseconds.
            success: Whether the extraction succeeded.
        """
        with self._lock:
            if success:
                self.extractions_completed += 1
            else:
                self.extractions_failed += 1

            self.total_extraction_ms += elapsed_ms
            self._extraction_times.append(elapsed_ms)

            # EMA per extractor
            self._per_extractor_counts[extractor_name] += 1
            prev = self.per_extractor_time_ms.get(extractor_name)
            if prev is None:
                self.per_extractor_time_ms[extractor_name] = elapsed_ms
            else:
                alpha = self._ema_alpha
                self.per_extractor_time_ms[extractor_name] = (
                    alpha * elapsed_ms + (1 - alpha) * prev
                )

    def record_frame_extracted(self) -> None:
        """Record that all extractors finished for one frame."""
        with self._lock:
            self.frames_extracted += 1

    def record_trigger(self) -> None:
        """Record a trigger fired by fusion."""
        with self._lock:
            self.triggers_fired += 1

    def record_observation_output(self) -> None:
        """Record an observation received in subscribe callback."""
        with self._lock:
            self.observations_output += 1

    def mark_pipeline_start(self) -> None:
        """Record pipeline start time."""
        with self._lock:
            self.pipeline_start_ns = time.perf_counter_ns()

    def mark_pipeline_end(self) -> None:
        """Record pipeline end time."""
        with self._lock:
            self.pipeline_end_ns = time.perf_counter_ns()

    def reset(self) -> None:
        """Reset all counters and timings."""
        with self._lock:
            self.frames_ingested = 0
            self.frames_extracted = 0
            self.extractions_completed = 0
            self.extractions_failed = 0
            self.triggers_fired = 0
            self.observations_output = 0
            self.total_extraction_ms = 0.0
            self.per_extractor_time_ms.clear()
            self._extraction_times.clear()
            self._per_extractor_counts.clear()
            self.pipeline_start_ns = 0
            self.pipeline_end_ns = 0

    # --- Computed properties ---------------------------------------------

    def _pipeline_duration_sec_unlocked(self) -> float:
        """Pipeline duration without acquiring lock. Caller must hold lock."""
        start = self.pipeline_start_ns
        end = self.pipeline_end_ns
        if start == 0 or end == 0:
            return 0.0
        return (end - start) / 1_000_000_000

    def _avg_extraction_ms_unlocked(self) -> float:
        """Average extraction time without lock. Caller must hold lock."""
        n = len(self._extraction_times)
        if n == 0:
            return 0.0
        return self.total_extraction_ms / n

    def _p95_extraction_ms_unlocked(self) -> float:
        """P95 extraction time without lock. Caller must hold lock."""
        times = sorted(self._extraction_times)
        if not times:
            return 0.0
        idx = int(len(times) * 0.95)
        idx = min(idx, len(times) - 1)
        return times[idx]

    def _throughput_fps_unlocked(self) -> float:
        """Throughput FPS without lock. Caller must hold lock."""
        duration = self._pipeline_duration_sec_unlocked()
        if duration <= 0:
            return 0.0
        return self.frames_extracted / duration

    @property
    def throughput_fps(self) -> float:
        """Frames extracted per second based on pipeline wall-clock."""
        with self._lock:
            return self._throughput_fps_unlocked()

    @property
    def avg_extraction_ms(self) -> float:
        """Average extraction time across all invocations."""
        with self._lock:
            return self._avg_extraction_ms_unlocked()

    @property
    def p95_extraction_ms(self) -> float:
        """95th-percentile extraction time."""
        with self._lock:
            return self._p95_extraction_ms_unlocked()

    @property
    def pipeline_duration_sec(self) -> float:
        """Pipeline wall-clock duration in seconds."""
        with self._lock:
            return self._pipeline_duration_sec_unlocked()

    # --- Serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        """Snapshot of all stats as a plain dict.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        with self._lock:
            return {
                "frames_ingested": self.frames_ingested,
                "frames_extracted": self.frames_extracted,
                "extractions_completed": self.extractions_completed,
                "extractions_failed": self.extractions_failed,
                "triggers_fired": self.triggers_fired,
                "observations_output": self.observations_output,
                "total_extraction_ms": self.total_extraction_ms,
                "per_extractor_time_ms": dict(self.per_extractor_time_ms),
                "pipeline_start_ns": self.pipeline_start_ns,
                "pipeline_end_ns": self.pipeline_end_ns,
                "throughput_fps": self._throughput_fps_unlocked(),
                "avg_extraction_ms": self._avg_extraction_ms_unlocked(),
                "p95_extraction_ms": self._p95_extraction_ms_unlocked(),
                "pipeline_duration_sec": self._pipeline_duration_sec_unlocked(),
            }


__all__ = ["PathwayStats"]
