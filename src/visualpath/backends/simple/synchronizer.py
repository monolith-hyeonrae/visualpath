"""Observation synchronization strategies for SimpleBackend.

Synchronizers align observations from multiple extractors based on
timestamps, handling late arrivals and out-of-order data.

Available strategies:
- NoSync: No synchronization (process immediately)
- TimeWindowSync: Fixed time window alignment
- WatermarkSync: Watermark-based synchronization
- BarrierSync: Wait for all sources before proceeding
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.core.extractor import Observation


@dataclass
class SyncStats:
    """Statistics from observation synchronization.

    Attributes:
        windows_completed: Number of completed sync windows.
        observations_synced: Total observations synchronized.
        observations_dropped: Observations dropped (late arrival).
        avg_window_fill: Average observations per window.
        max_delay_ns: Maximum observed delay.
    """
    windows_completed: int = 0
    observations_synced: int = 0
    observations_dropped: int = 0
    total_observations_in_windows: int = 0
    max_delay_ns: int = 0

    @property
    def avg_window_fill(self) -> float:
        """Average observations per completed window."""
        if self.windows_completed == 0:
            return 0.0
        return self.total_observations_in_windows / self.windows_completed


@dataclass
class SyncWindow:
    """A time window containing observations from multiple sources.

    Attributes:
        window_start_ns: Window start timestamp.
        window_end_ns: Window end timestamp.
        observations: Observations grouped by source.
        sources_received: Set of sources that contributed.
    """
    window_start_ns: int
    window_end_ns: int
    observations: Dict[str, List["Observation"]] = field(default_factory=lambda: defaultdict(list))
    sources_received: Set[str] = field(default_factory=set)

    def add(self, observation: "Observation") -> None:
        """Add observation to window."""
        self.observations[observation.source].append(observation)
        self.sources_received.add(observation.source)

    def is_complete(self, required_sources: Set[str]) -> bool:
        """Check if all required sources have contributed."""
        return self.sources_received >= required_sources

    def all_observations(self) -> List["Observation"]:
        """Get all observations in window."""
        result = []
        for obs_list in self.observations.values():
            result.extend(obs_list)
        return result


class Synchronizer(ABC):
    """Abstract base class for observation synchronization.

    Synchronizers collect observations from multiple extractors and
    group them into synchronized windows for fusion processing.

    Example:
        >>> sync = TimeWindowSync(window_ns=100_000_000)
        >>> for obs in observations:
        ...     windows = sync.add(obs)
        ...     for window in windows:
        ...         fusion.process(window.all_observations())
    """

    @abstractmethod
    def add(self, observation: "Observation") -> List[SyncWindow]:
        """Add observation and return any completed windows.

        Args:
            observation: Observation to add.

        Returns:
            List of completed sync windows (may be empty).
        """
        ...

    @abstractmethod
    def flush(self) -> List[SyncWindow]:
        """Flush all pending windows.

        Call at end of stream to get remaining buffered data.

        Returns:
            List of incomplete windows.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset synchronizer state."""
        ...

    @property
    @abstractmethod
    def stats(self) -> SyncStats:
        """Get synchronization statistics."""
        ...


class NoSyncSynchronizer(Synchronizer):
    """Synchronizer that passes observations through immediately.

    No buffering or alignment - each observation creates its own
    single-observation window. Use when extractors are already
    synchronized or synchronization is not needed.

    Example:
        >>> sync = NoSyncSynchronizer()
        >>> windows = sync.add(obs)  # Returns immediately
        >>> assert len(windows) == 1
    """

    def __init__(self) -> None:
        self._stats = SyncStats()

    def add(self, observation: "Observation") -> List[SyncWindow]:
        """Create window for single observation."""
        window = SyncWindow(
            window_start_ns=observation.t_ns,
            window_end_ns=observation.t_ns,
        )
        window.add(observation)

        self._stats.windows_completed += 1
        self._stats.observations_synced += 1
        self._stats.total_observations_in_windows += 1

        return [window]

    def flush(self) -> List[SyncWindow]:
        """Nothing to flush."""
        return []

    def reset(self) -> None:
        self._stats = SyncStats()

    @property
    def stats(self) -> SyncStats:
        return self._stats


class TimeWindowSync(Synchronizer):
    """Synchronizer using fixed time windows.

    Groups observations into fixed-size time windows. A window is
    completed when the next window's observations start arriving
    or when the window timeout expires.

    Example:
        >>> sync = TimeWindowSync(
        ...     window_ns=100_000_000,  # 100ms windows
        ...     allowed_lateness_ns=50_000_000,  # 50ms late tolerance
        ... )
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,  # 100ms
        allowed_lateness_ns: int = 0,
        expected_sources: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the synchronizer.

        Args:
            window_ns: Window size in nanoseconds.
            allowed_lateness_ns: How late an observation can arrive.
            expected_sources: Optional set of expected source names.
        """
        self._window_ns = window_ns
        self._allowed_lateness_ns = allowed_lateness_ns
        self._expected_sources = expected_sources or set()
        self._windows: Dict[int, SyncWindow] = {}
        self._watermark_ns = 0
        self._stats = SyncStats()

    def _quantize(self, t_ns: int) -> int:
        """Quantize timestamp to window boundary."""
        return (t_ns // self._window_ns) * self._window_ns

    def add(self, observation: "Observation") -> List[SyncWindow]:
        """Add observation to appropriate window."""
        t_ns = observation.t_ns
        window_key = self._quantize(t_ns)

        # Check if too late
        if window_key < self._watermark_ns - self._allowed_lateness_ns:
            self._stats.observations_dropped += 1
            delay = self._watermark_ns - t_ns
            self._stats.max_delay_ns = max(self._stats.max_delay_ns, delay)
            return []

        # Get or create window
        if window_key not in self._windows:
            self._windows[window_key] = SyncWindow(
                window_start_ns=window_key,
                window_end_ns=window_key + self._window_ns,
            )

        self._windows[window_key].add(observation)
        self._stats.observations_synced += 1

        # Update watermark
        self._watermark_ns = max(self._watermark_ns, t_ns)

        # Check for completed windows
        completed = []
        cutoff = self._watermark_ns - self._window_ns - self._allowed_lateness_ns

        for key in sorted(self._windows.keys()):
            if key < cutoff:
                window = self._windows.pop(key)
                self._stats.windows_completed += 1
                self._stats.total_observations_in_windows += len(window.all_observations())
                completed.append(window)

        return completed

    def flush(self) -> List[SyncWindow]:
        """Flush all remaining windows."""
        windows = []
        for key in sorted(self._windows.keys()):
            window = self._windows[key]
            self._stats.windows_completed += 1
            self._stats.total_observations_in_windows += len(window.all_observations())
            windows.append(window)
        self._windows.clear()
        return windows

    def reset(self) -> None:
        self._windows.clear()
        self._watermark_ns = 0
        self._stats = SyncStats()

    @property
    def stats(self) -> SyncStats:
        return self._stats

    @property
    def current_watermark_ns(self) -> int:
        """Get current watermark timestamp."""
        return self._watermark_ns


class WatermarkSync(Synchronizer):
    """Synchronizer using watermark-based advancement.

    More sophisticated than TimeWindowSync - tracks watermarks per
    source and advances the global watermark based on the minimum
    across all sources. Better handles varying extractor speeds.

    Example:
        >>> sync = WatermarkSync(
        ...     window_ns=100_000_000,
        ...     sources=["face", "pose", "scene"],
        ... )
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,
        allowed_lateness_ns: int = 50_000_000,
        sources: Optional[List[str]] = None,
    ) -> None:
        """Initialize the synchronizer.

        Args:
            window_ns: Window size in nanoseconds.
            allowed_lateness_ns: Allowed late arrival tolerance.
            sources: List of expected source names.
        """
        self._window_ns = window_ns
        self._allowed_lateness_ns = allowed_lateness_ns
        self._sources = set(sources) if sources else set()
        self._source_watermarks: Dict[str, int] = {}
        self._windows: Dict[int, SyncWindow] = {}
        self._global_watermark = 0
        self._stats = SyncStats()

    def _quantize(self, t_ns: int) -> int:
        """Quantize timestamp to window boundary."""
        return (t_ns // self._window_ns) * self._window_ns

    def _update_watermarks(self, source: str, t_ns: int) -> None:
        """Update source and global watermarks."""
        # Update source watermark
        current = self._source_watermarks.get(source, 0)
        self._source_watermarks[source] = max(current, t_ns)

        # Update global watermark as min of all source watermarks
        if self._sources:
            known_sources = self._sources & set(self._source_watermarks.keys())
            if known_sources == self._sources:
                self._global_watermark = min(
                    self._source_watermarks[s] for s in known_sources
                )
        else:
            # No expected sources - use min of seen sources
            if self._source_watermarks:
                self._global_watermark = min(self._source_watermarks.values())

    def add(self, observation: "Observation") -> List[SyncWindow]:
        """Add observation and check for completed windows."""
        t_ns = observation.t_ns
        source = observation.source
        window_key = self._quantize(t_ns)

        # Track source
        self._sources.add(source)

        # Check if too late
        if t_ns < self._global_watermark - self._allowed_lateness_ns:
            self._stats.observations_dropped += 1
            delay = self._global_watermark - t_ns
            self._stats.max_delay_ns = max(self._stats.max_delay_ns, delay)
            return []

        # Get or create window
        if window_key not in self._windows:
            self._windows[window_key] = SyncWindow(
                window_start_ns=window_key,
                window_end_ns=window_key + self._window_ns,
            )

        self._windows[window_key].add(observation)
        self._stats.observations_synced += 1

        # Update watermarks
        self._update_watermarks(source, t_ns)

        # Emit windows that are below watermark
        completed = []
        cutoff = self._global_watermark - self._allowed_lateness_ns

        for key in sorted(self._windows.keys()):
            if key + self._window_ns < cutoff:
                window = self._windows.pop(key)
                self._stats.windows_completed += 1
                self._stats.total_observations_in_windows += len(window.all_observations())
                completed.append(window)

        return completed

    def flush(self) -> List[SyncWindow]:
        """Flush all remaining windows."""
        windows = []
        for key in sorted(self._windows.keys()):
            window = self._windows[key]
            self._stats.windows_completed += 1
            self._stats.total_observations_in_windows += len(window.all_observations())
            windows.append(window)
        self._windows.clear()
        return windows

    def reset(self) -> None:
        self._windows.clear()
        self._source_watermarks.clear()
        self._global_watermark = 0
        self._stats = SyncStats()

    @property
    def stats(self) -> SyncStats:
        return self._stats

    @property
    def global_watermark_ns(self) -> int:
        """Get global watermark timestamp."""
        return self._global_watermark

    @property
    def source_watermarks(self) -> Dict[str, int]:
        """Get per-source watermarks."""
        return dict(self._source_watermarks)


class BarrierSync(Synchronizer):
    """Synchronizer that waits for all sources per frame.

    Buffers observations until all expected sources have contributed
    for a given frame_id, then emits the complete set. Use when you
    need all extractors' results for every frame.

    Example:
        >>> sync = BarrierSync(sources=["face", "pose"])
        >>> # Won't emit until both face and pose observations arrive
    """

    def __init__(
        self,
        sources: List[str],
        timeout_ns: int = 200_000_000,  # 200ms
    ) -> None:
        """Initialize the synchronizer.

        Args:
            sources: List of required source names.
            timeout_ns: How long to wait for all sources.
        """
        if not sources:
            raise ValueError("sources must not be empty")
        self._sources = set(sources)
        self._timeout_ns = timeout_ns
        self._frame_buffers: Dict[int, SyncWindow] = {}
        self._frame_start_times: Dict[int, int] = {}
        self._latest_t_ns = 0
        self._stats = SyncStats()

    def add(self, observation: "Observation") -> List[SyncWindow]:
        """Add observation and check for complete frames."""
        frame_id = observation.frame_id
        t_ns = observation.t_ns
        self._latest_t_ns = max(self._latest_t_ns, t_ns)

        # Get or create frame buffer
        if frame_id not in self._frame_buffers:
            self._frame_buffers[frame_id] = SyncWindow(
                window_start_ns=t_ns,
                window_end_ns=t_ns,
            )
            self._frame_start_times[frame_id] = t_ns

        self._frame_buffers[frame_id].add(observation)
        self._stats.observations_synced += 1

        # Check for completed frames
        completed = []

        for fid in list(self._frame_buffers.keys()):
            buffer = self._frame_buffers[fid]
            start_time = self._frame_start_times[fid]

            # Check if complete
            if buffer.is_complete(self._sources):
                self._frame_buffers.pop(fid)
                self._frame_start_times.pop(fid)
                self._stats.windows_completed += 1
                self._stats.total_observations_in_windows += len(buffer.all_observations())
                completed.append(buffer)
            # Check if timed out
            elif self._latest_t_ns - start_time > self._timeout_ns:
                self._frame_buffers.pop(fid)
                self._frame_start_times.pop(fid)
                missing = self._sources - buffer.sources_received
                # Emit partial (could also choose to drop)
                self._stats.windows_completed += 1
                self._stats.total_observations_in_windows += len(buffer.all_observations())
                completed.append(buffer)

        return completed

    def flush(self) -> List[SyncWindow]:
        """Flush all remaining frame buffers."""
        windows = []
        for fid in sorted(self._frame_buffers.keys()):
            buffer = self._frame_buffers[fid]
            self._stats.windows_completed += 1
            self._stats.total_observations_in_windows += len(buffer.all_observations())
            windows.append(buffer)
        self._frame_buffers.clear()
        self._frame_start_times.clear()
        return windows

    def reset(self) -> None:
        self._frame_buffers.clear()
        self._frame_start_times.clear()
        self._latest_t_ns = 0
        self._stats = SyncStats()

    @property
    def stats(self) -> SyncStats:
        return self._stats


__all__ = [
    "SyncStats",
    "SyncWindow",
    "Synchronizer",
    "NoSyncSynchronizer",
    "TimeWindowSync",
    "WatermarkSync",
    "BarrierSync",
]
