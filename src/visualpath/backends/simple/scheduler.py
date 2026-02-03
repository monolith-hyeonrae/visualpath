"""Frame scheduling strategies for SimpleBackend.

Schedulers control how frames are selected for processing when the
pipeline cannot keep up with the input rate.

Available strategies:
- PassThrough: Process all frames (no dropping)
- SkipOldest: Drop oldest frames when buffer is full
- SkipIntermediate: Keep only oldest and newest
- KeyframeOnly: Process every Nth frame
- AdaptiveRate: Dynamically adjust based on processing speed
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame


@dataclass
class SchedulerStats:
    """Statistics from frame scheduling.

    Attributes:
        frames_received: Total frames received.
        frames_processed: Frames passed to processing.
        frames_dropped: Frames dropped by scheduler.
        drop_rate: Current drop rate (0.0 - 1.0).
    """
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0

    @property
    def drop_rate(self) -> float:
        """Calculate current drop rate."""
        if self.frames_received == 0:
            return 0.0
        return self.frames_dropped / self.frames_received


class FrameScheduler(ABC):
    """Abstract base class for frame scheduling strategies.

    FrameSchedulers decide which frames to process when the pipeline
    cannot keep up with the input rate. Different strategies offer
    different trade-offs between latency, coverage, and accuracy.

    Example:
        >>> scheduler = SkipOldestScheduler(max_queue=10)
        >>> for frame in scheduler.schedule(frames):
        ...     process(frame)
    """

    @abstractmethod
    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Schedule frames for processing.

        Args:
            frames: Input frame iterator.

        Yields:
            Frames selected for processing.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset scheduler state."""
        ...

    @property
    @abstractmethod
    def stats(self) -> SchedulerStats:
        """Get scheduling statistics."""
        ...


class PassThroughScheduler(FrameScheduler):
    """Scheduler that processes all frames without dropping.

    This is the default scheduler when no backpressure is needed.
    Use when processing is guaranteed to keep up with input rate.

    Example:
        >>> scheduler = PassThroughScheduler()
        >>> for frame in scheduler.schedule(video.stream()):
        ...     process(frame)  # All frames processed
    """

    def __init__(self) -> None:
        self._stats = SchedulerStats()

    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Pass through all frames."""
        for frame in frames:
            self._stats.frames_received += 1
            self._stats.frames_processed += 1
            yield frame

    def reset(self) -> None:
        """Reset statistics."""
        self._stats = SchedulerStats()

    @property
    def stats(self) -> SchedulerStats:
        return self._stats


class SkipOldestScheduler(FrameScheduler):
    """Scheduler that drops oldest frames when queue is full.

    Maintains a bounded queue and drops the oldest frame when a
    new frame arrives and the queue is full. This minimizes latency
    at the cost of potentially missing events.

    Example:
        >>> scheduler = SkipOldestScheduler(max_queue=5)
        >>> # If 6 frames arrive before processing starts,
        >>> # the oldest frame is dropped
    """

    def __init__(self, max_queue: int = 10) -> None:
        """Initialize the scheduler.

        Args:
            max_queue: Maximum frames to buffer before dropping.
        """
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")
        self._max_queue = max_queue
        self._stats = SchedulerStats()

    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Schedule frames, dropping oldest when queue overflows."""
        queue: Deque["Frame"] = deque(maxlen=self._max_queue)

        for frame in frames:
            self._stats.frames_received += 1

            if len(queue) >= self._max_queue:
                # Queue full - will drop oldest on append
                self._stats.frames_dropped += 1

            queue.append(frame)

            # Yield from queue
            while queue:
                self._stats.frames_processed += 1
                yield queue.popleft()

    def reset(self) -> None:
        self._stats = SchedulerStats()

    @property
    def stats(self) -> SchedulerStats:
        return self._stats


class SkipIntermediateScheduler(FrameScheduler):
    """Scheduler that keeps only boundary frames.

    When buffer overflows, keeps only the oldest and newest frames,
    dropping intermediate frames. Useful when you need to track
    state transitions but can skip intermediate states.

    Example:
        >>> scheduler = SkipIntermediateScheduler(max_queue=5)
        >>> # If many frames arrive, keeps first and last
    """

    def __init__(self, max_queue: int = 10) -> None:
        """Initialize the scheduler.

        Args:
            max_queue: Maximum frames to buffer before dropping intermediates.
        """
        if max_queue < 2:
            raise ValueError("max_queue must be at least 2")
        self._max_queue = max_queue
        self._stats = SchedulerStats()

    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Schedule frames, keeping only boundaries when full."""
        queue: Deque["Frame"] = deque()
        oldest: Optional["Frame"] = None

        for frame in frames:
            self._stats.frames_received += 1

            if oldest is None:
                oldest = frame
                continue

            if len(queue) >= self._max_queue - 1:
                # Keep oldest, drop intermediates, keep newest
                dropped = len(queue) - 1
                self._stats.frames_dropped += dropped
                queue.clear()

            queue.append(frame)

        # Yield oldest first
        if oldest is not None:
            self._stats.frames_processed += 1
            yield oldest

        # Yield remaining
        for frame in queue:
            self._stats.frames_processed += 1
            yield frame

    def reset(self) -> None:
        self._stats = SchedulerStats()

    @property
    def stats(self) -> SchedulerStats:
        return self._stats


class KeyframeScheduler(FrameScheduler):
    """Scheduler that processes every Nth frame.

    Simple decimation strategy that processes frames at a fixed
    interval. Provides predictable throughput but may miss events
    between keyframes.

    Example:
        >>> scheduler = KeyframeScheduler(every_n=3)
        >>> # Processes frames 0, 3, 6, 9, ...
    """

    def __init__(self, every_n: int = 3) -> None:
        """Initialize the scheduler.

        Args:
            every_n: Process every Nth frame (1 = all frames).
        """
        if every_n < 1:
            raise ValueError("every_n must be at least 1")
        self._every_n = every_n
        self._stats = SchedulerStats()

    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Schedule every Nth frame."""
        for i, frame in enumerate(frames):
            self._stats.frames_received += 1

            if i % self._every_n == 0:
                self._stats.frames_processed += 1
                yield frame
            else:
                self._stats.frames_dropped += 1

    def reset(self) -> None:
        self._stats = SchedulerStats()

    @property
    def stats(self) -> SchedulerStats:
        return self._stats


class AdaptiveRateScheduler(FrameScheduler):
    """Scheduler that adapts drop rate based on processing speed.

    Monitors processing time and adjusts frame rate dynamically
    to maintain target throughput. Increases skip rate when falling
    behind, decreases when catching up.

    Example:
        >>> scheduler = AdaptiveRateScheduler(target_fps=10)
        >>> # Automatically adjusts to maintain 10 FPS output
    """

    def __init__(
        self,
        target_fps: float = 10.0,
        min_skip: int = 1,
        max_skip: int = 10,
    ) -> None:
        """Initialize the scheduler.

        Args:
            target_fps: Target output frames per second.
            min_skip: Minimum frames to skip (1 = no skip).
            max_skip: Maximum frames to skip.
        """
        self._target_fps = target_fps
        self._target_interval_ns = int(1e9 / target_fps)
        self._min_skip = min_skip
        self._max_skip = max_skip
        self._current_skip = min_skip
        self._last_process_ns = 0
        self._stats = SchedulerStats()

    def schedule(self, frames: Iterator["Frame"]) -> Iterator["Frame"]:
        """Schedule frames with adaptive rate."""
        import time

        skip_counter = 0

        for frame in frames:
            self._stats.frames_received += 1
            skip_counter += 1

            if skip_counter >= self._current_skip:
                skip_counter = 0

                # Measure processing time
                start_ns = time.perf_counter_ns()

                self._stats.frames_processed += 1
                yield frame

                # Adjust skip rate based on actual timing
                if self._last_process_ns > 0:
                    actual_interval = start_ns - self._last_process_ns
                    if actual_interval < self._target_interval_ns * 0.8:
                        # Processing too fast, can reduce skip
                        self._current_skip = max(
                            self._min_skip,
                            self._current_skip - 1
                        )
                    elif actual_interval > self._target_interval_ns * 1.2:
                        # Processing too slow, increase skip
                        self._current_skip = min(
                            self._max_skip,
                            self._current_skip + 1
                        )

                self._last_process_ns = start_ns
            else:
                self._stats.frames_dropped += 1

    def reset(self) -> None:
        self._stats = SchedulerStats()
        self._current_skip = self._min_skip
        self._last_process_ns = 0

    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    @property
    def current_skip_rate(self) -> int:
        """Get current skip rate."""
        return self._current_skip


__all__ = [
    "SchedulerStats",
    "FrameScheduler",
    "PassThroughScheduler",
    "SkipOldestScheduler",
    "SkipIntermediateScheduler",
    "KeyframeScheduler",
    "AdaptiveRateScheduler",
]
