"""Backpressure buffer management for SimpleBackend.

Buffers control the flow of data between pipeline stages,
preventing memory overflow when producers outpace consumers.

Available strategies:
- UnboundedBuffer: No limit (use with caution)
- BoundedBuffer: Fixed size with configurable overflow policy
- SlidingWindowBuffer: Time-based window
- PriorityBuffer: Priority-based eviction
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from heapq import heappush, heappop
from threading import Lock
from typing import Callable, Deque, Generic, List, Optional, TypeVar

T = TypeVar("T")


class OverflowPolicy(Enum):
    """Policy for handling buffer overflow.

    DROP_OLDEST: Remove oldest item when full.
    DROP_NEWEST: Reject new item when full.
    BLOCK: Block producer until space available.
    RAISE: Raise exception when full.
    """
    DROP_OLDEST = auto()
    DROP_NEWEST = auto()
    BLOCK = auto()
    RAISE = auto()


@dataclass
class BufferStats:
    """Statistics from buffer operations.

    Attributes:
        items_added: Total items added.
        items_removed: Total items removed.
        items_dropped: Items dropped due to overflow.
        peak_size: Maximum buffer size observed.
        current_size: Current number of items.
    """
    items_added: int = 0
    items_removed: int = 0
    items_dropped: int = 0
    peak_size: int = 0
    current_size: int = 0

    @property
    def throughput(self) -> int:
        """Items successfully processed."""
        return self.items_removed


class BackpressureBuffer(ABC, Generic[T]):
    """Abstract base class for backpressure buffers.

    Buffers manage the flow of items between producer and consumer,
    providing backpressure when the consumer cannot keep up.

    Example:
        >>> buffer = BoundedBuffer(max_size=100)
        >>> buffer.put(item)
        >>> item = buffer.get()
    """

    @abstractmethod
    def put(self, item: T) -> bool:
        """Add item to buffer.

        Args:
            item: Item to add.

        Returns:
            True if item was added, False if dropped.
        """
        ...

    @abstractmethod
    def get(self) -> Optional[T]:
        """Remove and return item from buffer.

        Returns:
            Item or None if buffer is empty.
        """
        ...

    @abstractmethod
    def peek(self) -> Optional[T]:
        """Return next item without removing.

        Returns:
            Item or None if buffer is empty.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from buffer."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Current number of items in buffer."""
        ...

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Whether buffer is empty."""
        ...

    @property
    @abstractmethod
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        ...

    @property
    @abstractmethod
    def stats(self) -> BufferStats:
        """Get buffer statistics."""
        ...


class UnboundedBuffer(BackpressureBuffer[T]):
    """Buffer with no size limit.

    Use with caution - can cause memory issues if producer
    significantly outpaces consumer.

    Example:
        >>> buffer = UnboundedBuffer()
        >>> for item in items:
        ...     buffer.put(item)  # Never blocks or drops
    """

    def __init__(self) -> None:
        self._queue: Deque[T] = deque()
        self._stats = BufferStats()
        self._lock = Lock()

    def put(self, item: T) -> bool:
        """Add item to buffer."""
        with self._lock:
            self._queue.append(item)
            self._stats.items_added += 1
            self._stats.current_size = len(self._queue)
            self._stats.peak_size = max(self._stats.peak_size, self._stats.current_size)
        return True

    def get(self) -> Optional[T]:
        """Remove and return oldest item."""
        with self._lock:
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._stats.items_removed += 1
            self._stats.current_size = len(self._queue)
            return item

    def peek(self) -> Optional[T]:
        """Return oldest item without removing."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._queue.clear()
            self._stats.current_size = 0

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return False  # Never full

    @property
    def stats(self) -> BufferStats:
        return self._stats


class BoundedBuffer(BackpressureBuffer[T]):
    """Buffer with fixed maximum size.

    Applies overflow policy when buffer is full. Thread-safe.

    Example:
        >>> buffer = BoundedBuffer(
        ...     max_size=100,
        ...     overflow_policy=OverflowPolicy.DROP_OLDEST,
        ... )
    """

    def __init__(
        self,
        max_size: int = 100,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        """Initialize the buffer.

        Args:
            max_size: Maximum number of items.
            overflow_policy: How to handle overflow.
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._policy = overflow_policy
        self._queue: Deque[T] = deque()
        self._stats = BufferStats()
        self._lock = Lock()

    def put(self, item: T) -> bool:
        """Add item to buffer, applying overflow policy if full."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                if self._policy == OverflowPolicy.DROP_OLDEST:
                    self._queue.popleft()
                    self._stats.items_dropped += 1
                elif self._policy == OverflowPolicy.DROP_NEWEST:
                    self._stats.items_dropped += 1
                    return False
                elif self._policy == OverflowPolicy.RAISE:
                    raise BufferOverflowError(
                        f"Buffer full (max_size={self._max_size})"
                    )
                # BLOCK not implemented in simple version

            self._queue.append(item)
            self._stats.items_added += 1
            self._stats.current_size = len(self._queue)
            self._stats.peak_size = max(self._stats.peak_size, self._stats.current_size)
            return True

    def get(self) -> Optional[T]:
        """Remove and return oldest item."""
        with self._lock:
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._stats.items_removed += 1
            self._stats.current_size = len(self._queue)
            return item

    def peek(self) -> Optional[T]:
        """Return oldest item without removing."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._queue.clear()
            self._stats.current_size = 0

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self._max_size

    @property
    def stats(self) -> BufferStats:
        return self._stats

    @property
    def max_size(self) -> int:
        """Maximum buffer size."""
        return self._max_size


class SlidingWindowBuffer(BackpressureBuffer[T]):
    """Buffer that keeps items within a time window.

    Items older than the window are automatically evicted.
    Requires items to have a timestamp attribute or extractor.

    Example:
        >>> buffer = SlidingWindowBuffer(
        ...     window_ns=100_000_000,  # 100ms
        ...     timestamp_fn=lambda x: x.t_ns,
        ... )
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,
        timestamp_fn: Callable[[T], int] = None,
        max_size: int = 1000,
    ) -> None:
        """Initialize the buffer.

        Args:
            window_ns: Time window in nanoseconds.
            timestamp_fn: Function to extract timestamp from item.
            max_size: Maximum items (hard limit).
        """
        self._window_ns = window_ns
        self._timestamp_fn = timestamp_fn or (lambda x: getattr(x, 't_ns', 0))
        self._max_size = max_size
        self._queue: Deque[T] = deque()
        self._stats = BufferStats()
        self._lock = Lock()
        self._latest_ts = 0

    def _evict_old(self) -> None:
        """Remove items outside the time window."""
        cutoff = self._latest_ts - self._window_ns
        while self._queue:
            oldest = self._queue[0]
            if self._timestamp_fn(oldest) < cutoff:
                self._queue.popleft()
                self._stats.items_dropped += 1
            else:
                break

    def put(self, item: T) -> bool:
        """Add item to buffer."""
        with self._lock:
            ts = self._timestamp_fn(item)
            self._latest_ts = max(self._latest_ts, ts)

            # Evict old items
            self._evict_old()

            # Check hard limit
            if len(self._queue) >= self._max_size:
                self._queue.popleft()
                self._stats.items_dropped += 1

            self._queue.append(item)
            self._stats.items_added += 1
            self._stats.current_size = len(self._queue)
            self._stats.peak_size = max(self._stats.peak_size, self._stats.current_size)
            return True

    def get(self) -> Optional[T]:
        """Remove and return oldest item."""
        with self._lock:
            self._evict_old()
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._stats.items_removed += 1
            self._stats.current_size = len(self._queue)
            return item

    def peek(self) -> Optional[T]:
        """Return oldest item without removing."""
        with self._lock:
            self._evict_old()
            if not self._queue:
                return None
            return self._queue[0]

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._queue.clear()
            self._stats.current_size = 0

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self._max_size

    @property
    def stats(self) -> BufferStats:
        return self._stats


class PriorityBuffer(BackpressureBuffer[T]):
    """Buffer that evicts lowest-priority items first.

    When full, removes the item with lowest priority. Useful
    when some frames are more important than others.

    Example:
        >>> buffer = PriorityBuffer(
        ...     max_size=50,
        ...     priority_fn=lambda x: x.score,  # Higher score = higher priority
        ... )
    """

    def __init__(
        self,
        max_size: int = 100,
        priority_fn: Callable[[T], float] = None,
    ) -> None:
        """Initialize the buffer.

        Args:
            max_size: Maximum number of items.
            priority_fn: Function to extract priority (higher = more important).
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._priority_fn = priority_fn or (lambda x: 0.0)
        # Min-heap (negate priority for max behavior)
        self._heap: List[tuple] = []
        self._counter = 0  # For tie-breaking
        self._stats = BufferStats()
        self._lock = Lock()

    def put(self, item: T) -> bool:
        """Add item, evicting lowest priority if full."""
        with self._lock:
            priority = self._priority_fn(item)

            if len(self._heap) >= self._max_size:
                # Check if new item has higher priority than lowest
                if self._heap and -self._heap[0][0] < priority:
                    heappop(self._heap)
                    self._stats.items_dropped += 1
                else:
                    # New item has lowest priority, drop it
                    self._stats.items_dropped += 1
                    return False

            # Add with negated priority (min-heap -> max behavior)
            heappush(self._heap, (-priority, self._counter, item))
            self._counter += 1
            self._stats.items_added += 1
            self._stats.current_size = len(self._heap)
            self._stats.peak_size = max(self._stats.peak_size, self._stats.current_size)
            return True

    def get(self) -> Optional[T]:
        """Remove and return highest priority item."""
        with self._lock:
            if not self._heap:
                return None
            _, _, item = heappop(self._heap)
            self._stats.items_removed += 1
            self._stats.current_size = len(self._heap)
            return item

    def peek(self) -> Optional[T]:
        """Return highest priority item without removing."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._heap.clear()
            self._stats.current_size = 0

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return len(self._heap) == 0

    @property
    def is_full(self) -> bool:
        return len(self._heap) >= self._max_size

    @property
    def stats(self) -> BufferStats:
        return self._stats


class BufferOverflowError(Exception):
    """Raised when buffer overflows with RAISE policy."""
    pass


__all__ = [
    "OverflowPolicy",
    "BufferStats",
    "BackpressureBuffer",
    "UnboundedBuffer",
    "BoundedBuffer",
    "SlidingWindowBuffer",
    "PriorityBuffer",
    "BufferOverflowError",
]
