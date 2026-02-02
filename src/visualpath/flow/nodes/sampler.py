"""Sampler nodes for frame rate control.

SamplerNode and RateLimiterNode control the rate of data flow
through the pipeline.
"""

import time
from typing import List

from visualpath.flow.node import FlowNode, FlowData


class SamplerNode(FlowNode):
    """Node that passes every Nth frame.

    SamplerNode is useful for reducing processing load when
    full frame rate is not needed.

    Example:
        >>> sampler = SamplerNode("sample_3", every_nth=3)
        >>> # Passes frames 0, 3, 6, 9, ...
    """

    def __init__(self, name: str = "sampler", every_nth: int = 1):
        """Initialize the sampler node.

        Args:
            name: Unique name for this node.
            every_nth: Pass every Nth frame (1 = all frames).
        """
        if every_nth < 1:
            raise ValueError("every_nth must be >= 1")

        self._name = name
        self._every_nth = every_nth
        self._count = 0

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def every_nth(self) -> int:
        """Get the sampling rate."""
        return self._every_nth

    def reset(self) -> None:
        """Reset the frame counter."""
        self._count = 0

    def process(self, data: FlowData) -> List[FlowData]:
        """Pass every Nth frame.

        Args:
            data: Input FlowData.

        Returns:
            Data if this is an Nth frame, empty list otherwise.
        """
        self._count += 1
        if self._count >= self._every_nth:
            self._count = 0
            return [data]
        return []


class RateLimiterNode(FlowNode):
    """Node that limits data rate by time interval.

    RateLimiterNode passes data at most once per specified time
    interval, useful for controlling FPS regardless of input rate.

    Example:
        >>> limiter = RateLimiterNode("limit_5fps", min_interval_ms=200)
        >>> # Passes at most 5 frames per second
    """

    def __init__(self, name: str = "rate_limiter", min_interval_ms: float = 0):
        """Initialize the rate limiter.

        Args:
            name: Unique name for this node.
            min_interval_ms: Minimum interval between passed frames in milliseconds.
        """
        if min_interval_ms < 0:
            raise ValueError("min_interval_ms must be >= 0")

        self._name = name
        self._min_interval_ms = min_interval_ms
        self._last_pass_time: float = 0

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def min_interval_ms(self) -> float:
        """Get the minimum interval in milliseconds."""
        return self._min_interval_ms

    def reset(self) -> None:
        """Reset the last pass time."""
        self._last_pass_time = 0

    def process(self, data: FlowData) -> List[FlowData]:
        """Pass data if enough time has elapsed.

        Args:
            data: Input FlowData.

        Returns:
            Data if interval has elapsed, empty list otherwise.
        """
        current_time = time.monotonic() * 1000  # Convert to ms

        if current_time - self._last_pass_time >= self._min_interval_ms:
            self._last_pass_time = current_time
            return [data]
        return []


class TimestampSamplerNode(FlowNode):
    """Node that samples based on FlowData timestamp.

    Unlike RateLimiterNode which uses wall clock time, this node
    uses the timestamp_ns in FlowData, useful for processing
    recorded video at different rates.

    Example:
        >>> sampler = TimestampSamplerNode("ts_sample", interval_ns=33_333_333)
        >>> # Passes at ~30fps based on video timestamps
    """

    def __init__(self, name: str = "timestamp_sampler", interval_ns: int = 0):
        """Initialize the timestamp sampler.

        Args:
            name: Unique name for this node.
            interval_ns: Minimum interval between samples in nanoseconds.
        """
        if interval_ns < 0:
            raise ValueError("interval_ns must be >= 0")

        self._name = name
        self._interval_ns = interval_ns
        self._last_timestamp_ns: int = 0
        self._first_frame = True

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def interval_ns(self) -> int:
        """Get the sampling interval in nanoseconds."""
        return self._interval_ns

    def reset(self) -> None:
        """Reset the sampler state."""
        self._last_timestamp_ns = 0
        self._first_frame = True

    def process(self, data: FlowData) -> List[FlowData]:
        """Sample based on FlowData timestamp.

        Args:
            data: Input FlowData.

        Returns:
            Data if interval has elapsed, empty list otherwise.
        """
        # Always pass first frame
        if self._first_frame:
            self._first_frame = False
            self._last_timestamp_ns = data.timestamp_ns
            return [data]

        elapsed = data.timestamp_ns - self._last_timestamp_ns

        if elapsed >= self._interval_ns:
            self._last_timestamp_ns = data.timestamp_ns
            return [data]
        return []
