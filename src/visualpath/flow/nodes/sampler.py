"""Sampler nodes for frame rate control.

All sampler nodes are declarative — they expose a spec describing
the sampling strategy. The backend interprets the spec and manages
sampling state internally.
"""

from visualpath.flow.node import FlowNode
from visualpath.flow.specs import SampleSpec, RateLimitSpec, TimestampSampleSpec


class SamplerNode(FlowNode):
    """Every-Nth frame sampler.

    Spec: SampleSpec(every_nth=N)
    Backend: maintains counter, passes every Nth frame.
    """

    def __init__(self, name: str = "sampler", every_nth: int = 1):
        if every_nth < 1:
            raise ValueError(f"every_nth must be >= 1, got {every_nth}")
        self._name = name
        self._every_nth = every_nth

    @property
    def name(self) -> str:
        return self._name

    @property
    def every_nth(self) -> int:
        return self._every_nth

    @property
    def spec(self) -> SampleSpec:
        return SampleSpec(every_nth=self._every_nth)


class RateLimiterNode(FlowNode):
    """Time-based rate limiter.

    Spec: RateLimitSpec(min_interval_ms=N)
    Backend: maintains wall-clock timer, passes if enough time elapsed.
    """

    def __init__(self, name: str = "rate_limiter", min_interval_ms: float = 0):
        if min_interval_ms < 0:
            raise ValueError("min_interval_ms must be >= 0")
        self._name = name
        self._min_interval_ms = min_interval_ms

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_interval_ms(self) -> float:
        return self._min_interval_ms

    @property
    def spec(self) -> RateLimitSpec:
        return RateLimitSpec(min_interval_ms=self._min_interval_ms)


class TimestampSamplerNode(FlowNode):
    """Timestamp-based sampler using data timestamps.

    Spec: TimestampSampleSpec(interval_ns=N)
    Backend: uses FlowData.timestamp_ns to decide if enough time passed.
    """

    def __init__(self, name: str = "timestamp_sampler", interval_ns: int = 0):
        if interval_ns < 0:
            raise ValueError("interval_ns must be >= 0")
        self._name = name
        self._interval_ns = interval_ns

    @property
    def name(self) -> str:
        return self._name

    @property
    def interval_ns(self) -> int:
        return self._interval_ns

    @property
    def spec(self) -> TimestampSampleSpec:
        return TimestampSampleSpec(interval_ns=self._interval_ns)
