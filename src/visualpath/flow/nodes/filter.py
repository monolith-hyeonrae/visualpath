"""Filter nodes for conditional data passing.

All filter nodes are declarative — they expose a spec describing
the filter condition. The backend interprets the spec and filters data.
"""

from visualpath.flow.node import FlowNode, Condition
from visualpath.flow.specs import FilterSpec, ObservationFilterSpec, SignalFilterSpec


class FilterNode(FlowNode):
    """Generic filter node with condition function.

    Spec: FilterSpec(condition)
    Backend: passes data if condition(data) is True.
    """

    def __init__(self, name: str, condition: Condition):
        self._name = name
        self._condition = condition

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> FilterSpec:
        return FilterSpec(condition=self._condition)


class ObservationFilter(FlowNode):
    """Filter that requires minimum observation count.

    Spec: ObservationFilterSpec(min_count)
    Backend: passes data if len(observations) >= min_count.
    """

    def __init__(self, name: str = "observation_filter", min_count: int = 1):
        self._name = name
        self._min_count = min_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> ObservationFilterSpec:
        return ObservationFilterSpec(min_count=self._min_count)


class SignalThresholdFilter(FlowNode):
    """Filter based on signal value threshold.

    Spec: SignalFilterSpec(signal_name, threshold, comparison)
    Backend: passes data if any observation's signal passes threshold.
    """

    def __init__(
        self,
        name: str,
        signal_name: str,
        threshold: float,
        comparison: str = "gt",
    ):
        self._name = name
        self._signal_name = signal_name
        self._threshold = threshold
        self._comparison = comparison

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> SignalFilterSpec:
        return SignalFilterSpec(
            signal_name=self._signal_name,
            threshold=self._threshold,
            comparison=self._comparison,
        )
