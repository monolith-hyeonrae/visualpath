"""Filter node for conditional data passing.

FilterNode passes or drops FlowData based on a condition function.
"""

from typing import Callable, List

from visualpath.flow.node import FlowNode, FlowData, Condition


class FilterNode(FlowNode):
    """Node that filters FlowData based on a condition.

    FilterNode evaluates a condition function on each FlowData:
    - If True: data passes through (returns [data])
    - If False: data is dropped (returns [])

    Example:
        >>> def has_observations(data: FlowData) -> bool:
        ...     return len(data.observations) > 0
        ...
        >>> filter_node = FilterNode("has_obs", condition=has_observations)
    """

    def __init__(self, name: str, condition: Condition):
        """Initialize the filter node.

        Args:
            name: Unique name for this node.
            condition: Function that takes FlowData and returns bool.
        """
        self._name = name
        self._condition = condition

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return FilterSpec for this node."""
        from visualpath.flow.specs import FilterSpec
        return FilterSpec(condition=self._condition)

    def process(self, data: FlowData) -> List[FlowData]:
        """Filter data based on condition.

        Args:
            data: Input FlowData to evaluate.

        Returns:
            Single-item list if condition is True, empty list otherwise.
        """
        if self._condition(data):
            return [data]
        return []


class ObservationFilter(FlowNode):
    """Filter that passes data only if it has observations.

    A convenience filter for the common case of filtering out
    frames that didn't produce any observations.
    """

    def __init__(self, name: str = "observation_filter", min_count: int = 1):
        """Initialize the observation filter.

        Args:
            name: Unique name for this node.
            min_count: Minimum number of observations required.
        """
        self._name = name
        self._min_count = min_count

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return ObservationFilterSpec for this node."""
        from visualpath.flow.specs import ObservationFilterSpec
        return ObservationFilterSpec(min_count=self._min_count)

    def process(self, data: FlowData) -> List[FlowData]:
        """Filter based on observation count.

        Args:
            data: Input FlowData.

        Returns:
            Data if it has enough observations, empty list otherwise.
        """
        if len(data.observations) >= self._min_count:
            return [data]
        return []


class SignalThresholdFilter(FlowNode):
    """Filter based on signal value threshold.

    Passes data if any observation has a signal above threshold.
    """

    def __init__(
        self,
        name: str,
        signal_name: str,
        threshold: float,
        comparison: str = "gt",
    ):
        """Initialize the signal threshold filter.

        Args:
            name: Unique name for this node.
            signal_name: Name of the signal to check.
            threshold: Threshold value.
            comparison: Comparison type ("gt", "ge", "lt", "le", "eq").
        """
        self._name = name
        self._signal_name = signal_name
        self._threshold = threshold
        self._comparison = comparison

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return SignalFilterSpec for this node."""
        from visualpath.flow.specs import SignalFilterSpec
        return SignalFilterSpec(
            signal_name=self._signal_name,
            threshold=self._threshold,
            comparison=self._comparison,
        )

    def _compare(self, value: float) -> bool:
        """Compare value against threshold."""
        if self._comparison == "gt":
            return value > self._threshold
        elif self._comparison == "ge":
            return value >= self._threshold
        elif self._comparison == "lt":
            return value < self._threshold
        elif self._comparison == "le":
            return value <= self._threshold
        elif self._comparison == "eq":
            return value == self._threshold
        return False

    def process(self, data: FlowData) -> List[FlowData]:
        """Filter based on signal threshold.

        Args:
            data: Input FlowData.

        Returns:
            Data if any observation passes threshold, empty otherwise.
        """
        for obs in data.observations:
            if self._signal_name in obs.signals:
                if self._compare(obs.signals[self._signal_name]):
                    return [data]
        return []
