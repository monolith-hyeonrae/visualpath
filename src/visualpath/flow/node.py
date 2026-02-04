"""Base flow node abstractions.

FlowNode is the abstract base class for all nodes in the flow graph.
FlowData is the container that flows between nodes.

Example:
    >>> from visualpath.flow.node import FlowNode, FlowData
    >>>
    >>> class MyNode(FlowNode):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_node"
    ...
    ...     def process(self, data: FlowData) -> List[FlowData]:
    ...         # Transform data
    ...         return [data]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.flow.specs import NodeSpec as _NodeSpec

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.extractor import Observation
    from visualpath.core.fusion import FusionResult


@dataclass
class FlowData:
    """Data container that flows between nodes.

    FlowData carries all information needed for processing through the
    flow graph, including the original frame, extracted observations,
    fusion results, and routing metadata.

    Attributes:
        frame: Optional source frame being processed.
        observations: List of observations from extractors.
        results: List of fusion results.
        metadata: Arbitrary key-value metadata.
        path_id: Routing identifier for branching/joining.
        timestamp_ns: Timestamp in nanoseconds for synchronization.
    """

    frame: Optional["Frame"] = None
    observations: List["Observation"] = field(default_factory=list)
    results: List["FusionResult"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    path_id: str = "default"
    timestamp_ns: int = 0

    def clone(self, **overrides: Any) -> "FlowData":
        """Create a shallow copy with optional field overrides.

        Args:
            **overrides: Fields to override in the copy.

        Returns:
            New FlowData instance with copied/overridden values.
        """
        return FlowData(
            frame=overrides.get("frame", self.frame),
            observations=overrides.get("observations", list(self.observations)),
            results=overrides.get("results", list(self.results)),
            metadata=overrides.get("metadata", dict(self.metadata)),
            path_id=overrides.get("path_id", self.path_id),
            timestamp_ns=overrides.get("timestamp_ns", self.timestamp_ns),
        )

    def with_path(self, path_id: str) -> "FlowData":
        """Create a copy with a new path_id.

        Args:
            path_id: New path identifier.

        Returns:
            New FlowData with updated path_id.
        """
        return self.clone(path_id=path_id)

    def with_observations(self, observations: List["Observation"]) -> "FlowData":
        """Create a copy with new observations.

        Args:
            observations: New observations list.

        Returns:
            New FlowData with updated observations.
        """
        return self.clone(observations=observations)

    def with_results(self, results: List["FusionResult"]) -> "FlowData":
        """Create a copy with new results.

        Args:
            results: New results list.

        Returns:
            New FlowData with updated results.
        """
        return self.clone(results=results)

    def add_observation(self, observation: "Observation") -> "FlowData":
        """Create a copy with an additional observation.

        Args:
            observation: Observation to add.

        Returns:
            New FlowData with the observation appended.
        """
        new_observations = list(self.observations)
        new_observations.append(observation)
        return self.clone(observations=new_observations)

    def add_result(self, result: "FusionResult") -> "FlowData":
        """Create a copy with an additional result.

        Args:
            result: FusionResult to add.

        Returns:
            New FlowData with the result appended.
        """
        new_results = list(self.results)
        new_results.append(result)
        return self.clone(results=new_results)


class FlowNode(ABC):
    """Abstract base class for flow graph nodes.

    FlowNodes process FlowData and produce zero or more output FlowData:
    - 0 outputs: data is filtered/dropped
    - 1 output: data passes through (possibly transformed)
    - N outputs: data is branched/fanned out

    Subclasses must implement:
    - name: Unique identifier for the node
    - process: Transform input FlowData to output(s)

    Example:
        >>> class PassthroughNode(FlowNode):
        ...     @property
        ...     def name(self) -> str:
        ...         return "passthrough"
        ...
        ...     def process(self, data: FlowData) -> List[FlowData]:
        ...         return [data]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this node."""
        ...

    @abstractmethod
    def process(self, data: FlowData) -> List[FlowData]:
        """Process input data and produce output(s).

        Args:
            data: Input FlowData to process.

        Returns:
            List of output FlowData. Empty list means data is filtered.
            Single item means pass-through. Multiple items means branching.
        """
        ...

    @property
    def spec(self) -> Optional["_NodeSpec"]:
        """Declarative spec describing this node's semantics.

        Backends use the spec for optimized dispatch. If ``None``,
        the backend falls back to calling ``process()`` directly.

        Override in subclasses to return a frozen dataclass from
        :mod:`visualpath.flow.specs`.
        """
        return None

    def initialize(self) -> None:
        """Initialize node resources.

        Override to allocate resources before processing starts.
        """
        pass

    def cleanup(self) -> None:
        """Clean up node resources.

        Override to release resources after processing ends.
        """
        pass

    def __enter__(self) -> "FlowNode":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()


# Type alias for condition functions
Condition = Callable[[FlowData], bool]
