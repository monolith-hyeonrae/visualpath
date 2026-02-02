"""Branch nodes for conditional routing.

BranchNode routes data to different paths based on conditions.
FanOutNode replicates data to multiple paths.
"""

from typing import Callable, List, Optional

from visualpath.flow.node import FlowNode, FlowData, Condition


class BranchNode(FlowNode):
    """Node that routes data to different paths based on condition.

    BranchNode evaluates a condition and sets the path_id accordingly,
    allowing edges with path_filter to route data conditionally.

    Example:
        >>> def has_face(data: FlowData) -> bool:
        ...     return any(obs.signals.get("face_count", 0) > 0
        ...                for obs in data.observations)
        ...
        >>> branch = BranchNode(
        ...     "face_branch",
        ...     condition=has_face,
        ...     if_true="human",
        ...     if_false="scene",
        ... )
    """

    def __init__(
        self,
        name: str,
        condition: Condition,
        if_true: str,
        if_false: str,
    ):
        """Initialize the branch node.

        Args:
            name: Unique name for this node.
            condition: Function that takes FlowData and returns bool.
            if_true: path_id to set when condition is True.
            if_false: path_id to set when condition is False.
        """
        self._name = name
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    def process(self, data: FlowData) -> List[FlowData]:
        """Route data based on condition.

        Args:
            data: Input FlowData.

        Returns:
            Single-item list with path_id set based on condition result.
        """
        if self._condition(data):
            return [data.with_path(self._if_true)]
        else:
            return [data.with_path(self._if_false)]


class FanOutNode(FlowNode):
    """Node that replicates data to multiple paths.

    FanOutNode creates copies of FlowData with different path_ids,
    allowing parallel processing through multiple paths.

    Example:
        >>> fanout = FanOutNode("split", paths=["human", "scene", "action"])
        >>> # Input data is copied 3 times with different path_ids
    """

    def __init__(self, name: str, paths: List[str]):
        """Initialize the fanout node.

        Args:
            name: Unique name for this node.
            paths: List of path_ids to assign to copies.
        """
        if not paths:
            raise ValueError("paths must not be empty")

        self._name = name
        self._paths = paths

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def paths(self) -> List[str]:
        """Get the output paths."""
        return list(self._paths)

    def process(self, data: FlowData) -> List[FlowData]:
        """Replicate data to all paths.

        Args:
            data: Input FlowData.

        Returns:
            List of FlowData copies, one per output path.
        """
        return [data.with_path(path_id) for path_id in self._paths]


class MultiBranchNode(FlowNode):
    """Node that routes to one of multiple paths based on conditions.

    MultiBranchNode evaluates multiple conditions in order and routes
    to the first matching path. Includes optional default path.

    Example:
        >>> branches = [
        ...     (lambda d: d.metadata.get("priority") == "high", "urgent"),
        ...     (lambda d: d.metadata.get("priority") == "medium", "normal"),
        ... ]
        >>> router = MultiBranchNode("priority_router", branches, default="low")
    """

    def __init__(
        self,
        name: str,
        branches: List[tuple[Condition, str]],
        default: Optional[str] = None,
    ):
        """Initialize the multi-branch node.

        Args:
            name: Unique name for this node.
            branches: List of (condition, path_id) tuples evaluated in order.
            default: Default path_id if no condition matches (None drops data).
        """
        self._name = name
        self._branches = branches
        self._default = default

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    def process(self, data: FlowData) -> List[FlowData]:
        """Route data to first matching branch.

        Args:
            data: Input FlowData.

        Returns:
            Single-item list with matching path_id, or empty if no match
            and no default.
        """
        for condition, path_id in self._branches:
            if condition(data):
                return [data.with_path(path_id)]

        if self._default is not None:
            return [data.with_path(self._default)]

        return []


class ConditionalFanOutNode(FlowNode):
    """Node that selectively replicates to paths based on conditions.

    Unlike FanOutNode which always outputs to all paths, this node
    evaluates conditions for each path and only outputs to matching ones.

    Example:
        >>> paths = [
        ...     ("human", lambda d: has_face(d)),
        ...     ("scene", lambda d: True),  # Always process scene
        ...     ("action", lambda d: has_motion(d)),
        ... ]
        >>> fanout = ConditionalFanOutNode("selective_split", paths)
    """

    def __init__(
        self,
        name: str,
        paths: List[tuple[str, Condition]],
    ):
        """Initialize the conditional fanout node.

        Args:
            name: Unique name for this node.
            paths: List of (path_id, condition) tuples.
        """
        if not paths:
            raise ValueError("paths must not be empty")

        self._name = name
        self._paths = paths

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    def process(self, data: FlowData) -> List[FlowData]:
        """Replicate data to matching paths.

        Args:
            data: Input FlowData.

        Returns:
            List of FlowData copies for paths where condition is True.
        """
        results = []
        for path_id, condition in self._paths:
            if condition(data):
                results.append(data.with_path(path_id))
        return results
