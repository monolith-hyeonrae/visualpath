"""Branch nodes for conditional routing.

All branch nodes are declarative — they expose a spec describing
the routing strategy. The backend interprets the spec and routes data.
"""

from typing import List, Optional

from visualpath.flow.node import FlowNode, Condition
from visualpath.flow.specs import BranchSpec, FanOutSpec, MultiBranchSpec, ConditionalFanOutSpec


class BranchNode(FlowNode):
    """Binary branch based on condition.

    Spec: BranchSpec(condition, if_true, if_false)
    Backend: routes to if_true or if_false path based on condition.
    """

    def __init__(
        self,
        name: str,
        condition: Condition,
        if_true: str,
        if_false: str,
    ):
        self._name = name
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> BranchSpec:
        return BranchSpec(
            condition=self._condition,
            if_true=self._if_true,
            if_false=self._if_false,
        )


class FanOutNode(FlowNode):
    """Replicate data to multiple paths.

    Spec: FanOutSpec(paths)
    Backend: clones data for each path.
    """

    def __init__(self, name: str, paths: List[str]):
        if not paths:
            raise ValueError("paths must not be empty")
        self._name = name
        self._paths = paths

    @property
    def name(self) -> str:
        return self._name

    @property
    def paths(self) -> List[str]:
        return list(self._paths)

    @property
    def spec(self) -> FanOutSpec:
        return FanOutSpec(paths=tuple(self._paths))


class MultiBranchNode(FlowNode):
    """Multi-way branch with ordered conditions.

    Spec: MultiBranchSpec(branches, default)
    Backend: routes to first matching condition, or default.
    """

    def __init__(
        self,
        name: str,
        branches: List[tuple[Condition, str]],
        default: Optional[str] = None,
    ):
        self._name = name
        self._branches = branches
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> MultiBranchSpec:
        return MultiBranchSpec(
            branches=tuple(self._branches),
            default=self._default,
        )


class ConditionalFanOutNode(FlowNode):
    """Selective fan-out based on per-path conditions.

    Spec: ConditionalFanOutSpec(paths)
    Backend: clones data only to paths where condition passes.
    """

    def __init__(
        self,
        name: str,
        paths: List[tuple[str, Condition]],
    ):
        if not paths:
            raise ValueError("paths must not be empty")
        self._name = name
        self._paths = paths

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> ConditionalFanOutSpec:
        return ConditionalFanOutSpec(paths=tuple(self._paths))
