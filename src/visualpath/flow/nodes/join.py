"""Join nodes for merging multiple paths.

All join nodes are declarative — they expose a spec describing
the join strategy. The backend interprets the spec and manages
join buffers internally.
"""

from typing import Callable, List, Set

from visualpath.flow.node import FlowNode, FlowData
from visualpath.flow.specs import JoinSpec, CascadeFusionSpec, CollectorSpec


class JoinNode(FlowNode):
    """Merge data from multiple paths.

    Spec: JoinSpec(input_paths, mode, window_ns, ...)
    Backend: buffers data, emits merged FlowData when join condition met.
    """

    def __init__(
        self,
        name: str,
        input_paths: List[str],
        mode: str = "all",
        window_ns: int = 100_000_000,
        lateness_ns: int = 0,
        merge_observations: bool = True,
        merge_results: bool = True,
        output_path_id: str = "merged",
    ):
        if not input_paths:
            raise ValueError("input_paths must not be empty")
        self._name = name
        self._input_paths = set(input_paths)
        self._mode = mode
        self._window_ns = window_ns
        self._lateness_ns = lateness_ns
        self._merge_observations = merge_observations
        self._merge_results = merge_results
        self._output_path_id = output_path_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_paths(self) -> Set[str]:
        return set(self._input_paths)

    @property
    def spec(self) -> JoinSpec:
        return JoinSpec(
            input_paths=tuple(sorted(self._input_paths)),
            mode=self._mode,
            window_ns=self._window_ns,
            lateness_ns=self._lateness_ns,
            merge_observations=self._merge_observations,
            merge_results=self._merge_results,
            output_path_id=self._output_path_id,
        )


class CascadeFusionNode(FlowNode):
    """Apply secondary fusion on merged data.

    Spec: CascadeFusionSpec(fusion_fn)
    Backend: calls fusion_fn(data) and returns result.
    """

    def __init__(
        self,
        name: str,
        fusion_fn: Callable[[FlowData], FlowData],
    ):
        self._name = name
        self._fusion_fn = fusion_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> CascadeFusionSpec:
        return CascadeFusionSpec(fusion_fn=self._fusion_fn)


class CollectorNode(FlowNode):
    """Collect data into batches.

    Spec: CollectorSpec(batch_size, timeout_ns, emit_partial)
    Backend: buffers data, emits merged batch when condition met.
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 0,
        timeout_ns: int = 0,
        emit_partial: bool = True,
    ):
        self._name = name
        self._batch_size = batch_size
        self._timeout_ns = timeout_ns
        self._emit_partial = emit_partial

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> CollectorSpec:
        return CollectorSpec(
            batch_size=self._batch_size,
            timeout_ns=self._timeout_ns,
            emit_partial=self._emit_partial,
        )
