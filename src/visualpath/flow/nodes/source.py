"""Source node for flow graph entry point.

SourceNode declares the entry point for the flow graph.
Execution is handled by the interpreter.
"""

from typing import TYPE_CHECKING

from visualpath.flow.node import FlowNode, FlowData
from visualpath.flow.specs import SourceSpec

if TYPE_CHECKING:
    from visualbase import Frame


class SourceNode(FlowNode):
    """Entry point node that declares source semantics.

    Spec: SourceSpec(default_path_id)
    Backend: creates FlowData from frames using the default_path_id.
    """

    def __init__(self, name: str = "source", default_path_id: str = "default"):
        self._name = name
        self._default_path_id = default_path_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> SourceSpec:
        return SourceSpec(default_path_id=self._default_path_id)

    def create_flow_data(self, frame: "Frame") -> FlowData:
        """Create FlowData from a Frame.

        Utility method for GraphExecutor to create initial FlowData.
        """
        return FlowData(
            frame=frame,
            observations=[],
            results=[],
            metadata={},
            path_id=self._default_path_id,
            timestamp_ns=getattr(frame, "t_src_ns", 0),
        )
