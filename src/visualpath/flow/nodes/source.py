"""Source node for flow graph entry point.

SourceNode converts Frame objects into FlowData, serving as the
entry point for the flow graph.
"""

from typing import List, TYPE_CHECKING

from visualpath.flow.node import FlowNode, FlowData

if TYPE_CHECKING:
    from visualbase import Frame


class SourceNode(FlowNode):
    """Entry point node that converts Frames to FlowData.

    SourceNode wraps incoming frames in FlowData containers,
    initializing the timestamp and path_id for downstream processing.

    Example:
        >>> source = SourceNode("video_input")
        >>> flow_data = source.process_frame(frame)
    """

    def __init__(self, name: str = "source", default_path_id: str = "default"):
        """Initialize the source node.

        Args:
            name: Unique name for this source node.
            default_path_id: Default path_id for created FlowData.
        """
        self._name = name
        self._default_path_id = default_path_id

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return SourceSpec for this node."""
        from visualpath.flow.specs import SourceSpec
        return SourceSpec(default_path_id=self._default_path_id)

    def process(self, data: FlowData) -> List[FlowData]:
        """Pass through existing FlowData.

        When used as a regular node in the graph, just passes data through.

        Args:
            data: Input FlowData.

        Returns:
            Single-item list containing the input data.
        """
        return [data]

    def process_frame(self, frame: "Frame") -> FlowData:
        """Convert a Frame to FlowData.

        This is the primary method for creating new FlowData from frames.

        Args:
            frame: Input frame to wrap.

        Returns:
            FlowData containing the frame.
        """
        return FlowData(
            frame=frame,
            observations=[],
            results=[],
            metadata={},
            path_id=self._default_path_id,
            timestamp_ns=frame.t_src_ns if hasattr(frame, "t_src_ns") else 0,
        )
