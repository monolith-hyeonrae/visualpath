"""Graph executor for running flow graphs.

GraphExecutor processes frames through a FlowGraph, handling
data routing, branching, and trigger firing.
"""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowData
from visualpath.flow.graph import FlowGraph

if TYPE_CHECKING:
    from visualbase import Frame


class GraphExecutor:
    """Executes a FlowGraph by processing frames through nodes.

    GraphExecutor handles:
    - Converting frames to FlowData via SourceNode
    - Routing data through nodes based on edges
    - Filtering by path_id for conditional routing
    - Firing triggers at terminal nodes

    Example:
        >>> executor = GraphExecutor(graph)
        >>> with executor:
        ...     for frame in video:
        ...         results = executor.process(frame)
    """

    def __init__(
        self,
        graph: FlowGraph,
        on_trigger: Optional[Callable[[FlowData], None]] = None,
    ):
        """Initialize the executor.

        Args:
            graph: FlowGraph to execute.
            on_trigger: Optional callback for triggers (in addition to
                any callbacks registered on the graph).
        """
        self._graph = graph
        self._initialized = False

        if on_trigger is not None:
            self._graph.on_trigger(on_trigger)

    @property
    def graph(self) -> FlowGraph:
        """Get the underlying flow graph."""
        return self._graph

    def initialize(self) -> None:
        """Initialize the executor and all graph nodes."""
        if self._initialized:
            return
        self._graph.validate()
        self._graph.initialize()
        self._initialized = True

    def cleanup(self) -> None:
        """Clean up the executor and all graph nodes."""
        if not self._initialized:
            return
        self._graph.cleanup()
        self._initialized = False

    def __enter__(self) -> "GraphExecutor":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def process(self, frame: "Frame") -> List[FlowData]:
        """Process a frame through the flow graph.

        Args:
            frame: Input frame to process.

        Returns:
            List of FlowData that reached terminal nodes.

        Raises:
            RuntimeError: If executor not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Executor not initialized. Use context manager or call initialize().")

        # Get entry node and create initial FlowData
        entry_name = self._graph.entry_node
        if entry_name is None:
            return []

        entry_node = self._graph.nodes[entry_name]

        # Use SourceNode.process_frame if available, otherwise wrap manually
        if hasattr(entry_node, "process_frame"):
            initial_data = entry_node.process_frame(frame)
        else:
            initial_data = FlowData(
                frame=frame,
                timestamp_ns=frame.t_src_ns if hasattr(frame, "t_src_ns") else 0,
            )

        return self.process_data(initial_data)

    def process_data(self, data: FlowData) -> List[FlowData]:
        """Process FlowData through the graph starting from entry.

        Args:
            data: Initial FlowData to process.

        Returns:
            List of FlowData that reached terminal nodes.
        """
        if not self._initialized:
            raise RuntimeError("Executor not initialized.")

        entry_name = self._graph.entry_node
        if entry_name is None:
            return []

        terminal_nodes = set(self._graph.get_terminal_nodes())
        terminal_results: List[FlowData] = []

        # BFS through the graph
        # Each item is (node_name, FlowData)
        queue: deque[tuple[str, FlowData]] = deque()
        queue.append((entry_name, data))

        while queue:
            node_name, current_data = queue.popleft()
            node = self._graph.nodes[node_name]

            # Process through this node
            outputs = node.process(current_data)

            # Route outputs to successors
            for output_data in outputs:
                # Get successors filtered by path_id
                successors = self._graph.get_successors(node_name, output_data.path_id)

                if not successors:
                    # This is a terminal point for this data
                    terminal_results.append(output_data)
                    # Fire triggers if this is a terminal node
                    if node_name in terminal_nodes:
                        self._graph.fire_triggers(output_data)
                else:
                    # Queue for next nodes
                    for successor in successors:
                        queue.append((successor, output_data))

        return terminal_results

    def process_batch(self, frames: List["Frame"]) -> List[List[FlowData]]:
        """Process multiple frames through the graph.

        Args:
            frames: List of frames to process.

        Returns:
            List of result lists, one per input frame.
        """
        return [self.process(frame) for frame in frames]
