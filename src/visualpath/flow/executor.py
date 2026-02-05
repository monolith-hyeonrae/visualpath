"""Graph executor for running flow graphs via the interpreter.

GraphExecutor drives frame processing through a FlowGraph, using
SimpleInterpreter to interpret each node's spec. The executor handles:
- Converting frames to FlowData via SourceSpec
- Routing data through nodes based on edges and path_id
- Firing triggers at terminal nodes

This is a convenience wrapper around FlowGraph + SimpleInterpreter.
"""

from collections import deque
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowData
from visualpath.flow.graph import FlowGraph
from visualpath.flow.interpreter import SimpleInterpreter
from visualpath.flow.specs import SourceSpec

if TYPE_CHECKING:
    from visualbase import Frame


class GraphExecutor:
    """Executes a FlowGraph using SimpleInterpreter.

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
        self._graph = graph
        self._interpreter = SimpleInterpreter()
        self._initialized = False

        if on_trigger is not None:
            self._graph.on_trigger(on_trigger)

    @property
    def graph(self) -> FlowGraph:
        return self._graph

    @property
    def interpreter(self) -> SimpleInterpreter:
        return self._interpreter

    def initialize(self) -> None:
        if self._initialized:
            return
        self._graph.validate()
        self._graph.initialize()
        self._interpreter.reset()
        self._initialized = True

    def cleanup(self) -> None:
        if not self._initialized:
            return
        self._graph.cleanup()
        self._interpreter.reset()
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
        """
        if not self._initialized:
            raise RuntimeError(
                "Executor not initialized. Use context manager or call initialize()."
            )

        entry_name = self._graph.entry_node
        if entry_name is None:
            return []

        entry_node = self._graph.nodes[entry_name]
        spec = entry_node.spec

        # Create FlowData from frame using SourceSpec's default_path_id
        if isinstance(spec, SourceSpec):
            initial_data = FlowData(
                frame=frame,
                path_id=spec.default_path_id,
                timestamp_ns=getattr(frame, "t_src_ns", 0),
            )
        else:
            initial_data = FlowData(
                frame=frame,
                timestamp_ns=getattr(frame, "t_src_ns", 0),
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
        queue: deque[tuple[str, FlowData]] = deque()
        queue.append((entry_name, data))

        while queue:
            node_name, current_data = queue.popleft()
            node = self._graph.nodes[node_name]

            # Interpret the node's spec
            outputs = self._interpreter.interpret(node, current_data)

            # Route outputs to successors
            for output_data in outputs:
                successors = self._graph.get_successors(
                    node_name, output_data.path_id
                )

                if not successors:
                    terminal_results.append(output_data)
                    if node_name in terminal_nodes:
                        self._graph.fire_triggers(output_data)
                else:
                    for successor in successors:
                        queue.append((successor, output_data))

        return terminal_results

    def process_batch(self, frames: List["Frame"]) -> List[List[FlowData]]:
        """Process multiple frames through the graph."""
        return [self.process(frame) for frame in frames]
