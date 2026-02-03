"""FlowGraph to Pathway dataflow converter.

This module converts visualpath FlowGraph structures into Pathway
streaming dataflows.

Conversion mapping:
- SourceNode -> pw.io.python.read()
- PathNode (extractors) -> @pw.udf with PyObjectWrapper
- JoinNode -> interval_join()
- Fusion -> subscribe callback
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.flow.graph import FlowGraph
    from visualpath.flow.node import FlowNode

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


class FlowGraphConverter:
    """Converts FlowGraph to Pathway dataflow.

    FlowGraphConverter traverses the FlowGraph in topological order
    and creates corresponding Pathway operators for each node type.

    Supported node types:
    - SourceNode: Creates input stream from VideoConnectorSubject
    - PathNode: Creates @pw.udf for extractors with PyObjectWrapper
    - JoinNode: Creates interval_join for synchronization
    - FilterNode: Creates filter operation
    - SamplerNode: Creates sampling operation

    Example:
        >>> converter = FlowGraphConverter()
        >>> pw_dataflow = converter.convert(graph, frames_table)
    """

    def __init__(
        self,
        window_ns: int = 100_000_000,  # 100ms default
        allowed_lateness_ns: int = 50_000_000,  # 50ms default
    ) -> None:
        """Initialize the converter.

        Args:
            window_ns: Default window size for joins (nanoseconds).
            allowed_lateness_ns: Allowed late arrival time (nanoseconds).
        """
        self._window_ns = window_ns
        self._allowed_lateness_ns = allowed_lateness_ns
        self._tables: Dict[str, "pw.Table"] = {}

    def convert(
        self,
        graph: "FlowGraph",
        frames_table: "pw.Table",
    ) -> "pw.Table":
        """Convert a FlowGraph to Pathway dataflow.

        Args:
            graph: The FlowGraph to convert.
            frames_table: Initial Pathway table of frames.

        Returns:
            Final Pathway table representing the output.

        Raises:
            ValueError: If graph contains unsupported node types.
        """
        if not PATHWAY_AVAILABLE:
            raise ImportError(
                "Pathway is not installed. Install with: pip install visualpath[pathway]"
            )

        # Get topological order
        order = graph.topological_order()

        # Process nodes in order
        for node_name in order:
            node = graph.nodes[node_name]
            self._convert_node(node, graph, frames_table)

        # Return the last table
        if order:
            return self._tables.get(order[-1], frames_table)
        return frames_table

    def _convert_node(
        self,
        node: "FlowNode",
        graph: "FlowGraph",
        frames_table: "pw.Table",
    ) -> None:
        """Convert a single node to Pathway operators."""
        from visualpath.flow.nodes import (
            SourceNode,
            PathNode,
            JoinNode,
            FilterNode,
            SamplerNode,
            BranchNode,
            FanOutNode,
        )

        node_name = node.name

        if isinstance(node, SourceNode):
            self._tables[node_name] = frames_table

        elif isinstance(node, PathNode):
            self._convert_path_node(node, graph)

        elif isinstance(node, JoinNode):
            self._convert_join_node(node, graph)

        elif isinstance(node, FilterNode):
            self._convert_filter_node(node, graph)

        elif isinstance(node, SamplerNode):
            self._convert_sampler_node(node, graph)

        elif isinstance(node, (BranchNode, FanOutNode)):
            self._convert_branch_node(node, graph)

        else:
            # Default: pass through from predecessor
            predecessors = graph.get_incoming_edges(node_name)
            if predecessors:
                pred_name = predecessors[0].source
                self._tables[node_name] = self._tables.get(pred_name)

    def _convert_path_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert PathNode to Pathway UDF with PyObjectWrapper."""
        from visualpath.flow.nodes import PathNode
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        if not isinstance(node, PathNode):
            return

        predecessors = graph.get_incoming_edges(node.name)
        if not predecessors:
            return

        pred_name = predecessors[0].source
        input_table = self._tables.get(pred_name)
        if input_table is None:
            return

        path = node._path
        if path and path.extractors:
            raw_udf = create_multi_extractor_udf(path.extractors)

            @pw.udf
            def extract_udf(
                frame_wrapped: pw.PyObjectWrapper,
            ) -> pw.PyObjectWrapper:
                frame = frame_wrapped.value
                results = raw_udf(frame)
                return pw.PyObjectWrapper(results)

            result_table = input_table.select(
                frame_id=pw.this.frame_id,
                t_ns=pw.this.t_ns,
                frame=pw.this.frame,
                results=extract_udf(pw.this.frame),
            )
            self._tables[node.name] = result_table
        else:
            self._tables[node.name] = input_table

    def _convert_join_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert JoinNode to Pathway interval_join."""
        from visualpath.flow.nodes import JoinNode

        if not isinstance(node, JoinNode):
            return

        input_paths = list(node.input_paths)
        if len(input_paths) < 2:
            if input_paths:
                self._tables[node.name] = self._tables.get(input_paths[0])
            return

        tables = [self._tables.get(path) for path in input_paths]
        if None in tables:
            return

        # Use interval_join for time-based joining
        left = tables[0]
        right = tables[1]

        joined = left.interval_join(
            right,
            pw.left.t_ns,
            pw.right.t_ns,
            pw.temporal.interval(-self._window_ns, self._window_ns),
        ).select(
            frame_id=pw.left.frame_id,
            t_ns=pw.left.t_ns,
            frame=pw.left.frame,
        )

        # Join remaining tables
        for table in tables[2:]:
            joined = joined.interval_join(
                table,
                pw.left.t_ns,
                pw.right.t_ns,
                pw.temporal.interval(-self._window_ns, self._window_ns),
            ).select(
                frame_id=pw.left.frame_id,
                t_ns=pw.left.t_ns,
                frame=pw.left.frame,
            )

        self._tables[node.name] = joined

    def _convert_filter_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert FilterNode to Pathway filter."""
        from visualpath.flow.nodes import FilterNode

        if not isinstance(node, FilterNode):
            return

        predecessors = graph.get_incoming_edges(node.name)
        if not predecessors:
            return

        pred_name = predecessors[0].source
        input_table = self._tables.get(pred_name)
        if input_table is None:
            return

        condition = node._condition

        @pw.udf
        def filter_udf(frame_wrapped: pw.PyObjectWrapper) -> bool:
            return condition(frame_wrapped.value)

        filtered = input_table.filter(filter_udf(pw.this.frame))
        self._tables[node.name] = filtered

    def _convert_sampler_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert SamplerNode to Pathway sampling."""
        from visualpath.flow.nodes import SamplerNode

        if not isinstance(node, SamplerNode):
            return

        predecessors = graph.get_incoming_edges(node.name)
        if not predecessors:
            return

        pred_name = predecessors[0].source
        input_table = self._tables.get(pred_name)
        if input_table is None:
            return

        every_nth = node._every_nth
        sampled = input_table.filter(
            pw.this.frame_id % every_nth == 0
        )
        self._tables[node.name] = sampled

    def _convert_branch_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert branching nodes (pass-through)."""
        predecessors = graph.get_incoming_edges(node.name)
        if not predecessors:
            return

        pred_name = predecessors[0].source
        input_table = self._tables.get(pred_name)
        if input_table is None:
            return

        self._tables[node.name] = input_table


__all__ = ["FlowGraphConverter"]
