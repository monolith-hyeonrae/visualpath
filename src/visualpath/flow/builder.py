"""Fluent builder for constructing flow graphs.

FlowGraphBuilder provides a declarative API for building FlowGraphs
with method chaining.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowNode, FlowData, Condition
from visualpath.flow.graph import FlowGraph
from visualpath.flow.nodes.source import SourceNode
from visualpath.flow.nodes.path import PathNode
from visualpath.flow.nodes.filter import FilterNode, ObservationFilter, SignalThresholdFilter
from visualpath.flow.nodes.sampler import SamplerNode, RateLimiterNode, TimestampSamplerNode
from visualpath.flow.nodes.branch import BranchNode, FanOutNode, MultiBranchNode
from visualpath.flow.nodes.join import JoinNode, CascadeFusionNode, CollectorNode

if TYPE_CHECKING:
    from visualpath.core.extractor import BaseExtractor
    from visualpath.core.fusion import BaseFusion
    from visualpath.core.path import Path


class FlowGraphBuilder:
    """Fluent builder for creating flow graphs.

    FlowGraphBuilder provides a chainable API for constructing
    FlowGraphs declaratively.

    Example:
        >>> graph = (FlowGraphBuilder()
        ...     .source("frames")
        ...     .sample(every_nth=3)
        ...     .branch(
        ...         condition=lambda d: has_face(d),
        ...         if_true="human",
        ...         if_false="scene",
        ...     )
        ...     .path("human", extractors=["face", "pose"], fusion="highlight")
        ...     .path("scene", extractors=["object"], fusion="scene")
        ...     .join(["human", "scene"])
        ...     .on_trigger(handle_trigger)
        ...     .build())
    """

    def __init__(self):
        """Initialize the builder."""
        self._graph = FlowGraph()
        self._last_node: Optional[str] = None
        self._path_last_nodes: Dict[str, str] = {}  # path_id -> last node name

        # For path resolution
        self._extractors: Dict[str, "BaseExtractor"] = {}
        self._fusions: Dict[str, "BaseFusion"] = {}

    def register_extractor(self, name: str, extractor: "BaseExtractor") -> "FlowGraphBuilder":
        """Register an extractor for later reference by name.

        Args:
            name: Name to reference the extractor.
            extractor: Extractor instance.

        Returns:
            Self for chaining.
        """
        self._extractors[name] = extractor
        return self

    def register_fusion(self, name: str, fusion: "BaseFusion") -> "FlowGraphBuilder":
        """Register a fusion module for later reference by name.

        Args:
            name: Name to reference the fusion.
            fusion: Fusion instance.

        Returns:
            Self for chaining.
        """
        self._fusions[name] = fusion
        return self

    def _add_node_with_edge(
        self,
        node: FlowNode,
        from_node: Optional[str] = None,
        path_filter: Optional[str] = None,
    ) -> str:
        """Add a node and connect it to the previous node.

        Args:
            node: Node to add.
            from_node: Source node name (defaults to last added).
            path_filter: Optional path filter for the edge.

        Returns:
            Name of the added node.
        """
        self._graph.add_node(node)

        source = from_node or self._last_node
        if source is not None:
            self._graph.add_edge(source, node.name, path_filter=path_filter)

        self._last_node = node.name
        return node.name

    def source(self, name: str = "source", default_path_id: str = "default") -> "FlowGraphBuilder":
        """Add a source node as entry point.

        Args:
            name: Name for the source node.
            default_path_id: Default path_id for created FlowData.

        Returns:
            Self for chaining.
        """
        node = SourceNode(name=name, default_path_id=default_path_id)
        self._graph.add_node(node)
        self._graph.entry_node = name
        self._last_node = name
        return self

    def add_node(self, node: FlowNode, from_node: Optional[str] = None) -> "FlowGraphBuilder":
        """Add a custom node to the graph.

        Args:
            node: Custom FlowNode to add.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        self._add_node_with_edge(node, from_node)
        return self

    def sample(
        self,
        every_nth: int = 1,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a sampler node.

        Args:
            every_nth: Pass every Nth frame.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"sampler_{every_nth}"
        node = SamplerNode(name=node_name, every_nth=every_nth)
        self._add_node_with_edge(node, from_node)
        return self

    def rate_limit(
        self,
        min_interval_ms: float,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a rate limiter node.

        Args:
            min_interval_ms: Minimum interval between frames in ms.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"rate_limit_{int(min_interval_ms)}ms"
        node = RateLimiterNode(name=node_name, min_interval_ms=min_interval_ms)
        self._add_node_with_edge(node, from_node)
        return self

    def filter(
        self,
        condition: Condition,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a filter node.

        Args:
            condition: Function that returns True to pass data.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"filter_{len(self._graph.nodes)}"
        node = FilterNode(name=node_name, condition=condition)
        self._add_node_with_edge(node, from_node)
        return self

    def filter_observations(
        self,
        min_count: int = 1,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add an observation filter.

        Args:
            min_count: Minimum required observations.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or "observation_filter"
        node = ObservationFilter(name=node_name, min_count=min_count)
        self._add_node_with_edge(node, from_node)
        return self

    def filter_signal(
        self,
        signal_name: str,
        threshold: float,
        comparison: str = "gt",
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a signal threshold filter.

        Args:
            signal_name: Name of signal to check.
            threshold: Threshold value.
            comparison: Comparison type ("gt", "ge", "lt", "le", "eq").
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"signal_filter_{signal_name}"
        node = SignalThresholdFilter(
            name=node_name,
            signal_name=signal_name,
            threshold=threshold,
            comparison=comparison,
        )
        self._add_node_with_edge(node, from_node)
        return self

    def branch(
        self,
        condition: Condition,
        if_true: str,
        if_false: str,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a branch node.

        Args:
            condition: Function that returns True/False for routing.
            if_true: path_id when condition is True.
            if_false: path_id when condition is False.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"branch_{if_true}_{if_false}"
        node = BranchNode(
            name=node_name,
            condition=condition,
            if_true=if_true,
            if_false=if_false,
        )
        self._add_node_with_edge(node, from_node)

        # Track this as the last node for both paths
        self._path_last_nodes[if_true] = node_name
        self._path_last_nodes[if_false] = node_name

        return self

    def fanout(
        self,
        paths: List[str],
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a fanout node.

        Args:
            paths: List of path_ids to create.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"fanout_{'_'.join(paths)}"
        node = FanOutNode(name=node_name, paths=paths)
        self._add_node_with_edge(node, from_node)

        # Track this as the last node for all paths
        for path_id in paths:
            self._path_last_nodes[path_id] = node_name

        return self

    def path(
        self,
        name: str,
        extractors: Optional[List["BaseExtractor"]] = None,
        fusion: Optional["BaseFusion"] = None,
        path: Optional["Path"] = None,
        run_fusion: bool = True,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a path node.

        Args:
            name: Name for the path (also used as path_id).
            extractors: List of extractors or extractor names.
            fusion: Fusion module or fusion name.
            path: Existing Path instance (alternative to extractors/fusion).
            run_fusion: Whether to run fusion in this node.
            from_node: Source node name (defaults to last node for this path).

        Returns:
            Self for chaining.
        """
        # Determine source node
        source = from_node
        if source is None:
            source = self._path_last_nodes.get(name, self._last_node)

        if path is not None:
            node = PathNode(path=path, run_fusion=run_fusion)
        else:
            # Resolve extractors and fusion by name if needed
            resolved_extractors = []
            if extractors:
                for ext in extractors:
                    if isinstance(ext, str):
                        if ext in self._extractors:
                            resolved_extractors.append(self._extractors[ext])
                        else:
                            raise ValueError(f"Unknown extractor: {ext}")
                    else:
                        resolved_extractors.append(ext)

            resolved_fusion = None
            if fusion is not None:
                if isinstance(fusion, str):
                    if fusion in self._fusions:
                        resolved_fusion = self._fusions[fusion]
                    else:
                        raise ValueError(f"Unknown fusion: {fusion}")
                else:
                    resolved_fusion = fusion

            node = PathNode(
                name=name,
                extractors=resolved_extractors,
                fusion=resolved_fusion,
                run_fusion=run_fusion,
            )

        self._graph.add_node(node)

        if source is not None:
            # Add edge with path filter if this path was created by branching
            path_filter = name if name in self._path_last_nodes else None
            self._graph.add_edge(source, node.name, path_filter=path_filter)

        self._last_node = node.name
        self._path_last_nodes[name] = node.name

        return self

    def join(
        self,
        input_paths: List[str],
        name: Optional[str] = None,
        mode: str = "all",
        window_ns: int = 100_000_000,
        output_path_id: str = "merged",
    ) -> "FlowGraphBuilder":
        """Add a join node.

        Args:
            input_paths: List of path_ids to join.
            name: Optional name for the node.
            mode: Join mode ("all" or "any").
            window_ns: Time window for grouping data.
            output_path_id: path_id for merged output.

        Returns:
            Self for chaining.
        """
        node_name = name or f"join_{'_'.join(input_paths)}"
        node = JoinNode(
            name=node_name,
            input_paths=input_paths,
            mode=mode,
            window_ns=window_ns,
            output_path_id=output_path_id,
        )

        self._graph.add_node(node)

        # Connect all input paths to this join
        for path_id in input_paths:
            if path_id in self._path_last_nodes:
                source = self._path_last_nodes[path_id]
                self._graph.add_edge(source, node_name, path_filter=path_id)

        self._last_node = node_name
        self._path_last_nodes[output_path_id] = node_name

        return self

    def cascade_fusion(
        self,
        fusion_fn: Callable[[FlowData], FlowData],
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a cascade fusion node.

        Args:
            fusion_fn: Function to apply to merged data.
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"cascade_{len(self._graph.nodes)}"
        node = CascadeFusionNode(name=node_name, fusion_fn=fusion_fn)
        self._add_node_with_edge(node, from_node)
        return self

    def collect(
        self,
        batch_size: int = 0,
        timeout_ns: int = 0,
        name: Optional[str] = None,
        from_node: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a collector node.

        Args:
            batch_size: Number of items to collect (0 = no limit).
            timeout_ns: Collection timeout (0 = no timeout).
            name: Optional name for the node.
            from_node: Source node name (defaults to last added).

        Returns:
            Self for chaining.
        """
        node_name = name or f"collector_{len(self._graph.nodes)}"
        node = CollectorNode(
            name=node_name,
            batch_size=batch_size,
            timeout_ns=timeout_ns,
        )
        self._add_node_with_edge(node, from_node)
        return self

    def on_trigger(self, callback: Callable[[FlowData], None]) -> "FlowGraphBuilder":
        """Register a trigger callback.

        Args:
            callback: Function to call when a trigger fires.

        Returns:
            Self for chaining.
        """
        self._graph.on_trigger(callback)
        return self

    def edge(
        self,
        source: str,
        target: str,
        path_filter: Optional[str] = None,
    ) -> "FlowGraphBuilder":
        """Add a custom edge between nodes.

        Args:
            source: Source node name.
            target: Target node name.
            path_filter: Optional path filter for conditional routing.

        Returns:
            Self for chaining.
        """
        self._graph.add_edge(source, target, path_filter=path_filter)
        return self

    def build(self) -> FlowGraph:
        """Build and validate the flow graph.

        Returns:
            Constructed FlowGraph.

        Raises:
            ValueError: If graph validation fails.
        """
        self._graph.validate()
        return self._graph
