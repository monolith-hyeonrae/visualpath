"""Flow graph for DAG-based processing.

FlowGraph manages a directed acyclic graph of FlowNodes,
handling node registration, edge connections, and validation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from visualpath.flow.node import FlowNode, FlowData


@dataclass
class Edge:
    """An edge connecting two nodes in the flow graph.

    Attributes:
        source: Name of the source node.
        target: Name of the target node.
        path_filter: Optional path_id filter - edge only activates
            for FlowData with matching path_id (None means any).
    """

    source: str
    target: str
    path_filter: Optional[str] = None


class FlowGraph:
    """Directed acyclic graph of flow nodes.

    FlowGraph manages the structure of a processing pipeline:
    - Nodes process FlowData
    - Edges define data flow between nodes
    - Edges can filter by path_id for conditional routing

    Example:
        >>> graph = FlowGraph()
        >>> graph.add_node(source_node)
        >>> graph.add_node(path_node)
        >>> graph.add_edge("source", "path")
        >>> graph.validate()
    """

    def __init__(self, entry_node: Optional[str] = None):
        """Initialize an empty flow graph.

        Args:
            entry_node: Name of the entry node (set automatically
                when first node is added if not specified).
        """
        self._nodes: Dict[str, FlowNode] = {}
        self._edges: List[Edge] = []
        self._entry_node: Optional[str] = entry_node

        # Cached adjacency lists (rebuilt on modification)
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}
        self._dirty = True

        # Trigger callbacks
        self._trigger_callbacks: List[Callable[[FlowData], None]] = []

    @property
    def entry_node(self) -> Optional[str]:
        """Get the entry node name."""
        return self._entry_node

    @entry_node.setter
    def entry_node(self, name: str) -> None:
        """Set the entry node name."""
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not in graph")
        self._entry_node = name

    @property
    def nodes(self) -> Dict[str, FlowNode]:
        """Get all nodes in the graph."""
        return dict(self._nodes)

    @property
    def edges(self) -> List[Edge]:
        """Get all edges in the graph."""
        return list(self._edges)

    def add_node(self, node: FlowNode) -> "FlowGraph":
        """Add a node to the graph.

        Args:
            node: FlowNode to add.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in graph")

        self._nodes[node.name] = node

        # Set first node as entry if not specified
        if self._entry_node is None:
            self._entry_node = node.name

        self._dirty = True
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        path_filter: Optional[str] = None,
    ) -> "FlowGraph":
        """Add an edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
            path_filter: Optional path_id filter for conditional routing.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not in graph")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not in graph")

        edge = Edge(source=source, target=target, path_filter=path_filter)
        self._edges.append(edge)
        self._dirty = True
        return self

    def on_trigger(self, callback: Callable[[FlowData], None]) -> "FlowGraph":
        """Register a callback for trigger events.

        Called when FlowData reaches the end of the graph with
        should_trigger=True in any of its results.

        Args:
            callback: Function to call with triggering FlowData.

        Returns:
            Self for chaining.
        """
        self._trigger_callbacks.append(callback)
        return self

    def get_outgoing_edges(self, node_name: str) -> List[Edge]:
        """Get edges leaving a node.

        Args:
            node_name: Name of the source node.

        Returns:
            List of outgoing edges.
        """
        self._rebuild_adjacency()
        return self._outgoing.get(node_name, [])

    def get_incoming_edges(self, node_name: str) -> List[Edge]:
        """Get edges entering a node.

        Args:
            node_name: Name of the target node.

        Returns:
            List of incoming edges.
        """
        self._rebuild_adjacency()
        return self._incoming.get(node_name, [])

    def get_successors(self, node_name: str, path_id: Optional[str] = None) -> List[str]:
        """Get successor node names, optionally filtered by path_id.

        Args:
            node_name: Name of the source node.
            path_id: Optional path_id to filter edges.

        Returns:
            List of target node names.
        """
        edges = self.get_outgoing_edges(node_name)
        successors = []
        for edge in edges:
            if edge.path_filter is None or edge.path_filter == path_id:
                successors.append(edge.target)
        return successors

    def get_terminal_nodes(self) -> List[str]:
        """Get names of nodes with no outgoing edges.

        Returns:
            List of terminal node names.
        """
        self._rebuild_adjacency()
        return [
            name for name in self._nodes
            if not self._outgoing.get(name)
        ]

    def _rebuild_adjacency(self) -> None:
        """Rebuild adjacency lists if dirty."""
        if not self._dirty:
            return

        self._outgoing = {name: [] for name in self._nodes}
        self._incoming = {name: [] for name in self._nodes}

        for edge in self._edges:
            self._outgoing[edge.source].append(edge)
            self._incoming[edge.target].append(edge)

        self._dirty = False

    def validate(self) -> None:
        """Validate the graph structure.

        Checks:
        - Entry node is set and exists
        - No cycles (DAG property)
        - All nodes are reachable from entry

        Raises:
            ValueError: If validation fails.
        """
        # Check entry node
        if self._entry_node is None:
            raise ValueError("No entry node set")
        if self._entry_node not in self._nodes:
            raise ValueError(f"Entry node '{self._entry_node}' not in graph")

        # Check for cycles using DFS
        self._check_cycles()

        # Check reachability
        self._check_reachability()

    def _check_cycles(self) -> None:
        """Check for cycles using DFS with color marking."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {name: WHITE for name in self._nodes}

        def dfs(node: str) -> None:
            color[node] = GRAY
            for edge in self.get_outgoing_edges(node):
                target = edge.target
                if color[target] == GRAY:
                    raise ValueError(f"Cycle detected involving '{node}' -> '{target}'")
                if color[target] == WHITE:
                    dfs(target)
            color[node] = BLACK

        for node in self._nodes:
            if color[node] == WHITE:
                dfs(node)

    def _check_reachability(self) -> None:
        """Check that all nodes are reachable from entry."""
        if self._entry_node is None:
            return

        visited: Set[str] = set()
        queue = deque([self._entry_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for edge in self.get_outgoing_edges(node):
                queue.append(edge.target)

        unreachable = set(self._nodes.keys()) - visited
        if unreachable:
            raise ValueError(f"Unreachable nodes from entry: {unreachable}")

    def topological_order(self) -> List[str]:
        """Get nodes in topological order.

        Returns:
            List of node names in processing order.

        Raises:
            ValueError: If graph has cycles.
        """
        self._rebuild_adjacency()

        # Kahn's algorithm
        in_degree = {name: len(self._incoming.get(name, [])) for name in self._nodes}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self.get_outgoing_edges(node):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self._nodes):
            raise ValueError("Graph has cycles - cannot compute topological order")

        return result

    def fire_triggers(self, data: FlowData) -> None:
        """Fire trigger callbacks for data with triggering results.

        Args:
            data: FlowData that reached a terminal node.
        """
        # Check if any result should trigger
        should_fire = any(r.should_trigger for r in data.results)
        if should_fire:
            for callback in self._trigger_callbacks:
                callback(data)

    def initialize(self) -> None:
        """Initialize all nodes."""
        for node in self._nodes.values():
            node.initialize()

    def cleanup(self) -> None:
        """Clean up all nodes."""
        for node in self._nodes.values():
            node.cleanup()

    def __enter__(self) -> "FlowGraph":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
