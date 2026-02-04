"""FlowGraph to Pathway dataflow converter.

This module converts visualpath FlowGraph structures into Pathway
streaming dataflows using declarative NodeSpec dispatch.

Conversion mapping (spec-based):
- SourceSpec -> pw.io.python.read()
- ExtractSpec -> @pw.udf with PyObjectWrapper (per-extractor parallel branches)
- JoinSpec -> interval_join() with spec-defined window/lateness
- FilterSpec/ObservationFilterSpec/SignalFilterSpec -> pw.filter()
- SampleSpec -> frame_id modulo filter
- RateLimitSpec/TimestampSampleSpec -> pass-through (wall-clock based)
- BranchSpec/FanOutSpec/MultiBranchSpec/ConditionalFanOutSpec -> table replication
- CascadeFusionSpec/CollectorSpec -> pass-through
- spec=None -> pass-through (fallback to process())
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
    and creates corresponding Pathway operators based on each node's
    ``spec`` property.  Nodes without a spec (``spec is None``) are
    treated as pass-through, relying on ``process()`` for execution
    in non-Pathway backends.

    When ``ExtractSpec.parallel=True`` and there are independent
    extractor groups (no cross-dependencies), each group gets its
    own Pathway UDF so the engine can schedule them in parallel.
    The branches are then rejoined via ``interval_join``.

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
        """Convert a single node to Pathway operators using its spec."""
        from visualpath.flow.specs import (
            SourceSpec,
            ExtractSpec,
            JoinSpec,
            FilterSpec,
            ObservationFilterSpec,
            SignalFilterSpec,
            SampleSpec,
            RateLimitSpec,
            TimestampSampleSpec,
            BranchSpec,
            FanOutSpec,
            MultiBranchSpec,
            ConditionalFanOutSpec,
            CascadeFusionSpec,
            CollectorSpec,
        )

        spec = node.spec

        if spec is None:
            # No spec — pass through from predecessor
            self._convert_passthrough_node(node, graph)
            return

        if isinstance(spec, SourceSpec):
            self._tables[node.name] = frames_table

        elif isinstance(spec, ExtractSpec):
            self._convert_extract_spec(node.name, spec, graph)

        elif isinstance(spec, JoinSpec):
            self._convert_join_spec(node.name, spec, graph)

        elif isinstance(spec, (FilterSpec, ObservationFilterSpec, SignalFilterSpec)):
            self._convert_filter_spec(node.name, spec, graph)

        elif isinstance(spec, SampleSpec):
            self._convert_sample_spec(node.name, spec, graph)

        elif isinstance(spec, (RateLimitSpec, TimestampSampleSpec)):
            self._convert_passthrough_node(node, graph)

        elif isinstance(spec, (BranchSpec, FanOutSpec, MultiBranchSpec, ConditionalFanOutSpec)):
            self._convert_branch_node(node, graph)

        elif isinstance(spec, (CascadeFusionSpec, CollectorSpec)):
            self._convert_passthrough_node(node, graph)

        else:
            self._convert_passthrough_node(node, graph)

    # ------------------------------------------------------------------
    # Helper: get input table from predecessor
    # ------------------------------------------------------------------

    def _get_input_table(self, node_name: str, graph: "FlowGraph"):
        """Get the Pathway table from the first predecessor of a node."""
        predecessors = graph.get_incoming_edges(node_name)
        if not predecessors:
            return None
        pred_name = predecessors[0].source
        return self._tables.get(pred_name)

    # ------------------------------------------------------------------
    # Pass-through / branch
    # ------------------------------------------------------------------

    def _convert_passthrough_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert a node as pass-through from predecessor."""
        input_table = self._get_input_table(node.name, graph)
        if input_table is not None:
            self._tables[node.name] = input_table

    def _convert_branch_node(self, node: "FlowNode", graph: "FlowGraph") -> None:
        """Convert branching nodes (pass-through — table replication)."""
        input_table = self._get_input_table(node.name, graph)
        if input_table is not None:
            self._tables[node.name] = input_table

    # ------------------------------------------------------------------
    # ExtractSpec — per-extractor parallel branches
    # ------------------------------------------------------------------

    def _convert_extract_spec(
        self,
        node_name: str,
        spec: "ExtractSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert ExtractSpec to Pathway UDF(s).

        When ``spec.parallel=True`` and there are independent extractor
        groups (based on dependency analysis), each group gets its own
        UDF so Pathway can schedule them in parallel.  The branches are
        then rejoined via ``interval_join``.
        """
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        extractors = list(spec.extractors)
        if not extractors:
            self._tables[node_name] = input_table
            return

        # Single UDF path: no parallelism or single extractor
        if not spec.parallel or len(extractors) == 1:
            self._tables[node_name] = self._build_single_udf(
                node_name, extractors, input_table,
            )
            return

        # Dependency graph analysis: split into independent groups
        groups = self._split_by_dependency(extractors)

        if len(groups) == 1:
            # All extractors in a dependency chain — single UDF
            self._tables[node_name] = self._build_single_udf(
                node_name, extractors, input_table,
            )
            return

        # Independent groups — separate UDFs, then interval_join
        branch_tables = []
        for i, group in enumerate(groups):
            branch_name = f"{node_name}__branch_{i}"
            branch_table = self._build_single_udf(
                branch_name, group, input_table,
            )
            branch_tables.append(branch_table)
            self._tables[branch_name] = branch_table

        # Auto interval_join to merge branches
        joined = self._auto_join(branch_tables, spec.join_window_ns)
        self._tables[node_name] = joined

    def _build_single_udf(
        self,
        name: str,
        extractors: list,
        input_table: "pw.Table",
    ) -> "pw.Table":
        """Build a single Pathway UDF that runs extractors sequentially."""
        from visualpath.backends.pathway.operators import create_multi_extractor_udf

        raw_udf = create_multi_extractor_udf(extractors)

        @pw.udf
        def extract_udf(
            frame_wrapped: pw.PyObjectWrapper,
        ) -> pw.PyObjectWrapper:
            frame = frame_wrapped.value
            results = raw_udf(frame)
            return pw.PyObjectWrapper(results)

        return input_table.select(
            frame_id=pw.this.frame_id,
            t_ns=pw.this.t_ns,
            frame=pw.this.frame,
            results=extract_udf(pw.this.frame),
        )

    @staticmethod
    def _split_by_dependency(extractors: list) -> List[list]:
        """Split extractors into independent groups by dependency graph.

        Extractors sharing a dependency chain (direct or transitive) are
        placed in the same group.  Extractors with no dependency
        relationship are placed in separate groups.

        Example::

            [face_detect, pose_detect, face_expression(depends=face_detect)]
            -> [[face_detect, face_expression], [pose_detect]]

        Returns:
            List of extractor groups.  Each group preserves insertion order.
        """
        # Build name -> extractor map
        by_name = {ext.name: ext for ext in extractors}

        # Build undirected adjacency (connected component analysis)
        adj: Dict[str, set] = {ext.name: set() for ext in extractors}
        for ext in extractors:
            for dep_name in (ext.depends or []):
                if dep_name in adj:
                    adj[ext.name].add(dep_name)
                    adj[dep_name].add(ext.name)

        # Find connected components via BFS
        visited: set = set()
        components: List[List[str]] = []
        for ext in extractors:
            if ext.name in visited:
                continue
            # BFS
            component = []
            queue = [ext.name]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbour in adj.get(current, set()):
                    if neighbour not in visited:
                        queue.append(neighbour)
            components.append(component)

        # Convert name lists back to extractor lists (preserving order)
        name_order = {ext.name: i for i, ext in enumerate(extractors)}
        groups = []
        for comp in components:
            comp_sorted = sorted(comp, key=lambda n: name_order[n])
            groups.append([by_name[n] for n in comp_sorted])

        return groups

    def _auto_join(
        self,
        tables: List["pw.Table"],
        window_ns: int,
    ) -> "pw.Table":
        """Join multiple branch tables via interval_join."""
        if len(tables) == 1:
            return tables[0]

        joined = tables[0]
        for table in tables[1:]:
            joined = joined.interval_join(
                table,
                pw.left.t_ns,
                pw.right.t_ns,
                pw.temporal.interval(-window_ns, window_ns),
            ).select(
                frame_id=pw.left.frame_id,
                t_ns=pw.left.t_ns,
                frame=pw.left.frame,
            )

        return joined

    # ------------------------------------------------------------------
    # JoinSpec — temporal config from graph
    # ------------------------------------------------------------------

    def _convert_join_spec(
        self,
        node_name: str,
        spec: "JoinSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert JoinSpec to Pathway interval_join.

        Uses ``spec.window_ns`` (from graph) as the window size.
        Falls back to ``self._window_ns`` only if spec has no value.
        """
        input_paths = list(spec.input_paths)
        if len(input_paths) < 2:
            if input_paths:
                self._tables[node_name] = self._tables.get(input_paths[0])
            return

        tables = [self._tables.get(path) for path in input_paths]
        if None in tables:
            return

        # Prefer spec window, fall back to converter default
        window = spec.window_ns if spec.window_ns > 0 else self._window_ns

        left = tables[0]
        right = tables[1]

        joined = left.interval_join(
            right,
            pw.left.t_ns,
            pw.right.t_ns,
            pw.temporal.interval(-window, window),
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
                pw.temporal.interval(-window, window),
            ).select(
                frame_id=pw.left.frame_id,
                t_ns=pw.left.t_ns,
                frame=pw.left.frame,
            )

        self._tables[node_name] = joined

    # ------------------------------------------------------------------
    # FilterSpec variants
    # ------------------------------------------------------------------

    def _convert_filter_spec(
        self,
        node_name: str,
        spec,
        graph: "FlowGraph",
    ) -> None:
        """Convert filter specs to Pathway filter."""
        from visualpath.flow.specs import FilterSpec, ObservationFilterSpec, SignalFilterSpec

        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        if isinstance(spec, FilterSpec):
            condition = spec.condition

            @pw.udf
            def filter_udf(frame_wrapped: pw.PyObjectWrapper) -> bool:
                return condition(frame_wrapped.value)

            self._tables[node_name] = input_table.filter(
                filter_udf(pw.this.frame)
            )
        else:
            # ObservationFilterSpec, SignalFilterSpec — pass through in Pathway
            # (these operate on FlowData which is not available in the
            # Pathway table; they are handled by process() fallback)
            self._tables[node_name] = input_table

    # ------------------------------------------------------------------
    # SampleSpec
    # ------------------------------------------------------------------

    def _convert_sample_spec(
        self,
        node_name: str,
        spec: "SampleSpec",
        graph: "FlowGraph",
    ) -> None:
        """Convert SampleSpec to Pathway sampling via frame_id modulo."""
        input_table = self._get_input_table(node_name, graph)
        if input_table is None:
            return

        every_nth = spec.every_nth
        sampled = input_table.filter(
            pw.this.frame_id % every_nth == 0
        )
        self._tables[node_name] = sampled


__all__ = ["FlowGraphConverter"]
