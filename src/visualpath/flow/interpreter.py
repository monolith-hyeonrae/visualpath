"""SimpleInterpreter — spec-based execution engine.

The interpreter reads NodeSpec from each FlowNode and executes accordingly.
This is the "interpreter" half of the AST/interpreter pattern:

    FlowGraph = AST
    NodeSpec  = token
    Interpreter = this module

Each spec type maps to an interpret_* method that contains
the execution logic previously spread across node.process() methods.

Stateful interpreters maintain per-node state (counters, buffers, timers)
in a separate dict, keeping nodes themselves stateless/declarative.

Example:
    >>> from visualpath.flow.interpreter import SimpleInterpreter
    >>> interpreter = SimpleInterpreter()
    >>> results = interpreter.interpret(node, data)
"""

import operator
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowData, FlowNode
from visualpath.flow.specs import (
    NodeSpec,
    SourceSpec,
    ExtractSpec,
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
    JoinSpec,
    CascadeFusionSpec,
    CollectorSpec,
    CustomSpec,
)

if TYPE_CHECKING:
    from visualpath.core.extractor import BaseExtractor, Observation
    from visualpath.core.fusion import BaseFusion


# Comparison operators for signal filters
_COMPARISONS: Dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


class SimpleInterpreter:
    """Spec-based interpreter for synchronous execution.

    Reads NodeSpec from FlowNode instances and executes them.
    Maintains per-node state (counters, buffers, timers) internally.

    This interpreter is used by SimpleBackend and GraphExecutor for
    sequential/local execution of flow graphs.
    """

    def __init__(self) -> None:
        # Per-node mutable state, keyed by node name
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def reset(self) -> None:
        """Clear all interpreter state."""
        self._state.clear()

    def reset_node(self, node_name: str) -> None:
        """Clear state for a specific node."""
        self._state.pop(node_name, None)

    def interpret(self, node: FlowNode, data: FlowData) -> List[FlowData]:
        """Interpret a node's spec and execute it on the given data.

        Args:
            node: The FlowNode whose spec to interpret.
            data: Input FlowData to process.

        Returns:
            List of output FlowData (may be empty if filtered/buffered).

        Raises:
            TypeError: If the spec type is not recognized.
        """
        spec = node.spec

        match spec:
            case SourceSpec():
                return self._interpret_source(spec, data)
            case ExtractSpec():
                return self._interpret_extract(node, spec, data)
            case FilterSpec():
                return self._interpret_filter(spec, data)
            case ObservationFilterSpec():
                return self._interpret_observation_filter(spec, data)
            case SignalFilterSpec():
                return self._interpret_signal_filter(spec, data)
            case SampleSpec():
                return self._interpret_sample(node.name, spec, data)
            case RateLimitSpec():
                return self._interpret_rate_limit(node.name, spec, data)
            case TimestampSampleSpec():
                return self._interpret_timestamp_sample(node.name, spec, data)
            case BranchSpec():
                return self._interpret_branch(spec, data)
            case FanOutSpec():
                return self._interpret_fanout(spec, data)
            case MultiBranchSpec():
                return self._interpret_multi_branch(spec, data)
            case ConditionalFanOutSpec():
                return self._interpret_conditional_fanout(spec, data)
            case JoinSpec():
                return self._interpret_join(node.name, spec, data)
            case CascadeFusionSpec():
                return self._interpret_cascade_fusion(spec, data)
            case CollectorSpec():
                return self._interpret_collector(node.name, spec, data)
            case CustomSpec():
                return self._interpret_custom(spec, data)
            case _:
                raise TypeError(
                    f"SimpleInterpreter does not know how to interpret "
                    f"{type(spec).__name__} from node '{node.name}'"
                )

    def flush_node(self, node: FlowNode) -> List[FlowData]:
        """Flush any buffered data for a stateful node.

        Used for JoinNode (pending buffers) and CollectorNode (partial batches).
        """
        spec = node.spec
        match spec:
            case JoinSpec():
                return self._flush_join(node.name, spec)
            case CollectorSpec():
                return self._flush_collector(node.name, spec)
            case _:
                return []

    # -----------------------------------------------------------------
    # Source
    # -----------------------------------------------------------------

    def _interpret_source(
        self, spec: SourceSpec, data: FlowData
    ) -> List[FlowData]:
        """Source just passes data through with the default path_id."""
        return [data]

    # -----------------------------------------------------------------
    # Extract (runs extractors + optional fusion)
    # -----------------------------------------------------------------

    def _interpret_extract(
        self, node: FlowNode, spec: ExtractSpec, data: FlowData
    ) -> List[FlowData]:
        """Run extractors on the frame, optionally apply fusion."""
        frame = data.frame
        if frame is None:
            return [data]

        # Build dependency map from existing observations
        deps: Dict[str, "Observation"] = {
            obs.source: obs for obs in data.observations
        }
        observations: List["Observation"] = []

        for extractor in spec.extractors:
            extractor_deps = None
            if extractor.depends:
                extractor_deps = {
                    name: deps[name]
                    for name in extractor.depends
                    if name in deps
                }

            obs = self._call_extract(extractor, frame, extractor_deps)
            if obs is not None:
                observations.append(obs)
                deps[extractor.name] = obs

        # Update FlowData with observations
        result = data.clone(
            observations=list(data.observations) + observations,
            path_id=node.name,
        )

        # Optionally run fusion
        if spec.run_fusion and spec.fusion is not None:
            from visualpath.core.fusion import FusionResult

            results: List[FusionResult] = []
            for obs in observations:
                fusion_result = spec.fusion.update(obs)
                results.append(fusion_result)

            result = result.clone(
                results=list(result.results) + results,
            )

        return [result]

    def _call_extract(
        self,
        extractor: "BaseExtractor",
        frame: Any,
        deps: Optional[Dict[str, "Observation"]],
    ) -> Optional["Observation"]:
        """Call extractor.extract with backward compatibility."""
        try:
            return extractor.extract(frame, deps)
        except TypeError:
            return extractor.extract(frame)

    # -----------------------------------------------------------------
    # Filters
    # -----------------------------------------------------------------

    def _interpret_filter(
        self, spec: FilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if condition is true, drop otherwise."""
        if spec.condition(data):
            return [data]
        return []

    def _interpret_observation_filter(
        self, spec: ObservationFilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if enough observations exist."""
        if len(data.observations) >= spec.min_count:
            return [data]
        return []

    def _interpret_signal_filter(
        self, spec: SignalFilterSpec, data: FlowData
    ) -> List[FlowData]:
        """Pass data if any observation's signal passes threshold."""
        cmp_fn = _COMPARISONS.get(spec.comparison, operator.gt)
        for obs in data.observations:
            value = obs.signals.get(spec.signal_name)
            if value is not None and cmp_fn(value, spec.threshold):
                return [data]
        return []

    # -----------------------------------------------------------------
    # Samplers
    # -----------------------------------------------------------------

    def _interpret_sample(
        self, node_name: str, spec: SampleSpec, data: FlowData
    ) -> List[FlowData]:
        """Every-Nth sampler using internal counter."""
        state = self._state[node_name]
        count = state.get("count", 0) + 1
        state["count"] = count

        if count % spec.every_nth == 0:
            return [data]
        return []

    def _interpret_rate_limit(
        self, node_name: str, spec: RateLimitSpec, data: FlowData
    ) -> List[FlowData]:
        """Rate limiter using wall-clock time."""
        state = self._state[node_name]
        now = time.monotonic()
        last_time = state.get("last_time")

        if last_time is None:
            state["last_time"] = now
            return [data]

        elapsed_ms = (now - last_time) * 1000
        if elapsed_ms >= spec.min_interval_ms:
            state["last_time"] = now
            return [data]
        return []

    def _interpret_timestamp_sample(
        self, node_name: str, spec: TimestampSampleSpec, data: FlowData
    ) -> List[FlowData]:
        """Timestamp-based sampler using data timestamps."""
        state = self._state[node_name]
        last_ts = state.get("last_timestamp_ns")

        if last_ts is None:
            state["last_timestamp_ns"] = data.timestamp_ns
            return [data]

        if data.timestamp_ns - last_ts >= spec.interval_ns:
            state["last_timestamp_ns"] = data.timestamp_ns
            return [data]
        return []

    # -----------------------------------------------------------------
    # Branching
    # -----------------------------------------------------------------

    def _interpret_branch(
        self, spec: BranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Binary branch: route to if_true or if_false path."""
        if spec.condition(data):
            return [data.with_path(spec.if_true)]
        return [data.with_path(spec.if_false)]

    def _interpret_fanout(
        self, spec: FanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Fan-out: clone data for each path."""
        return [data.clone(path_id=path) for path in spec.paths]

    def _interpret_multi_branch(
        self, spec: MultiBranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Multi-way branch: route to first matching condition."""
        for condition, path_id in spec.branches:
            if condition(data):
                return [data.with_path(path_id)]
        if spec.default is not None:
            return [data.with_path(spec.default)]
        return []

    def _interpret_conditional_fanout(
        self, spec: ConditionalFanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Conditional fan-out: clone for each passing condition."""
        results = []
        for path_id, condition in spec.paths:
            if condition(data):
                results.append(data.clone(path_id=path_id))
        return results

    # -----------------------------------------------------------------
    # Join
    # -----------------------------------------------------------------

    def _interpret_join(
        self, node_name: str, spec: JoinSpec, data: FlowData
    ) -> List[FlowData]:
        """Join: buffer data from multiple paths, emit when complete."""
        state = self._state[node_name]

        # Unknown path → pass through
        if data.path_id not in spec.input_paths:
            return [data]

        # Buffer
        buffers = state.setdefault("buffers", {})
        buffers[data.path_id] = data

        # Check emit condition
        if spec.mode == "any":
            return self._emit_join(node_name, spec)
        elif spec.mode == "all":
            if set(buffers.keys()) >= set(spec.input_paths):
                return self._emit_join(node_name, spec)
        return []

    def _emit_join(
        self, node_name: str, spec: JoinSpec
    ) -> List[FlowData]:
        """Emit merged data and clear buffers."""
        state = self._state[node_name]
        buffers: Dict[str, FlowData] = state.get("buffers", {})

        if not buffers:
            return []

        # Merge observations and results from all buffered data
        merged_observations: List = []
        merged_results: List = []
        merged_metadata: Dict = {}
        first_data = next(iter(buffers.values()))

        for path_data in buffers.values():
            if spec.merge_observations:
                merged_observations.extend(path_data.observations)
            if spec.merge_results:
                merged_results.extend(path_data.results)
            merged_metadata.update(path_data.metadata)

        merged = first_data.clone(
            path_id=spec.output_path_id,
            observations=merged_observations,
            results=merged_results,
            metadata=merged_metadata,
        )

        # Clear buffers
        state["buffers"] = {}

        return [merged]

    def _flush_join(
        self, node_name: str, spec: JoinSpec
    ) -> List[FlowData]:
        """Flush pending join buffers."""
        return self._emit_join(node_name, spec)

    # -----------------------------------------------------------------
    # Cascade Fusion
    # -----------------------------------------------------------------

    def _interpret_cascade_fusion(
        self, spec: CascadeFusionSpec, data: FlowData
    ) -> List[FlowData]:
        """Apply cascade fusion function to data."""
        result = spec.fusion_fn(data)
        return [result]

    # -----------------------------------------------------------------
    # Collector
    # -----------------------------------------------------------------

    def _interpret_collector(
        self, node_name: str, spec: CollectorSpec, data: FlowData
    ) -> List[FlowData]:
        """Accumulate data into batches."""
        state = self._state[node_name]
        buffer: List[FlowData] = state.setdefault("buffer", [])
        buffer.append(data)

        if len(buffer) >= spec.batch_size:
            return self._emit_collector(node_name, spec)
        return []

    def _emit_collector(
        self, node_name: str, spec: CollectorSpec
    ) -> List[FlowData]:
        """Emit collected batch."""
        state = self._state[node_name]
        buffer: List[FlowData] = state.get("buffer", [])

        if not buffer:
            return []

        # Merge all buffered data
        merged_observations: List = []
        merged_results: List = []
        for item in buffer:
            merged_observations.extend(item.observations)
            merged_results.extend(item.results)

        batch = buffer[0].clone(
            observations=merged_observations,
            results=merged_results,
            metadata={**buffer[0].metadata, "_batch_size": len(buffer)},
        )

        # Clear buffer
        state["buffer"] = []

        return [batch]

    def _flush_collector(
        self, node_name: str, spec: CollectorSpec
    ) -> List[FlowData]:
        """Flush partial batch if emit_partial is True."""
        if spec.emit_partial:
            return self._emit_collector(node_name, spec)
        return []


__all__ = ["SimpleInterpreter"]
