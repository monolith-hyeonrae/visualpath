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

Debug hooks:
    >>> def on_interpret(event):
    ...     print(f"{event['phase']} {event['node']}: {event}")
    >>> interpreter = SimpleInterpreter(debug_hook=on_interpret)
"""

import logging
import operator
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from visualpath.flow.node import FlowData, FlowNode
from visualpath.flow.specs import (
    NodeSpec,
    SourceSpec,
    ModuleSpec,
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
    from visualpath.core.extractor import Observation


# Comparison operators for signal filters
_COMPARISONS: Dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


@dataclass
class DebugEvent:
    """Debug event emitted by interpreter hooks.

    Attributes:
        phase: 'enter', 'exit', or 'state_change'
        node: Node name
        spec_type: Type name of the spec being interpreted
        input_data: Input FlowData (for 'enter')
        output_data: Output FlowData list (for 'exit')
        output_count: Number of outputs produced
        elapsed_ms: Processing time in milliseconds (for 'exit')
        state_key: State key that changed (for 'state_change')
        state_value: New state value (for 'state_change')
        dropped: True if node filtered/dropped the data
        extra: Additional context-specific info
    """

    phase: str
    node: str
    spec_type: str
    input_data: Optional["FlowData"] = None
    output_data: Optional[List["FlowData"]] = None
    output_count: int = 0
    elapsed_ms: float = 0.0
    state_key: Optional[str] = None
    state_value: Any = None
    dropped: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.phase == "enter":
            frame_id = getattr(self.input_data.frame, "frame_id", "?") if self.input_data else "?"
            return f"[ENTER] {self.node} ({self.spec_type}) frame={frame_id}"
        elif self.phase == "exit":
            status = "DROPPED" if self.dropped else f"OUT={self.output_count}"
            return f"[EXIT]  {self.node} ({self.spec_type}) {status} ({self.elapsed_ms:.3f}ms)"
        elif self.phase == "state_change":
            return f"[STATE] {self.node}.{self.state_key} = {self.state_value}"
        return f"[{self.phase.upper()}] {self.node}"


# Type alias for debug hook callback
DebugHook = Callable[[DebugEvent], None]


class NodeProcessingError(Exception):
    """Raised when a user-provided callable within a node fails.

    Wraps the original exception with context about which node and
    spec type caused the error, enabling better diagnostics.

    Attributes:
        node_name: Name of the node where the error occurred.
        spec_type: Type name of the spec being interpreted.
        original_error: The underlying exception.
    """

    def __init__(self, node_name: str, spec_type: str, original_error: Exception):
        self.node_name = node_name
        self.spec_type = spec_type
        self.original_error = original_error
        super().__init__(
            f"Error in node '{node_name}' ({spec_type}): {original_error}"
        )


class SimpleInterpreter:
    """Spec-based interpreter for synchronous execution.

    Reads NodeSpec from FlowNode instances and executes them.
    Maintains per-node state (counters, buffers, timers) internally.

    This interpreter is used by SimpleBackend and GraphExecutor for
    sequential/local execution of flow graphs.

    Error handling:
        When ``on_error="raise"`` (default), errors from user-provided
        callables (conditions, fusion_fn, processors) are wrapped in
        :class:`NodeProcessingError` and re-raised.
        When ``on_error="drop"``, errors are logged and the data is
        dropped (empty output).

    Debug hooks:
        Pass a debug_hook callback to observe internal operations:

        >>> def my_hook(event: DebugEvent):
        ...     print(event)
        >>> interpreter = SimpleInterpreter(debug_hook=my_hook)

        Or use the built-in print hook:

        >>> interpreter = SimpleInterpreter(debug=True)
    """

    def __init__(
        self,
        debug: bool = False,
        debug_hook: Optional[DebugHook] = None,
        on_error: str = "raise",
    ) -> None:
        # Per-node mutable state, keyed by node name
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._debug = debug
        self._debug_hook = debug_hook
        if on_error not in ("raise", "drop"):
            raise ValueError(f"on_error must be 'raise' or 'drop', got '{on_error}'")
        self._on_error = on_error
        self._current_node: str = ""  # Set during interpret()

    def _emit_debug(self, event: DebugEvent) -> None:
        """Emit a debug event to registered hooks."""
        if self._debug:
            print(event)
        if self._debug_hook is not None:
            self._debug_hook(event)

    def _emit_state_change(
        self, node_name: str, key: str, value: Any, spec_type: str = ""
    ) -> None:
        """Emit a state change debug event."""
        if self._debug or self._debug_hook is not None:
            self._emit_debug(DebugEvent(
                phase="state_change",
                node=node_name,
                spec_type=spec_type,
                state_key=key,
                state_value=value,
            ))

    def _handle_error(
        self, spec_type: str, error: Exception
    ) -> List[FlowData]:
        """Handle an error from a user-provided callable.

        Uses self._current_node for the actual graph node name.
        If on_error="drop", logs the error and returns empty list.
        If on_error="raise", wraps and re-raises as NodeProcessingError.
        """
        node_name = self._current_node
        if self._on_error == "drop":
            logger.error(
                "Error in node '%s' (%s): %s — data dropped",
                node_name, spec_type, error,
            )
            if self._debug or self._debug_hook is not None:
                self._emit_debug(DebugEvent(
                    phase="error",
                    node=node_name,
                    spec_type=spec_type,
                    extra={"error": str(error), "error_type": type(error).__name__},
                ))
            return []
        raise NodeProcessingError(node_name, spec_type, error) from error

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
        spec_type = type(spec).__name__
        self._current_node = node.name
        has_debug = self._debug or self._debug_hook is not None

        # Emit enter event
        if has_debug:
            self._emit_debug(DebugEvent(
                phase="enter",
                node=node.name,
                spec_type=spec_type,
                input_data=data,
            ))

        start_time = time.perf_counter() if has_debug else 0

        match spec:
            case SourceSpec():
                outputs = self._interpret_source(spec, data)
            case ModuleSpec():
                outputs = self._interpret_modules(node, spec, data)
            case ExtractSpec():
                outputs = self._interpret_extract(node, spec, data)
            case FilterSpec():
                outputs = self._interpret_filter(spec, data)
            case ObservationFilterSpec():
                outputs = self._interpret_observation_filter(spec, data)
            case SignalFilterSpec():
                outputs = self._interpret_signal_filter(spec, data)
            case SampleSpec():
                outputs = self._interpret_sample(node.name, spec, data)
            case RateLimitSpec():
                outputs = self._interpret_rate_limit(node.name, spec, data)
            case TimestampSampleSpec():
                outputs = self._interpret_timestamp_sample(node.name, spec, data)
            case BranchSpec():
                outputs = self._interpret_branch(spec, data)
            case FanOutSpec():
                outputs = self._interpret_fanout(spec, data)
            case MultiBranchSpec():
                outputs = self._interpret_multi_branch(spec, data)
            case ConditionalFanOutSpec():
                outputs = self._interpret_conditional_fanout(spec, data)
            case JoinSpec():
                outputs = self._interpret_join(node.name, spec, data)
            case CascadeFusionSpec():
                outputs = self._interpret_cascade_fusion(spec, data)
            case CollectorSpec():
                outputs = self._interpret_collector(node.name, spec, data)
            case CustomSpec():
                outputs = self._interpret_custom(spec, data)
            case _:
                raise TypeError(
                    f"SimpleInterpreter does not know how to interpret "
                    f"{type(spec).__name__} from node '{node.name}'"
                )

        # Emit exit event
        if has_debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._emit_debug(DebugEvent(
                phase="exit",
                node=node.name,
                spec_type=spec_type,
                output_data=outputs,
                output_count=len(outputs),
                elapsed_ms=elapsed_ms,
                dropped=len(outputs) == 0,
            ))

        return outputs

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
    # Modules (unified extractor/fusion processing)
    # -----------------------------------------------------------------

    def _interpret_modules(
        self, node: FlowNode, spec: ModuleSpec, data: FlowData
    ) -> List[FlowData]:
        """Run modules on the frame.

        Modules are processed in order. Each module can depend on
        previous modules' outputs. Observations with should_trigger=True
        are added to results.
        """
        from visualpath.core.extractor import Observation

        frame = data.frame
        if frame is None:
            return [data]

        # Build dependency map from existing data
        deps: Dict[str, Any] = {
            obs.source: obs for obs in data.observations
        }
        # Also include existing results by source if available
        for result in data.results:
            if hasattr(result, 'source') and result.source:
                deps[result.source] = result

        observations: List["Observation"] = []
        results: List["Observation"] = []

        for module in spec.modules:
            # Collect dependencies for this module
            module_deps = None
            if hasattr(module, 'depends') and module.depends:
                module_deps = {
                    name: deps[name]
                    for name in module.depends
                    if name in deps
                }

            # Call module.process() or legacy extract()
            output = self._call_module(module, frame, module_deps)

            if output is not None:
                # Route output based on should_trigger property
                observations.append(output)
                deps[module.name] = output

                # If this is a trigger observation, also add to results
                if output.should_trigger:
                    results.append(output)

        # Update FlowData
        result = data.clone(
            observations=list(data.observations) + observations,
            results=list(data.results) + results,
            path_id=node.name,
        )

        return [result]

    def _call_module(
        self,
        module: Any,
        frame: Any,
        deps: Optional[Dict[str, Any]],
    ) -> Any:
        """Call a module's process method with backward compatibility.

        Handles both:
        - New Module interface: process(frame, deps)
        - Legacy Extractor: extract(frame, deps)
        - Legacy Fusion: update(observation)
        """
        # Try new Module.process() first
        if hasattr(module, 'process'):
            try:
                return module.process(frame, deps)
            except TypeError:
                # Might be old signature
                return module.process(frame)

        # Legacy Extractor
        if hasattr(module, 'extract'):
            try:
                return module.extract(frame, deps)
            except TypeError:
                return module.extract(frame)

        # Legacy Fusion (takes Observation, not frame)
        if hasattr(module, 'update'):
            # Fusion needs observation from deps
            if deps:
                for obs in deps.values():
                    if hasattr(obs, 'source'):  # It's an Observation
                        return module.update(obs)
            return None

        raise TypeError(
            f"Module {getattr(module, 'name', module)} has no "
            f"process(), extract(), or update() method"
        )

    # -----------------------------------------------------------------
    # Extract (legacy - uses extractors + optional fusion)
    # -----------------------------------------------------------------

    def _interpret_extract(
        self, node: FlowNode, spec: ExtractSpec, data: FlowData
    ) -> List[FlowData]:
        """Run extractors on the frame, optionally apply fusion.

        DEPRECATED: Use ModuleSpec with _interpret_modules instead.
        """
        from visualpath.core.extractor import Observation

        frame = data.frame
        if frame is None:
            return [data]

        # If spec has unified modules, delegate to _interpret_modules
        if spec.modules:
            module_spec = ModuleSpec(
                modules=spec.modules,
                parallel=spec.parallel,
                join_window_ns=spec.join_window_ns,
            )
            return self._interpret_modules(node, module_spec, data)

        # Legacy: separate extractors and fusion
        # Build dependency map from existing observations
        deps: Dict[str, "Observation"] = {
            obs.source: obs for obs in data.observations
        }
        observations: List["Observation"] = []

        for extractor in spec.extractors:
            extractor_deps = None
            if hasattr(extractor, 'depends') and extractor.depends:
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

        # Optionally run fusion/trigger module
        if spec.run_fusion and spec.fusion is not None:
            results: List["Observation"] = []
            fusion = spec.fusion

            for obs in observations:
                # Support both legacy .update() and new .process()
                if hasattr(fusion, 'update'):
                    fusion_result = fusion.update(obs)
                elif hasattr(fusion, 'process'):
                    # New Module interface - pass deps
                    deps_for_fusion = {obs.source: obs}
                    fusion_result = fusion.process(frame, deps_for_fusion)
                else:
                    continue

                if fusion_result is not None:
                    # Only add to results if it's a trigger
                    if fusion_result.should_trigger:
                        results.append(fusion_result)

            result = result.clone(
                results=list(result.results) + results,
            )

        return [result]

    def _call_extract(
        self,
        extractor: Any,
        frame: Any,
        deps: Optional[Dict[str, "Observation"]],
    ) -> Optional["Observation"]:
        """Call extractor.extract with backward compatibility.

        Also supports new Module.process() interface.
        """
        # Try new Module.process() first
        if hasattr(extractor, 'process') and not hasattr(extractor, 'extract'):
            try:
                return extractor.process(frame, deps)
            except TypeError:
                return extractor.process(frame)

        # Legacy extract()
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
        try:
            if spec.condition(data):
                return [data]
        except Exception as e:
            return self._handle_error("FilterSpec", e)
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
        self._emit_state_change(node_name, "count", count, "SampleSpec")

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
            self._emit_state_change(node_name, "last_time", now, "RateLimitSpec")
            return [data]

        elapsed_ms = (now - last_time) * 1000
        if elapsed_ms >= spec.min_interval_ms:
            state["last_time"] = now
            self._emit_state_change(node_name, "last_time", now, "RateLimitSpec")
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
            self._emit_state_change(
                node_name, "last_timestamp_ns", data.timestamp_ns, "TimestampSampleSpec"
            )
            return [data]

        if data.timestamp_ns - last_ts >= spec.interval_ns:
            state["last_timestamp_ns"] = data.timestamp_ns
            self._emit_state_change(
                node_name, "last_timestamp_ns", data.timestamp_ns, "TimestampSampleSpec"
            )
            return [data]
        return []

    # -----------------------------------------------------------------
    # Branching
    # -----------------------------------------------------------------

    def _interpret_branch(
        self, spec: BranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Binary branch: route to if_true or if_false path."""
        try:
            if spec.condition(data):
                return [data.with_path(spec.if_true)]
            return [data.with_path(spec.if_false)]
        except Exception as e:
            return self._handle_error("BranchSpec", e)

    def _interpret_fanout(
        self, spec: FanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Fan-out: clone data for each path."""
        return [data.clone(path_id=path) for path in spec.paths]

    def _interpret_multi_branch(
        self, spec: MultiBranchSpec, data: FlowData
    ) -> List[FlowData]:
        """Multi-way branch: route to first matching condition."""
        try:
            for condition, path_id in spec.branches:
                if condition(data):
                    return [data.with_path(path_id)]
        except Exception as e:
            return self._handle_error("MultiBranchSpec", e)
        if spec.default is not None:
            return [data.with_path(spec.default)]
        return []

    def _interpret_conditional_fanout(
        self, spec: ConditionalFanOutSpec, data: FlowData
    ) -> List[FlowData]:
        """Conditional fan-out: clone for each passing condition."""
        results = []
        try:
            for path_id, condition in spec.paths:
                if condition(data):
                    results.append(data.clone(path_id=path_id))
        except Exception as e:
            return self._handle_error("ConditionalFanOutSpec", e)
        return results

    # -----------------------------------------------------------------
    # Join
    # -----------------------------------------------------------------

    def _interpret_join(
        self, node_name: str, spec: JoinSpec, data: FlowData
    ) -> List[FlowData]:
        """Join: buffer data from multiple paths, emit when complete.

        Temporal semantics (when window_ns > 0):
        - Data from different paths is only joined if their timestamps
          are within ``window_ns`` of each other.
        - When ``lateness_ns > 0``, buffered data older than
          ``lateness_ns`` relative to the newest arrival is evicted.
        """
        state = self._state[node_name]

        # Unknown path → pass through
        if data.path_id not in spec.input_paths:
            return [data]

        # Buffer
        buffers = state.setdefault("buffers", {})
        buffers[data.path_id] = data
        self._emit_state_change(
            node_name, f"buffers[{data.path_id}]", "buffered", "JoinSpec"
        )

        # Evict stale data beyond lateness_ns
        if spec.lateness_ns > 0 and len(buffers) > 1:
            newest_ts = max(d.timestamp_ns for d in buffers.values())
            stale_paths = [
                pid for pid, d in buffers.items()
                if (newest_ts - d.timestamp_ns) > spec.lateness_ns
            ]
            for pid in stale_paths:
                del buffers[pid]
                self._emit_state_change(
                    node_name, f"buffers[{pid}]", "evicted (late)", "JoinSpec"
                )

        # Check temporal alignment (window_ns)
        if spec.window_ns > 0 and len(buffers) > 1:
            timestamps = [d.timestamp_ns for d in buffers.values()]
            spread = max(timestamps) - min(timestamps)
            if spread > spec.window_ns:
                # Timestamps too far apart — don't join yet
                return []

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
        try:
            result = spec.fusion_fn(data)
        except Exception as e:
            return self._handle_error("CascadeFusionSpec", e)
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
        self._emit_state_change(
            node_name, "buffer_size", len(buffer), "CollectorSpec"
        )

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

    # -----------------------------------------------------------------
    # Custom
    # -----------------------------------------------------------------

    def _interpret_custom(
        self, spec: CustomSpec, data: FlowData
    ) -> List[FlowData]:
        """Execute a user-provided custom processor."""
        try:
            result = spec.processor(data)
        except Exception as e:
            return self._handle_error("CustomSpec", e)
        if isinstance(result, list):
            return result
        return [result]


__all__ = ["SimpleInterpreter", "NodeProcessingError", "DebugEvent", "DebugHook"]
