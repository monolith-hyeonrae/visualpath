"""Join nodes for merging multiple paths.

JoinNode merges data from multiple paths based on timestamp alignment.
CascadeFusionNode performs secondary fusion on merged data.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from visualpath.flow.node import FlowNode, FlowData


@dataclass
class JoinBuffer:
    """Buffer for accumulating data from multiple paths.

    Attributes:
        timestamp_ns: Timestamp for this join window.
        path_data: Mapping of path_id to FlowData.
        received_paths: Set of paths that have contributed data.
    """

    timestamp_ns: int
    path_data: Dict[str, FlowData] = field(default_factory=dict)
    received_paths: Set[str] = field(default_factory=set)

    def add(self, data: FlowData) -> None:
        """Add data from a path to the buffer."""
        self.path_data[data.path_id] = data
        self.received_paths.add(data.path_id)

    def is_complete(self, required_paths: Set[str]) -> bool:
        """Check if all required paths have contributed."""
        return self.received_paths >= required_paths


class JoinNode(FlowNode):
    """Node that merges data from multiple paths.

    JoinNode buffers data from multiple input paths and merges them
    when all expected paths have contributed data for a timestamp window.

    Merge modes:
    - "all": Wait for all input paths (inner join)
    - "any": Merge whenever any path has data (outer join)
    - "timeout": Merge after timeout even if incomplete

    Example:
        >>> join = JoinNode("merge", input_paths=["human", "scene"])
        >>> # Waits for both paths then merges observations
    """

    def __init__(
        self,
        name: str,
        input_paths: List[str],
        mode: str = "all",
        window_ns: int = 100_000_000,  # 100ms default
        merge_observations: bool = True,
        merge_results: bool = True,
        output_path_id: str = "merged",
    ):
        """Initialize the join node.

        Args:
            name: Unique name for this node.
            input_paths: List of path_ids to join.
            mode: Join mode ("all", "any").
            window_ns: Time window for grouping data (nanoseconds).
            merge_observations: Whether to merge observations from all paths.
            merge_results: Whether to merge results from all paths.
            output_path_id: path_id for merged output.
        """
        if not input_paths:
            raise ValueError("input_paths must not be empty")

        self._name = name
        self._input_paths = set(input_paths)
        self._mode = mode
        self._window_ns = window_ns
        self._merge_observations = merge_observations
        self._merge_results = merge_results
        self._output_path_id = output_path_id

        # Buffer keyed by quantized timestamp
        self._buffers: Dict[int, JoinBuffer] = {}

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def input_paths(self) -> Set[str]:
        """Get the expected input paths."""
        return set(self._input_paths)

    @property
    def spec(self):
        """Return JoinSpec for this node."""
        from visualpath.flow.specs import JoinSpec
        return JoinSpec(
            input_paths=tuple(sorted(self._input_paths)),
            mode=self._mode,
            window_ns=self._window_ns,
            lateness_ns=0,
            merge_observations=self._merge_observations,
            merge_results=self._merge_results,
            output_path_id=self._output_path_id,
        )

    def _quantize_timestamp(self, timestamp_ns: int) -> int:
        """Quantize timestamp to window boundary."""
        if self._window_ns <= 0:
            return timestamp_ns
        return (timestamp_ns // self._window_ns) * self._window_ns

    def _get_or_create_buffer(self, timestamp_ns: int) -> JoinBuffer:
        """Get or create buffer for a timestamp window."""
        key = self._quantize_timestamp(timestamp_ns)
        if key not in self._buffers:
            self._buffers[key] = JoinBuffer(timestamp_ns=key)
        return self._buffers[key]

    def _cleanup_old_buffers(self, current_ts: int) -> None:
        """Remove buffers older than 2x window size."""
        if self._window_ns <= 0:
            return

        cutoff = current_ts - (2 * self._window_ns)
        old_keys = [k for k in self._buffers if k < cutoff]
        for key in old_keys:
            del self._buffers[key]

    def _merge_buffer(self, buffer: JoinBuffer) -> FlowData:
        """Merge all data in a buffer into single FlowData."""
        all_observations = []
        all_results = []
        all_metadata: Dict[str, Any] = {}
        frame = None

        for path_id, data in buffer.path_data.items():
            if data.frame is not None and frame is None:
                frame = data.frame

            if self._merge_observations:
                all_observations.extend(data.observations)

            if self._merge_results:
                all_results.extend(data.results)

            # Merge metadata with path prefix
            for key, value in data.metadata.items():
                all_metadata[f"{path_id}_{key}"] = value

        # Add join metadata
        all_metadata["_joined_paths"] = list(buffer.received_paths)
        all_metadata["_join_timestamp_ns"] = buffer.timestamp_ns

        return FlowData(
            frame=frame,
            observations=all_observations,
            results=all_results,
            metadata=all_metadata,
            path_id=self._output_path_id,
            timestamp_ns=buffer.timestamp_ns,
        )

    def process(self, data: FlowData) -> List[FlowData]:
        """Buffer and potentially merge data.

        Args:
            data: Input FlowData from one of the input paths.

        Returns:
            Empty list if still buffering, merged FlowData if complete.
        """
        # Ignore data from unexpected paths
        if data.path_id not in self._input_paths:
            return [data]  # Pass through unchanged

        buffer = self._get_or_create_buffer(data.timestamp_ns)
        buffer.add(data)

        # Cleanup old buffers
        self._cleanup_old_buffers(data.timestamp_ns)

        # Check if we should emit
        should_emit = False

        if self._mode == "all":
            should_emit = buffer.is_complete(self._input_paths)
        elif self._mode == "any":
            should_emit = True

        if should_emit:
            merged = self._merge_buffer(buffer)
            # Remove buffer after emitting
            key = self._quantize_timestamp(data.timestamp_ns)
            if key in self._buffers:
                del self._buffers[key]
            return [merged]

        return []

    def flush(self) -> List[FlowData]:
        """Flush all pending buffers.

        Call this at end of stream to emit any remaining buffered data.

        Returns:
            List of merged FlowData for each buffer.
        """
        results = []
        for buffer in self._buffers.values():
            if buffer.received_paths:
                results.append(self._merge_buffer(buffer))
        self._buffers.clear()
        return results

    def reset(self) -> None:
        """Reset all buffers."""
        self._buffers.clear()


class CascadeFusionNode(FlowNode):
    """Node that performs secondary fusion on merged data.

    CascadeFusionNode runs a custom fusion function on merged FlowData,
    typically used after JoinNode to make final trigger decisions.

    Example:
        >>> def meta_fusion(data: FlowData) -> FlowData:
        ...     # Combine signals from multiple paths
        ...     total_score = sum(r.score for r in data.results)
        ...     if total_score > 0.8:
        ...         # Add final trigger result
        ...         ...
        ...     return data
        ...
        >>> cascade = CascadeFusionNode("meta", fusion_fn=meta_fusion)
    """

    def __init__(
        self,
        name: str,
        fusion_fn: Callable[[FlowData], FlowData],
    ):
        """Initialize the cascade fusion node.

        Args:
            name: Unique name for this node.
            fusion_fn: Function that takes FlowData and returns modified FlowData.
        """
        self._name = name
        self._fusion_fn = fusion_fn

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return CascadeFusionSpec for this node."""
        from visualpath.flow.specs import CascadeFusionSpec
        return CascadeFusionSpec(fusion_fn=self._fusion_fn)

    def process(self, data: FlowData) -> List[FlowData]:
        """Apply fusion function to data.

        Args:
            data: Input FlowData (typically from JoinNode).

        Returns:
            Single-item list with fusion result.
        """
        result = self._fusion_fn(data)
        return [result]


class CollectorNode(FlowNode):
    """Node that collects data into batches.

    CollectorNode accumulates FlowData until a condition is met,
    then emits all collected data as a list in metadata.

    Example:
        >>> collector = CollectorNode("batch", batch_size=10)
        >>> # Collects 10 items then emits with batch in metadata
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 0,
        timeout_ns: int = 0,
        emit_partial: bool = True,
    ):
        """Initialize the collector node.

        Args:
            name: Unique name for this node.
            batch_size: Number of items to collect before emitting (0 = no limit).
            timeout_ns: Max time to collect before emitting (0 = no timeout).
            emit_partial: Whether to emit partial batches on flush.
        """
        self._name = name
        self._batch_size = batch_size
        self._timeout_ns = timeout_ns
        self._emit_partial = emit_partial

        self._buffer: List[FlowData] = []
        self._first_timestamp_ns: int = 0

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def spec(self):
        """Return CollectorSpec for this node."""
        from visualpath.flow.specs import CollectorSpec
        return CollectorSpec(
            batch_size=self._batch_size,
            timeout_ns=self._timeout_ns,
            emit_partial=self._emit_partial,
        )

    def process(self, data: FlowData) -> List[FlowData]:
        """Collect data into batch.

        Args:
            data: Input FlowData.

        Returns:
            Empty list while collecting, merged FlowData when batch complete.
        """
        if not self._buffer:
            self._first_timestamp_ns = data.timestamp_ns

        self._buffer.append(data)

        should_emit = False

        # Check batch size
        if self._batch_size > 0 and len(self._buffer) >= self._batch_size:
            should_emit = True

        # Check timeout
        if self._timeout_ns > 0:
            elapsed = data.timestamp_ns - self._first_timestamp_ns
            if elapsed >= self._timeout_ns:
                should_emit = True

        if should_emit:
            return self._emit_batch()

        return []

    def _emit_batch(self) -> List[FlowData]:
        """Emit collected batch as single FlowData."""
        if not self._buffer:
            return []

        # Merge all observations and results
        all_observations = []
        all_results = []
        for item in self._buffer:
            all_observations.extend(item.observations)
            all_results.extend(item.results)

        # Use last frame and timestamp
        last_item = self._buffer[-1]

        result = FlowData(
            frame=last_item.frame,
            observations=all_observations,
            results=all_results,
            metadata={
                "_batch_size": len(self._buffer),
                "_batch_items": self._buffer,
            },
            path_id=last_item.path_id,
            timestamp_ns=last_item.timestamp_ns,
        )

        self._buffer = []
        return [result]

    def flush(self) -> List[FlowData]:
        """Flush any remaining buffered data."""
        if self._emit_partial and self._buffer:
            return self._emit_batch()
        self._buffer = []
        return []

    def reset(self) -> None:
        """Reset the collector."""
        self._buffer = []
        self._first_timestamp_ns = 0
