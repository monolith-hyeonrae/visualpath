"""Declarative node specifications for flow graph nodes.

NodeSpec types describe the semantics of each node type as frozen
dataclasses. Backends can inspect these specs to generate optimized
execution plans without accessing private node attributes.

Each FlowNode subclass exposes its spec via the ``spec`` property.
Custom nodes that don't override ``spec`` return ``None``, and
backends fall back to calling ``process()`` directly.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from visualpath.core.extractor import BaseExtractor
    from visualpath.core.fusion import BaseFusion
    from visualpath.flow.node import FlowData, Condition


@dataclass(frozen=True)
class NodeSpec:
    """Base spec for all flow nodes."""


@dataclass(frozen=True)
class SourceSpec(NodeSpec):
    """Spec for SourceNode.

    Attributes:
        default_path_id: Default path_id assigned to created FlowData.
    """

    default_path_id: str = "default"


@dataclass(frozen=True)
class ExtractSpec(NodeSpec):
    """Spec for PathNode (extractor execution).

    Attributes:
        extractors: List of extractors to run.
        fusion: Optional fusion module.
        parallel: Whether extractors can run in parallel.
        run_fusion: Whether to run fusion after extraction.
        join_window_ns: Window for auto-joining parallel extractor branches.
    """

    extractors: tuple = ()
    fusion: Optional[Any] = None
    parallel: bool = False
    run_fusion: bool = True
    join_window_ns: int = 100_000_000  # 100ms


@dataclass(frozen=True)
class FilterSpec(NodeSpec):
    """Spec for FilterNode.

    Attributes:
        condition: Condition function for filtering.
    """

    condition: Any = None  # Callable[[FlowData], bool]


@dataclass(frozen=True)
class ObservationFilterSpec(NodeSpec):
    """Spec for ObservationFilter.

    Attributes:
        min_count: Minimum number of observations required.
    """

    min_count: int = 1


@dataclass(frozen=True)
class SignalFilterSpec(NodeSpec):
    """Spec for SignalThresholdFilter.

    Attributes:
        signal_name: Name of the signal to check.
        threshold: Threshold value.
        comparison: Comparison operator ("gt", "ge", "lt", "le", "eq").
    """

    signal_name: str = ""
    threshold: float = 0.0
    comparison: str = "gt"


@dataclass(frozen=True)
class SampleSpec(NodeSpec):
    """Spec for SamplerNode.

    Attributes:
        every_nth: Pass every Nth frame.
    """

    every_nth: int = 1


@dataclass(frozen=True)
class RateLimitSpec(NodeSpec):
    """Spec for RateLimiterNode.

    Attributes:
        min_interval_ms: Minimum interval between frames in milliseconds.
    """

    min_interval_ms: float = 0


@dataclass(frozen=True)
class TimestampSampleSpec(NodeSpec):
    """Spec for TimestampSamplerNode.

    Attributes:
        interval_ns: Minimum interval between samples in nanoseconds.
    """

    interval_ns: int = 0


@dataclass(frozen=True)
class BranchSpec(NodeSpec):
    """Spec for BranchNode.

    Attributes:
        condition: Condition function for branching.
        if_true: path_id when condition is True.
        if_false: path_id when condition is False.
    """

    condition: Any = None  # Callable[[FlowData], bool]
    if_true: str = ""
    if_false: str = ""


@dataclass(frozen=True)
class FanOutSpec(NodeSpec):
    """Spec for FanOutNode.

    Attributes:
        paths: List of path_ids to replicate to.
    """

    paths: tuple = ()


@dataclass(frozen=True)
class MultiBranchSpec(NodeSpec):
    """Spec for MultiBranchNode.

    Attributes:
        branches: List of (condition, path_id) tuples.
        default: Default path_id if no condition matches.
    """

    branches: tuple = ()
    default: Optional[str] = None


@dataclass(frozen=True)
class ConditionalFanOutSpec(NodeSpec):
    """Spec for ConditionalFanOutNode.

    Attributes:
        paths: List of (path_id, condition) tuples.
    """

    paths: tuple = ()


@dataclass(frozen=True)
class JoinSpec(NodeSpec):
    """Spec for JoinNode.

    Attributes:
        input_paths: Sorted tuple of input path_ids.
        mode: Join mode ("all", "any").
        window_ns: Time window for grouping data (nanoseconds).
        lateness_ns: Allowed late arrival time (nanoseconds).
        merge_observations: Whether to merge observations.
        merge_results: Whether to merge results.
        output_path_id: path_id for merged output.
    """

    input_paths: tuple = ()
    mode: str = "all"
    window_ns: int = 100_000_000
    lateness_ns: int = 0
    merge_observations: bool = True
    merge_results: bool = True
    output_path_id: str = "merged"


@dataclass(frozen=True)
class CascadeFusionSpec(NodeSpec):
    """Spec for CascadeFusionNode.

    Attributes:
        fusion_fn: Fusion function.
    """

    fusion_fn: Any = None  # Callable[[FlowData], FlowData]


@dataclass(frozen=True)
class CollectorSpec(NodeSpec):
    """Spec for CollectorNode.

    Attributes:
        batch_size: Number of items to collect before emitting.
        timeout_ns: Max time to collect before emitting.
        emit_partial: Whether to emit partial batches on flush.
    """

    batch_size: int = 0
    timeout_ns: int = 0
    emit_partial: bool = True


@dataclass(frozen=True)
class CustomSpec(NodeSpec):
    """Spec for user-defined custom nodes.

    Attributes:
        processor: The processing function/callable.
        input_type: Description of expected input type.
        output_type: Description of expected output type.
    """

    processor: Any = None
    input_type: str = ""
    output_type: str = ""


__all__ = [
    "NodeSpec",
    "SourceSpec",
    "ExtractSpec",
    "FilterSpec",
    "ObservationFilterSpec",
    "SignalFilterSpec",
    "SampleSpec",
    "RateLimitSpec",
    "TimestampSampleSpec",
    "BranchSpec",
    "FanOutSpec",
    "MultiBranchSpec",
    "ConditionalFanOutSpec",
    "JoinSpec",
    "CascadeFusionSpec",
    "CollectorSpec",
    "CustomSpec",
]
