"""Flow control system for visualpath.

The flow module provides a DAG-based pipeline for video analysis,
enabling complex routing, branching, and merging of data.

Key Components:
- FlowData: Data container that flows between nodes
- FlowNode: Abstract base class for processing nodes
- FlowGraph: Directed acyclic graph of nodes
- GraphExecutor: Executes the flow graph on frames
- FlowGraphBuilder: Fluent API for graph construction

Example:
    >>> from visualpath.flow import FlowGraphBuilder, GraphExecutor
    >>>
    >>> graph = (FlowGraphBuilder()
    ...     .source("frames")
    ...     .sample(every_nth=3)
    ...     .branch(
    ...         condition=lambda d: has_face(d),
    ...         if_true="human",
    ...         if_false="scene",
    ...     )
    ...     .path("human", extractors=[face_ext])
    ...     .path("scene", extractors=[scene_ext])
    ...     .join(["human", "scene"])
    ...     .on_trigger(handle_trigger)
    ...     .build())
    >>>
    >>> executor = GraphExecutor(graph)
    >>> with executor:
    ...     for frame in video:
    ...         executor.process(frame)
"""

from visualpath.flow.node import FlowNode, FlowData, Condition
from visualpath.flow.graph import FlowGraph, Edge
from visualpath.flow.interpreter import SimpleInterpreter
from visualpath.flow.executor import GraphExecutor
from visualpath.flow.builder import FlowGraphBuilder

# Import all node types
from visualpath.flow.nodes import (
    SourceNode,
    PathNode,
    FilterNode,
    ObservationFilter,
    SignalThresholdFilter,
    SamplerNode,
    RateLimiterNode,
    TimestampSamplerNode,
    BranchNode,
    FanOutNode,
    MultiBranchNode,
    ConditionalFanOutNode,
    JoinNode,
    CascadeFusionNode,
    CollectorNode,
)

# Import all spec types
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

__all__ = [
    # Core
    "FlowNode",
    "FlowData",
    "Condition",
    "FlowGraph",
    "Edge",
    "SimpleInterpreter",
    "GraphExecutor",
    "FlowGraphBuilder",
    # Nodes
    "SourceNode",
    "PathNode",
    "FilterNode",
    "ObservationFilter",
    "SignalThresholdFilter",
    "SamplerNode",
    "RateLimiterNode",
    "TimestampSamplerNode",
    "BranchNode",
    "FanOutNode",
    "MultiBranchNode",
    "ConditionalFanOutNode",
    "JoinNode",
    "CascadeFusionNode",
    "CollectorNode",
    # Specs
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
