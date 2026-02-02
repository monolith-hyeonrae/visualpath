"""Flow node implementations.

This module provides various node types for building flow graphs:

Source:
- SourceNode: Entry point that converts Frames to FlowData

Path:
- PathNode: Wraps existing Path for extractor execution

Filter:
- FilterNode: Generic condition-based filtering
- ObservationFilter: Filters by observation count
- SignalThresholdFilter: Filters by signal threshold

Sampler:
- SamplerNode: Passes every Nth frame
- RateLimiterNode: Time-based rate limiting
- TimestampSamplerNode: Timestamp-based sampling

Branch:
- BranchNode: Two-way conditional routing
- FanOutNode: Replicates to multiple paths
- MultiBranchNode: Multi-way conditional routing
- ConditionalFanOutNode: Selective replication

Join:
- JoinNode: Merges data from multiple paths
- CascadeFusionNode: Secondary fusion on merged data
- CollectorNode: Collects data into batches
"""

from visualpath.flow.nodes.source import SourceNode
from visualpath.flow.nodes.path import PathNode
from visualpath.flow.nodes.filter import (
    FilterNode,
    ObservationFilter,
    SignalThresholdFilter,
)
from visualpath.flow.nodes.sampler import (
    SamplerNode,
    RateLimiterNode,
    TimestampSamplerNode,
)
from visualpath.flow.nodes.branch import (
    BranchNode,
    FanOutNode,
    MultiBranchNode,
    ConditionalFanOutNode,
)
from visualpath.flow.nodes.join import (
    JoinNode,
    CascadeFusionNode,
    CollectorNode,
)

__all__ = [
    # Source
    "SourceNode",
    # Path
    "PathNode",
    # Filter
    "FilterNode",
    "ObservationFilter",
    "SignalThresholdFilter",
    # Sampler
    "SamplerNode",
    "RateLimiterNode",
    "TimestampSamplerNode",
    # Branch
    "BranchNode",
    "FanOutNode",
    "MultiBranchNode",
    "ConditionalFanOutNode",
    # Join
    "JoinNode",
    "CascadeFusionNode",
    "CollectorNode",
]
