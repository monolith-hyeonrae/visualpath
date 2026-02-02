"""Core abstractions for visualpath.

This module provides the fundamental building blocks for video analysis pipelines:
- BaseExtractor: Abstract base class for feature extractors
- Observation: Generic container for extraction results
- BaseFusion: Abstract base class for fusion modules
- FusionResult: Container for fusion decisions
- IsolationLevel: Enum for plugin isolation levels
- Path: A group of extractors with shared fusion logic
- PathOrchestrator: Orchestrates multiple Paths
"""

from visualpath.core.extractor import BaseExtractor, Observation, DummyExtractor
from visualpath.core.fusion import BaseFusion, FusionResult
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator

__all__ = [
    # Extractor
    "BaseExtractor",
    "Observation",
    "DummyExtractor",
    # Fusion
    "BaseFusion",
    "FusionResult",
    # Isolation
    "IsolationLevel",
    "IsolationConfig",
    # Path
    "Path",
    "PathConfig",
    "PathOrchestrator",
]
