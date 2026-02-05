"""Core abstractions for visualpath.

This module provides the fundamental building blocks for video analysis pipelines:

Primary interface:
- Module: Unified base class for all processing components

Module outputs:
- Observation: Analysis results (features, detections)
- FusionResult: Trigger decisions

Legacy interfaces (deprecated):
- BaseExtractor: Use Module instead
- BaseFusion: Use Module with FusionResult output instead

Other:
- IsolationLevel: Enum for plugin isolation levels
- Path: A group of modules with shared configuration
- PathOrchestrator: Orchestrates multiple Paths
"""

from visualpath.core.module import Module, FusionModule
from visualpath.core.extractor import BaseExtractor, Observation, DummyExtractor
from visualpath.core.fusion import BaseFusion, FusionResult
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator

__all__ = [
    # Primary interface
    "Module",
    "FusionModule",
    # Data types
    "Observation",
    "FusionResult",
    # Legacy (deprecated)
    "BaseExtractor",
    "DummyExtractor",
    "BaseFusion",
    # Isolation
    "IsolationLevel",
    "IsolationConfig",
    # Path
    "Path",
    "PathConfig",
    "PathOrchestrator",
]
