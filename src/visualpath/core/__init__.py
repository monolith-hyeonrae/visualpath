"""Core abstractions for visualpath.

This module provides the fundamental building blocks for video analysis pipelines:

Primary interface:
- Module: Unified base class for all processing components
- Observation: Analysis results and trigger decisions

Other:
- IsolationLevel: Enum for plugin isolation levels
- Path: A group of modules with shared configuration
- PathOrchestrator: Orchestrates multiple Paths
"""

from visualpath.core.module import Module
from visualpath.core.extractor import Observation, DummyExtractor
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator

__all__ = [
    # Primary interface
    "Module",
    # Data types
    "Observation",
    # Testing
    "DummyExtractor",
    # Isolation
    "IsolationLevel",
    "IsolationConfig",
    # Path
    "Path",
    "PathConfig",
    "PathOrchestrator",
]
