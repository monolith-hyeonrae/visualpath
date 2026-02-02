"""visualpath - Video analysis pipeline platform.

visualpath provides a plugin-based platform for building video analysis pipelines.
It defines core abstractions for extractors, fusion modules, and orchestration,
allowing independent plugins to be developed and combined.

Key Concepts:
- Extractor: Analyzes frames and produces Observations
- Fusion: Combines Observations to make trigger decisions
- Path: A group of related extractors with shared fusion logic
- IsolationLevel: Controls how extractors are isolated (inline, thread, process, venv)

Example:
    >>> from visualpath.core import BaseExtractor, Observation
    >>> from visualpath.core import BaseFusion, FusionResult
    >>> from visualpath.core import IsolationLevel, Path, PathOrchestrator
"""

try:
    from visualpath._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

# Core exports
from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion, FusionResult
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator

__all__ = [
    # Core
    "BaseExtractor",
    "Observation",
    "BaseFusion",
    "FusionResult",
    "IsolationLevel",
    "IsolationConfig",
    "Path",
    "PathConfig",
    "PathOrchestrator",
]
