"""visualpath - Video analysis pipeline platform.

visualpath provides a plugin-based platform for building video analysis pipelines.

Quick Start:
    >>> import visualpath as vp
    >>>
    >>> # Process a video
    >>> result = vp.process_video("video.mp4", extractors=["face", "pose"])
    >>>
    >>> # Create a custom extractor (decorator)
    >>> @vp.extractor("brightness")
    >>> def check_brightness(frame):
    ...     return {"brightness": float(frame.data.mean())}
    >>>
    >>> # Create a custom fusion (decorator)
    >>> @vp.fusion(sources=["face"], cooldown=2.0)
    >>> def smile_detector(face):
    ...     if face.get("happy", 0) > 0.5:
    ...         return vp.trigger("smile", score=face["happy"])

For advanced usage, see:
- visualpath.core: BaseExtractor, BaseFusion, Observation, FusionResult
- visualpath.flow: FlowGraph, DAG-based pipeline
- visualpath.process: Distributed processing (IPC, workers)
"""

try:
    from visualpath._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

# =============================================================================
# High-level API (recommended)
# =============================================================================
from visualpath.api import (
    # Configuration
    DEFAULT_FPS,
    DEFAULT_COOLDOWN,
    DEFAULT_PRE_SEC,
    DEFAULT_POST_SEC,
    # Decorators
    extractor,
    fusion,
    trigger,
    # Registry
    get_extractor,
    get_fusion,
    list_extractors,
    list_fusions,
    # Types
    TriggerSpec,
)

# Pipeline runner (from runner.py)
from visualpath.runner import (
    process_video,
    run,
    ProcessResult,
)

# =============================================================================
# Core exports (for advanced use)
# =============================================================================
from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion, FusionResult
from visualpath.core.isolation import IsolationLevel, IsolationConfig
from visualpath.core.path import Path, PathConfig, PathOrchestrator

__all__ = [
    # Configuration
    "DEFAULT_FPS",
    "DEFAULT_COOLDOWN",
    "DEFAULT_PRE_SEC",
    "DEFAULT_POST_SEC",
    # High-level API
    "extractor",
    "fusion",
    "trigger",
    "process_video",
    "run",
    "get_extractor",
    "get_fusion",
    "list_extractors",
    "list_fusions",
    "ProcessResult",
    "TriggerSpec",
    # Core (advanced)
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
