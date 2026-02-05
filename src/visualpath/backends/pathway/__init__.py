"""Pathway streaming backend for visualpath.

This package provides a Pathway-based execution backend that enables:
- Event-time windows with watermarks
- Late arrival handling via allowed_lateness
- Built-in backpressure management
- Interval joins for multi-path synchronization

Example:
    >>> import visualpath as vp
    >>>
    >>> result = vp.process_video("video.mp4", modules=[face_detector], backend="pathway")

Requirements:
    Install with: pip install visualpath[pathway]
"""

from visualpath.backends.pathway.backend import PathwayBackend
from visualpath.backends.pathway.stats import PathwayStats

__all__ = ["PathwayBackend", "PathwayStats"]
