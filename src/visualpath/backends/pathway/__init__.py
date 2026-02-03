"""Pathway streaming backend for visualpath.

This package provides a Pathway-based execution backend that enables:
- Event-time windows with watermarks
- Late arrival handling via allowed_lateness
- Built-in backpressure management
- Interval joins for multi-path synchronization

Example:
    >>> from visualpath.backends.pathway import PathwayBackend
    >>>
    >>> backend = PathwayBackend()
    >>> triggers = backend.run(frames, extractors=[face_ext])

Requirements:
    Install with: pip install visualpath[pathway]
"""

from visualpath.backends.pathway.backend import PathwayBackend
from visualpath.backends.pathway.stats import PathwayStats

__all__ = ["PathwayBackend", "PathwayStats"]
