"""Backend protocols for visualpath.

This module defines Protocol classes for ML backends. Plugins implement
these protocols to provide swappable ML implementations.
"""

from visualpath.backends.protocols import (
    DetectionBackend,
    DetectionResult,
)

__all__ = [
    "DetectionBackend",
    "DetectionResult",
]
