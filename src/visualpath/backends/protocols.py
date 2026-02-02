"""Backend protocol definitions for ML backends.

Protocols define interfaces that ML backends should implement.
This allows extractors to work with different backend implementations
without tight coupling.

Example:
    >>> from visualpath.backends import DetectionBackend, DetectionResult
    >>> import numpy as np
    >>>
    >>> class MyDetector:
    ...     def initialize(self, device: str = "cuda:0") -> None:
    ...         self._model = load_model(device)
    ...
    ...     def detect(self, image: np.ndarray) -> list[DetectionResult]:
    ...         results = self._model(image)
    ...         return [DetectionResult(...) for r in results]
    ...
    ...     def cleanup(self) -> None:
    ...         del self._model
"""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """Generic detection result from a detection backend.

    This is a base dataclass for detection results. Domain-specific
    backends (face, pose, etc.) may define more specialized result types.

    Attributes:
        bbox: Bounding box (x, y, width, height) in pixels.
        confidence: Detection confidence [0, 1].
        class_id: Optional class/category identifier.
        class_name: Optional class/category name.
        metadata: Additional detection-specific data.
    """

    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels
    confidence: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DetectionBackend(Protocol):
    """Protocol for generic detection backends.

    Implementations detect objects/regions in images and return
    bounding boxes with confidence scores.

    Example implementations:
    - YOLO for object detection
    - Face detectors (SCRFD, RetinaFace)
    - Text detectors (EAST, CRAFT)
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models.

        Args:
            device: Device to use (e.g., "cuda:0", "cpu").
        """
        ...

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects in an image.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detection results with bounding boxes.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...
