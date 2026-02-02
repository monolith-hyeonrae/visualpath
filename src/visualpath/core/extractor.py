"""Base extractor interface for feature extraction.

Extractors are the fundamental building blocks of visualpath pipelines.
Each extractor analyzes frames and produces Observations containing
extracted features.

Example:
    >>> from visualpath.core import BaseExtractor, Observation
    >>> from visualbase import Frame
    >>>
    >>> class MyExtractor(BaseExtractor):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_extractor"
    ...
    ...     def extract(self, frame: Frame) -> Observation:
    ...         # Analyze the frame
    ...         features = self._analyze(frame.data)
    ...         return Observation(
    ...             source=self.name,
    ...             frame_id=frame.frame_id,
    ...             t_ns=frame.t_src_ns,
    ...             data=features,
    ...         )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TypeVar, Generic

from visualbase import Frame


T = TypeVar("T")


@dataclass
class Observation(Generic[T]):
    """Observation output from an extractor.

    Observations are timestamped feature extractions that flow from
    extractors to fusion modules.

    The generic type parameter T allows domain-specific data structures
    to be attached to observations.

    Attributes:
        source: Name of the extractor that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features (scalar values).
        data: Optional domain-specific data (e.g., list of detected objects).
        metadata: Additional metadata about the observation.
        timing: Optional per-component timing information in milliseconds.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, float] = field(default_factory=dict)
    data: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None


class BaseExtractor(ABC):
    """Abstract base class for feature extractors.

    Extractors analyze frames and produce observations containing
    extracted features. Multiple extractors can run in parallel,
    each focusing on different aspects of the video.

    Subclasses must implement:
    - name: Unique identifier for the extractor
    - extract: Process a frame and return an observation

    Optional overrides:
    - initialize: Load models or resources
    - cleanup: Release resources
    - recommended_isolation: Suggest isolation level for this extractor

    Example:
        >>> class ObjectDetector(BaseExtractor):
        ...     @property
        ...     def name(self) -> str:
        ...         return "object"
        ...
        ...     def extract(self, frame: Frame) -> Optional[Observation]:
        ...         objects = self._detect(frame.data)
        ...         return Observation(
        ...             source=self.name,
        ...             frame_id=frame.frame_id,
        ...             t_ns=frame.t_src_ns,
        ...             data=objects,
        ...             signals={"object_count": len(objects)},
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this extractor."""
        ...

    @abstractmethod
    def extract(self, frame: Frame) -> Optional[Observation]:
        """Extract features from a frame.

        Args:
            frame: Input frame to analyze.

        Returns:
            Observation containing extracted features, or None if
            no meaningful observation could be made.
        """
        ...

    @property
    def recommended_isolation(self) -> "IsolationLevel":
        """Recommended isolation level for this extractor.

        Override to suggest an appropriate isolation level based on
        the extractor's resource requirements. This can be overridden
        by configuration at runtime.

        Returns:
            Default IsolationLevel.INLINE.
        """
        from visualpath.core.isolation import IsolationLevel
        return IsolationLevel.INLINE

    def initialize(self) -> None:
        """Initialize extractor resources (models, etc.).

        Override this method to load models or initialize resources.
        Called once before processing begins.
        """
        pass

    def cleanup(self) -> None:
        """Clean up extractor resources.

        Override this method to release resources.
        Called when processing ends.
        """
        pass

    def __enter__(self) -> "BaseExtractor":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


class DummyExtractor(BaseExtractor):
    """Dummy extractor for testing.

    Always returns a simple observation with fixed signals.
    Useful for integration tests and subprocess verification.
    """

    def __init__(self, delay_ms: float = 0.0):
        """Initialize the dummy extractor.

        Args:
            delay_ms: Optional delay in milliseconds to simulate processing time.
        """
        self._delay_ms = delay_ms
        self._extract_count = 0

    @property
    def name(self) -> str:
        return "dummy"

    def extract(self, frame: Frame) -> Optional[Observation]:
        """Extract dummy observation from frame."""
        import time

        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)

        self._extract_count += 1

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "count": float(self._extract_count),
                "dummy": 1.0,
            },
            data={"status": "ok"},
            metadata={"extractor": "dummy"},
        )


# Import for type checking only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from visualpath.core.isolation import IsolationLevel
