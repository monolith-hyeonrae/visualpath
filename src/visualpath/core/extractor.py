"""Observation dataclass and DummyExtractor for testing.

Observations are timestamped feature extractions that flow from
modules to downstream consumers.

Example:
    >>> from visualpath.core import Observation
    >>> from visualbase import Frame
    >>>
    >>> obs = Observation(
    ...     source="face_detect",
    ...     frame_id=frame.frame_id,
    ...     t_ns=frame.t_src_ns,
    ...     signals={"face_count": 3},
    ...     data={"faces": [...]},
    ... )
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypeVar, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame


T = TypeVar("T")


@dataclass
class Observation(Generic[T]):
    """Observation output from a module.

    Observations are timestamped feature extractions that flow from
    modules to downstream consumers.

    The generic type parameter T allows domain-specific data structures
    to be attached to observations.

    Attributes:
        source: Name of the module that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features (scalar values).
            For trigger observations, may include:
            - should_trigger: Whether a trigger should fire
            - trigger_score: Confidence score [0, 1]
            - trigger_reason: Primary reason for the trigger
        data: Optional domain-specific data (e.g., list of detected objects).
        metadata: Additional metadata about the observation.
            For trigger observations, may include:
            - trigger: Trigger object if should_trigger is True
        timing: Optional per-component timing information in milliseconds.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, Any] = field(default_factory=dict)
    data: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None

    # Trigger helper properties

    @property
    def should_trigger(self) -> bool:
        """Check if this observation indicates a trigger should fire.

        Returns:
            True if signals["should_trigger"] is truthy.
        """
        return bool(self.signals.get("should_trigger", False))

    @property
    def trigger_score(self) -> float:
        """Get the trigger confidence score.

        Returns:
            Score from signals["trigger_score"], or 0.0 if not set.
        """
        return float(self.signals.get("trigger_score", 0.0))

    @property
    def trigger_reason(self) -> str:
        """Get the trigger reason.

        Returns:
            Reason from signals["trigger_reason"], or empty string if not set.
        """
        return str(self.signals.get("trigger_reason", ""))

    @property
    def trigger(self) -> Optional[Any]:
        """Get the Trigger object if present.

        Returns:
            Trigger from metadata["trigger"], or None if not set.
        """
        return self.metadata.get("trigger")

    # Backwards-compatible aliases for FusionResult API

    @property
    def score(self) -> float:
        """Alias for trigger_score (backwards compatibility)."""
        return self.trigger_score

    @property
    def reason(self) -> str:
        """Alias for trigger_reason (backwards compatibility)."""
        return self.trigger_reason


class DummyExtractor:
    """Dummy extractor (Module) for testing.

    Always returns a simple observation with fixed signals.
    Useful for integration tests and subprocess verification.

    Implements the Module interface.
    """

    depends = []  # No dependencies

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

    def process(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, "Observation"]] = None,
    ) -> Optional[Observation]:
        """Process frame and return dummy observation."""
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

    # Alias for backward compatibility
    def extract(
        self,
        frame: "Frame",
        deps: Optional[Dict[str, "Observation"]] = None,
    ) -> Optional[Observation]:
        """Alias for process() for backward compatibility."""
        return self.process(frame, deps)

    def initialize(self) -> None:
        """Initialize resources (no-op for dummy)."""
        pass

    def cleanup(self) -> None:
        """Clean up resources (no-op for dummy)."""
        pass

    def reset(self) -> None:
        """Reset state."""
        self._extract_count = 0

    def __enter__(self) -> "DummyExtractor":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
