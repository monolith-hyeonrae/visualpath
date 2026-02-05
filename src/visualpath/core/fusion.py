"""Base fusion interface for combining observations.

Fusion modules receive observations from multiple extractors,
align them by timestamp, and decide when to fire triggers.

Example:
    >>> from visualpath.core import BaseFusion, FusionResult, Observation
    >>> from visualbase import Trigger
    >>>
    >>> class SimpleFusion(BaseFusion):
    ...     def __init__(self, threshold: float = 0.5):
    ...         self._threshold = threshold
    ...         self._gate_open = False
    ...         self._cooldown_until = 0
    ...
    ...     def update(self, observation: Observation) -> FusionResult:
    ...         score = observation.signals.get("score", 0)
    ...         if score > self._threshold and self._gate_open:
    ...             return FusionResult(
    ...                 should_trigger=True,
    ...                 trigger=Trigger(...),
    ...                 score=score,
    ...                 reason="threshold_exceeded",
    ...             )
    ...         return FusionResult(should_trigger=False)
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from visualbase import Trigger

from visualpath.core.extractor import Observation


@dataclass
class FusionResult:
    """Result from fusion module decision.

    Attributes:
        should_trigger: Whether a highlight trigger should fire.
        trigger: The trigger to send if should_trigger is True.
        score: Confidence/quality score [0, 1].
        reason: Primary reason for the trigger (e.g., "expression_spike").
        observations_used: Number of observations used in this decision.
        metadata: Additional metadata about the decision.
    """

    should_trigger: bool
    trigger: Optional[Trigger] = None
    score: float = 0.0
    reason: str = ""
    observations_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFusion(ABC):
    """Abstract base class for fusion modules.

    .. deprecated::
        Use :class:`visualpath.core.Module` instead.
        BaseFusion will be removed in a future version.

    Fusion modules receive observations from multiple extractors,
    align them by timestamp, and decide when to fire triggers.

    The fusion module maintains state across frames to implement
    hysteresis, cooldown, and temporal smoothing.

    Subclasses must implement:
    - update: Process a new observation and decide on trigger
    - reset: Reset internal state
    - is_gate_open: Property indicating quality gate state
    - in_cooldown: Property indicating cooldown state

    Example:
        >>> fusion = MyFusion(cooldown_sec=2.0)
        >>> for obs in observations:
        ...     result = fusion.update(obs)
        ...     if result.should_trigger:
        ...         handle_trigger(result.trigger)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip warning for internal classes
        if cls.__module__.startswith("visualpath."):
            return
        warnings.warn(
            f"{cls.__name__}: BaseFusion is deprecated. "
            "Use visualpath.core.Module instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def update(self, observation: Observation) -> FusionResult:
        """Process a new observation and decide on trigger.

        Args:
            observation: New observation from an extractor.

        Returns:
            FusionResult indicating whether to trigger and with what parameters.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset fusion state.

        Call this when starting a new video or after a significant
        discontinuity in the stream.
        """
        ...

    @property
    @abstractmethod
    def is_gate_open(self) -> bool:
        """Whether the quality gate is currently open.

        The gate controls whether triggers can fire based on
        composition quality (face framing, angles, etc.).
        """
        ...

    @property
    @abstractmethod
    def in_cooldown(self) -> bool:
        """Whether the fusion is in cooldown after a recent trigger."""
        ...
