"""Unified Module interface for video analysis.

Module is the base class for all processing components in visualpath.
It unifies the previously separate Extractor and Fusion concepts into
a single, consistent interface.

A Module:
- Has a unique name
- Can declare dependencies on other modules
- Processes frames and produces Observations
- Manages its own lifecycle (initialize/cleanup)

Example - Analysis module:
    >>> class FaceDetector(Module):
    ...     @property
    ...     def name(self) -> str:
    ...         return "face_detect"
    ...
    ...     def process(self, frame, deps=None) -> Observation:
    ...         faces = self._detect(frame.data)
    ...         return Observation(
    ...             source=self.name,
    ...             frame_id=frame.frame_id,
    ...             t_ns=frame.t_src_ns,
    ...             signals={"face_count": len(faces)},
    ...             data={"faces": faces},
    ...         )

Example - Trigger module:
    >>> class SmileTrigger(Module):
    ...     depends = ["face_detect", "expression"]
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "smile_trigger"
    ...
    ...     def process(self, frame, deps=None) -> Observation:
    ...         expr_obs = deps.get("expression") if deps else None
    ...         if not expr_obs:
    ...             return Observation(
    ...                 source=self.name,
    ...                 frame_id=frame.frame_id,
    ...                 t_ns=frame.t_src_ns,
    ...                 signals={"should_trigger": False},
    ...             )
    ...
    ...         smile_score = expr_obs.signals.get("smile", 0)
    ...         if smile_score > 0.8:
    ...             return Observation(
    ...                 source=self.name,
    ...                 frame_id=frame.frame_id,
    ...                 t_ns=frame.t_src_ns,
    ...                 signals={
    ...                     "should_trigger": True,
    ...                     "trigger_score": smile_score,
    ...                     "trigger_reason": "smile_detected",
    ...                 },
    ...                 metadata={"trigger": Trigger.point(...)},
    ...             )
    ...         return Observation(
    ...             source=self.name,
    ...             frame_id=frame.frame_id,
    ...             t_ns=frame.t_src_ns,
    ...             signals={"should_trigger": False},
    ...         )

Dependency chain example:
    >>> # face_detect -> expression -> smile_trigger
    >>> modules = [FaceDetector(), ExpressionAnalyzer(), SmileTrigger()]
    >>> # Dependencies are resolved automatically by name
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.extractor import Observation


# Type alias for dependency context
DepsContext = Optional[Dict[str, "Observation"]]


class Module(ABC):
    """Base class for all processing modules.

    Modules are the building blocks of visualpath pipelines. Each module
    processes frames and produces Observations that can be consumed by
    other modules downstream.

    For trigger modules, set signals["should_trigger"] = True in the
    returned Observation. Helper properties on Observation make this easy:
    - obs.should_trigger -> signals["should_trigger"]
    - obs.trigger_score -> signals["trigger_score"]
    - obs.trigger_reason -> signals["trigger_reason"]
    - obs.trigger -> metadata["trigger"]

    Attributes:
        depends: List of module names this module depends on.
                 Dependencies are passed to process() as the deps dict.
    """

    # Class attribute: list of module names this module depends on
    depends: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this module.

        Used for:
        - Dependency resolution
        - Logging and debugging
        - Result attribution
        """
        ...

    @abstractmethod
    def process(
        self,
        frame: "Frame",
        deps: DepsContext = None,
    ) -> Optional["Observation"]:
        """Process a frame and produce output.

        Args:
            frame: The current video frame to process.
            deps: Dict of Observations from dependency modules.
                  Keys are module names from the `depends` list.

        Returns:
            Observation with analysis results.
            For trigger modules, set signals["should_trigger"] = True
            and optionally include metadata["trigger"].
            Return None if no meaningful output can be produced.

        Example:
            >>> def process(self, frame, deps=None):
            ...     # Access dependency output
            ...     face_obs = deps.get("face_detect") if deps else None
            ...     if not face_obs:
            ...         return None
            ...
            ...     # Process and return
            ...     result = self._analyze(frame, face_obs.data)
            ...     return Observation(...)
        """
        ...

    def initialize(self) -> None:
        """Initialize module resources.

        Called once before processing begins. Override to:
        - Load ML models
        - Initialize GPU contexts
        - Open file handles
        - Set up connections
        """
        pass

    def cleanup(self) -> None:
        """Release module resources.

        Called when processing ends. Override to:
        - Unload models
        - Release GPU memory
        - Close handles
        """
        pass

    def reset(self) -> None:
        """Reset module state.

        Called when starting a new video or after discontinuity.
        Override for stateful modules (triggers with cooldown, etc.).
        """
        pass

    def extract(
        self,
        frame: "Frame",
        deps: DepsContext = None,
    ) -> Optional["Observation"]:
        """Alias for process() for backward compatibility.

        This allows Module subclasses to work with code that calls extract().

        Args:
            frame: The current video frame to process.
            deps: Dict of Observations from dependency modules.

        Returns:
            Same as process().
        """
        return self.process(frame, deps)

    def update(
        self,
        observation: "Observation",
    ) -> Optional["Observation"]:
        """Legacy fusion API: process an observation and produce output.

        This method provides backwards compatibility with the old fusion API
        that used update(observation) instead of process(frame, deps).

        Creates a mock frame from the observation and delegates to process().

        Args:
            observation: Observation from an upstream module.

        Returns:
            Observation with trigger decision.
        """
        from dataclasses import dataclass

        @dataclass
        class _MockFrame:
            frame_id: int
            t_src_ns: int
            data: object = None

        mock_frame = _MockFrame(
            frame_id=observation.frame_id,
            t_src_ns=observation.t_ns,
            data=None,
        )
        # Pass observation in deps using source as key
        deps = {observation.source: observation}
        return self.process(mock_frame, deps)

    def __enter__(self) -> "Module":
        """Context manager entry - calls initialize()."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - calls cleanup()."""
        self.cleanup()


__all__ = ["Module", "DepsContext"]
