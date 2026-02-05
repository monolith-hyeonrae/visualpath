"""Unified Module interface for video analysis.

Module is the base class for all processing components in visualpath.
It unifies the previously separate Extractor and Fusion concepts into
a single, consistent interface.

A Module:
- Has a unique name
- Can declare dependencies on other modules
- Processes frames and produces outputs (Observation or FusionResult)
- Manages its own lifecycle (initialize/cleanup)

Example - Analysis module (produces Observation):
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

Example - Trigger module (produces FusionResult):
    >>> class SmileTrigger(Module):
    ...     depends = ["face_detect", "expression"]
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "smile_trigger"
    ...
    ...     def process(self, frame, deps=None) -> FusionResult:
    ...         expr_obs = deps.get("expression") if deps else None
    ...         if not expr_obs:
    ...             return FusionResult(should_trigger=False)
    ...
    ...         smile_score = expr_obs.signals.get("smile", 0)
    ...         if smile_score > 0.8:
    ...             return FusionResult(
    ...                 should_trigger=True,
    ...                 trigger=Trigger.point(...),
    ...                 score=smile_score,
    ...             )
    ...         return FusionResult(should_trigger=False)

Dependency chain example:
    >>> # face_detect → expression → smile_trigger
    >>> modules = [FaceDetector(), ExpressionAnalyzer(), SmileTrigger()]
    >>> # Dependencies are resolved automatically by name
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.extractor import Observation
    from visualpath.core.fusion import FusionResult


# Type alias for module outputs
ModuleOutput = Union["Observation", "FusionResult", None]

# Type alias for dependency context
DepsContext = Optional[Dict[str, Union["Observation", "FusionResult"]]]


class Module(ABC):
    """Base class for all processing modules.

    Modules are the building blocks of visualpath pipelines. Each module
    processes frames and produces outputs that can be consumed by other
    modules downstream.

    Output types:
    - Observation: Analysis results (features, detections, etc.)
    - FusionResult: Trigger decisions (should_trigger=True/False)

    The output type determines the module's role:
    - Modules producing Observation are "analyzers"
    - Modules producing FusionResult are "triggers"

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
    ) -> ModuleOutput:
        """Process a frame and produce output.

        Args:
            frame: The current video frame to process.
            deps: Dict of outputs from dependency modules.
                  Keys are module names from the `depends` list.
                  Values are Observation or FusionResult from those modules.

        Returns:
            - Observation: For analysis modules
            - FusionResult: For trigger modules
            - None: If no meaningful output can be produced

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

    def __enter__(self) -> "Module":
        """Context manager entry - calls initialize()."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - calls cleanup()."""
        self.cleanup()

    # Convenience properties for checking module type

    @property
    def is_trigger(self) -> bool:
        """Check if this module produces triggers.

        Default implementation returns False. Trigger modules
        should override to return True, or this can be detected
        at runtime by checking the output type.
        """
        return False


# Backward compatibility aliases
# These will be deprecated in future versions

def _create_extractor_adapter():
    """Create BaseExtractor as an alias for Module."""
    import warnings

    class BaseExtractor(Module):
        """DEPRECATED: Use Module instead.

        BaseExtractor is now an alias for Module.
        Modules that return Observation are extractors.
        """

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            warnings.warn(
                "BaseExtractor is deprecated. Use Module instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Alias extract -> process for backward compatibility
        def process(self, frame, deps=None):
            return self.extract(frame, deps)

        @abstractmethod
        def extract(self, frame, deps=None):
            """DEPRECATED: Override process() instead."""
            ...

    return BaseExtractor


def _create_fusion_adapter():
    """Create adapter for BaseFusion to work as Module."""
    import warnings

    class FusionModule(Module):
        """Adapter to use BaseFusion as a Module.

        Wraps a BaseFusion instance to provide Module interface.
        """

        def __init__(self, fusion: "BaseFusion", name: str = "fusion"):
            self._fusion = fusion
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        @property
        def is_trigger(self) -> bool:
            return True

        def process(self, frame, deps=None):
            # FusionModule needs observations from deps
            # Typically depends on all upstream analyzers
            if not deps:
                from visualpath.core.fusion import FusionResult
                return FusionResult(should_trigger=False)

            # Process each observation through fusion
            results = []
            for obs in deps.values():
                if hasattr(obs, 'source'):  # It's an Observation
                    result = self._fusion.update(obs)
                    if result.should_trigger:
                        return result
                    results.append(result)

            # Return last result or non-trigger
            if results:
                return results[-1]
            from visualpath.core.fusion import FusionResult
            return FusionResult(should_trigger=False)

        def reset(self) -> None:
            self._fusion.reset()

        @property
        def is_gate_open(self) -> bool:
            return self._fusion.is_gate_open

        @property
        def in_cooldown(self) -> bool:
            return self._fusion.in_cooldown

    return FusionModule


# Export the adapter classes
FusionModule = _create_fusion_adapter()


__all__ = ["Module", "ModuleOutput", "DepsContext", "FusionModule"]
