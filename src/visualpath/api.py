"""High-level API for visualpath.

This module provides a simple, declarative API for building video analysis
pipelines. It's designed for ease of use while still allowing advanced
customization when needed.

Quick Start:
    >>> import visualpath as vp
    >>>
    >>> # Define a custom extractor
    >>> @vp.extractor("brightness")
    >>> def check_brightness(frame):
    ...     gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
    ...     return {"brightness": float(gray.mean())}
    >>>
    >>> # Define a custom fusion
    >>> @vp.fusion(sources=["face"], cooldown=2.0)
    >>> def smile_detector(face):
    ...     if face.get("happy", 0) > 0.5:
    ...         return vp.trigger("smile", score=face["happy"])
    >>>
    >>> # Run with modules
    >>> result = vp.process_video("video.mp4", modules=[check_brightness, smile_detector])
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from visualbase import Frame, Trigger

from visualpath.core.extractor import Observation
from visualpath.core.module import Module


# =============================================================================
# Configuration
# =============================================================================

# Default values
DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0
DEFAULT_PRE_SEC = 2.0
DEFAULT_POST_SEC = 2.0


# =============================================================================
# Registry
# =============================================================================

_extractor_registry: Dict[str, "FunctionExtractor"] = {}
_fusion_registry: Dict[str, "FunctionFusion"] = {}


def get_extractor(name: str) -> Optional[Module]:
    """Get a registered extractor by name.

    First checks function-based extractors, then plugin registry.
    """
    if name in _extractor_registry:
        return _extractor_registry[name]

    # Fallback to plugin discovery
    try:
        from visualpath.plugin import discover_extractors
        extractors = discover_extractors()
        if name in extractors:
            return extractors[name]()
    except ImportError:
        pass

    return None


def get_fusion(name: str) -> Optional[Module]:
    """Get a registered fusion by name.

    Returns a Module instance (FunctionFusion or plugin-discovered fusion).
    """
    if name in _fusion_registry:
        return _fusion_registry[name]

    try:
        from visualpath.plugin import discover_fusions
        fusions = discover_fusions()
        if name in fusions:
            return fusions[name]()
    except ImportError:
        pass

    return None


def list_extractors() -> List[str]:
    """List all available extractor names."""
    names = list(_extractor_registry.keys())

    try:
        from visualpath.plugin import discover_extractors
        names.extend(discover_extractors().keys())
    except ImportError:
        pass

    return sorted(set(names))


def list_fusions() -> List[str]:
    """List all available fusion names."""
    names = list(_fusion_registry.keys())

    try:
        from visualpath.plugin import discover_fusions
        names.extend(discover_fusions().keys())
    except ImportError:
        pass

    return sorted(set(names))


# =============================================================================
# Function-based Extractor
# =============================================================================

ExtractFn = Callable[[Frame], Optional[Dict[str, Any]]]


class FunctionExtractor(Module):
    """Extractor wrapper for simple functions.

    Wraps a function that takes a Frame and returns a dict of signals.

    Example:
        >>> @vp.extractor("brightness")
        >>> def check_brightness(frame):
        ...     return {"brightness": float(frame.data.mean())}
    """

    def __init__(
        self,
        name: str,
        fn: ExtractFn,
        *,
        init_fn: Optional[Callable[[], None]] = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
    ):
        self._name = name
        self._fn = fn
        self._init_fn = init_fn
        self._cleanup_fn = cleanup_fn

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame: Frame, deps=None) -> Optional[Observation]:
        result = self._fn(frame)
        if result is None:
            return None

        # Handle both dict and Observation returns
        if isinstance(result, Observation):
            return result

        # Convert dict to Observation
        signals = {}
        data = None

        for key, value in result.items():
            if isinstance(value, (int, float)):
                signals[key] = float(value)
            else:
                # Non-scalar goes to data
                if data is None:
                    data = {}
                data[key] = value

        return Observation(
            source=self._name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals=signals,
            data=data,
        )

    def initialize(self) -> None:
        if self._init_fn:
            self._init_fn()

    def cleanup(self) -> None:
        if self._cleanup_fn:
            self._cleanup_fn()


def extractor(
    name: str,
    *,
    init: Optional[Callable[[], None]] = None,
    cleanup: Optional[Callable[[], None]] = None,
) -> Callable[[ExtractFn], FunctionExtractor]:
    """Decorator to create an extractor from a function.

    The decorated function should take a Frame and return a dict of signals,
    or None to skip the frame.

    Args:
        name: Unique name for this extractor.
        init: Optional initialization function.
        cleanup: Optional cleanup function.

    Returns:
        Decorator that creates a FunctionExtractor.

    Example:
        >>> @vp.extractor("quality")
        >>> def check_quality(frame):
        ...     blur = cv2.Laplacian(frame.data, cv2.CV_64F).var()
        ...     return {"blur_score": blur, "is_sharp": blur > 100}
    """
    def decorator(fn: ExtractFn) -> FunctionExtractor:
        ext = FunctionExtractor(name, fn, init_fn=init, cleanup_fn=cleanup)
        _extractor_registry[name] = ext
        return ext

    return decorator


# =============================================================================
# Function-based Fusion
# =============================================================================

FusionFn = Callable[..., Optional["TriggerSpec"]]


@dataclass
class TriggerSpec:
    """Specification for creating a trigger.

    Use the `trigger()` function to create this.
    """
    reason: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def trigger(reason: str, score: float = 1.0, **metadata) -> TriggerSpec:
    """Create a trigger specification.

    Use this in fusion functions to indicate a trigger should fire.

    Args:
        reason: Why the trigger fired (e.g., "smile", "wave").
        score: Confidence score [0, 1].
        **metadata: Additional metadata.

    Returns:
        TriggerSpec that the fusion framework converts to a real Trigger.

    Example:
        >>> @vp.fusion(sources=["face"])
        >>> def smile_detector(face):
        ...     if face.get("happy", 0) > 0.5:
        ...         return vp.trigger("smile", score=face["happy"])
    """
    return TriggerSpec(reason=reason, score=score, metadata=metadata)


class FunctionFusion(Module):
    """Fusion wrapper for simple functions.

    Wraps a function that takes observation dicts and returns a TriggerSpec.
    Handles cooldown, gate logic, and state management automatically.

    Returns Observation with trigger info in signals/metadata.
    """

    depends: List[str] = []  # Will be set from sources

    def __init__(
        self,
        name: str,
        fn: FusionFn,
        sources: List[str],
        *,
        cooldown: float = DEFAULT_COOLDOWN,
        gate_sources: Optional[List[str]] = None,
    ):
        self._name = name
        self._fn = fn
        self._sources = sources
        self.depends = sources  # Set depends from sources
        self._cooldown_sec = cooldown
        self._gate_sources = gate_sources or []

        # State
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._gate_open = True
        self._cooldown_until_ns = 0
        self._last_t_ns = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trigger(self) -> bool:
        return True

    def process(self, frame: Frame, deps: Optional[Dict[str, Observation]] = None) -> Observation:
        """Process frame with dependencies and decide on trigger."""
        t_ns = getattr(frame, "t_src_ns", 0)
        frame_id = getattr(frame, "frame_id", 0)
        self._last_t_ns = t_ns

        # Build no-trigger result helper
        def no_trigger(state: str = "no_trigger") -> Observation:
            return Observation(
                source=self._name,
                frame_id=frame_id,
                t_ns=t_ns,
                signals={
                    "should_trigger": False,
                    "trigger_score": 0.0,
                    "trigger_reason": "",
                },
                metadata={"state": state},
            )

        # Update latest from deps
        if deps:
            for src, obs in deps.items():
                obs_dict = dict(obs.signals)
                if obs.data:
                    if isinstance(obs.data, dict):
                        obs_dict.update(obs.data)
                    else:
                        obs_dict["data"] = obs.data
                self._latest[src] = obs_dict

        # Check cooldown
        if self.in_cooldown:
            return no_trigger("cooldown")

        # Check if we have all required sources
        if not all(src in self._latest for src in self._sources):
            return no_trigger("missing_sources")

        # Call the fusion function
        args = [self._latest[src] for src in self._sources]

        try:
            result = self._fn(*args)
        except Exception:
            return no_trigger("error")

        if result is None:
            return no_trigger()

        # Create trigger
        self._cooldown_until_ns = t_ns + int(self._cooldown_sec * 1e9)

        trig = Trigger.point(
            event_time_ns=t_ns,
            pre_sec=DEFAULT_PRE_SEC,
            post_sec=DEFAULT_POST_SEC,
            label=result.reason,
            score=result.score,
            metadata=result.metadata,
        )

        return Observation(
            source=self._name,
            frame_id=frame_id,
            t_ns=t_ns,
            signals={
                "should_trigger": True,
                "trigger_score": result.score,
                "trigger_reason": result.reason,
            },
            metadata={"trigger": trig},
        )

    def update(self, observation: Observation) -> Observation:
        """Legacy API: update with observation (for legacy fusion compatibility)."""
        self._last_t_ns = observation.t_ns

        # Store latest observation
        obs_dict = dict(observation.signals)
        if observation.data:
            if isinstance(observation.data, dict):
                obs_dict.update(observation.data)
            else:
                obs_dict["data"] = observation.data

        self._latest[observation.source] = obs_dict

        # Build no-trigger result helper
        def no_trigger(state: str = "no_trigger") -> Observation:
            return Observation(
                source=self._name,
                frame_id=observation.frame_id,
                t_ns=observation.t_ns,
                signals={
                    "should_trigger": False,
                    "trigger_score": 0.0,
                    "trigger_reason": "",
                },
                metadata={"state": state},
            )

        # Check cooldown
        if self.in_cooldown:
            return no_trigger("cooldown")

        # Check if we have all required sources
        if not all(src in self._latest for src in self._sources):
            return no_trigger("missing_sources")

        # Call the fusion function
        args = [self._latest[src] for src in self._sources]

        try:
            result = self._fn(*args)
        except Exception:
            return no_trigger("error")

        if result is None:
            return no_trigger()

        # Create trigger
        self._cooldown_until_ns = observation.t_ns + int(self._cooldown_sec * 1e9)

        trig = Trigger.point(
            event_time_ns=observation.t_ns,
            pre_sec=DEFAULT_PRE_SEC,
            post_sec=DEFAULT_POST_SEC,
            label=result.reason,
            score=result.score,
            metadata=result.metadata,
        )

        return Observation(
            source=self._name,
            frame_id=observation.frame_id,
            t_ns=observation.t_ns,
            signals={
                "should_trigger": True,
                "trigger_score": result.score,
                "trigger_reason": result.reason,
            },
            metadata={"trigger": trig},
        )

    def reset(self) -> None:
        self._latest.clear()
        self._gate_open = True
        self._cooldown_until_ns = 0
        self._last_t_ns = 0

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        return self._last_t_ns < self._cooldown_until_ns


def fusion(
    sources: List[str],
    *,
    name: Optional[str] = None,
    cooldown: float = DEFAULT_COOLDOWN,
    gate_sources: Optional[List[str]] = None,
) -> Callable[[FusionFn], FunctionFusion]:
    """Decorator to create a fusion from a function.

    The decorated function receives dicts of signals from each source
    and should return a TriggerSpec (via `vp.trigger()`) or None.

    Args:
        sources: List of extractor names this fusion combines.
        name: Optional name for this fusion (defaults to function name).
        cooldown: Seconds to wait between triggers.
        gate_sources: Sources that must pass quality gate.

    Returns:
        Decorator that creates a FunctionFusion.

    Example:
        >>> @vp.fusion(sources=["face", "pose"], cooldown=3.0)
        >>> def interaction_detector(face, pose):
        ...     if face.get("happy", 0) > 0.5 and pose.get("hand_wave"):
        ...         return vp.trigger("greeting", score=0.9)
    """
    def decorator(fn: FusionFn) -> FunctionFusion:
        fusion_name = name or fn.__name__
        fus = FunctionFusion(
            fusion_name, fn, sources,
            cooldown=cooldown,
            gate_sources=gate_sources,
        )
        _fusion_registry[fusion_name] = fus
        return fus

    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "DEFAULT_FPS",
    "DEFAULT_COOLDOWN",
    "DEFAULT_PRE_SEC",
    "DEFAULT_POST_SEC",
    # Decorators
    "extractor",
    "fusion",
    "trigger",
    # Classes (for advanced use)
    "FunctionExtractor",
    "FunctionFusion",
    "TriggerSpec",
    # Registry
    "get_extractor",
    "get_fusion",
    "list_extractors",
    "list_fusions",
]
