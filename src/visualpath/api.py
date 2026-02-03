"""High-level API for visualpath.

This module provides a simple, declarative API for building video analysis
pipelines. It's designed for ease of use while still allowing advanced
customization when needed.

Quick Start:
    >>> import visualpath as vp
    >>>
    >>> # Process a video with built-in extractors
    >>> clips = vp.process("video.mp4", extractors=["face", "pose"])
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
    >>> # Run with custom components
    >>> vp.run("video.mp4", extractors=["brightness"], fusion=smile_detector)
"""

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    Sequence,
    Iterator,
)

from visualbase import Frame, Trigger

from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion, FusionResult

# Backend type alias
BackendType = Literal["simple", "pathway"]


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


def get_extractor(name: str) -> Optional[BaseExtractor]:
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


def get_fusion(name: str) -> Optional[BaseFusion]:
    """Get a registered fusion by name."""
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


class FunctionExtractor(BaseExtractor):
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

    def extract(self, frame: Frame) -> Optional[Observation]:
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

        >>> # With init/cleanup
        >>> model = None
        >>>
        >>> def load_model():
        ...     global model
        ...     model = load_my_model()
        >>>
        >>> @vp.extractor("detector", init=load_model)
        >>> def detect(frame):
        ...     return {"count": model.detect(frame.data)}
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


class FunctionFusion(BaseFusion):
    """Fusion wrapper for simple functions.

    Wraps a function that takes observation dicts and returns a TriggerSpec.
    Handles cooldown, gate logic, and state management automatically.
    """

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

    def update(self, observation: Observation) -> FusionResult:
        self._last_t_ns = observation.t_ns

        # Store latest observation
        obs_dict = dict(observation.signals)
        if observation.data:
            if isinstance(observation.data, dict):
                obs_dict.update(observation.data)
            else:
                obs_dict["data"] = observation.data

        self._latest[observation.source] = obs_dict

        # Check cooldown
        if self.in_cooldown:
            return FusionResult(should_trigger=False)

        # Check if we have all required sources
        if not all(src in self._latest for src in self._sources):
            return FusionResult(should_trigger=False)

        # Call the fusion function
        args = [self._latest[src] for src in self._sources]

        try:
            result = self._fn(*args)
        except Exception:
            return FusionResult(should_trigger=False)

        if result is None:
            return FusionResult(should_trigger=False)

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

        return FusionResult(
            should_trigger=True,
            trigger=trig,
            score=result.score,
            reason=result.reason,
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
# Pipeline Runner
# =============================================================================

@dataclass
class ProcessResult:
    """Result from processing a video."""
    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0


def _get_backend(backend: BackendType) -> "ExecutionBackend":
    """Get execution backend by name.

    Args:
        backend: Backend name ("simple" or "pathway").

    Returns:
        ExecutionBackend instance.

    Raises:
        ValueError: If backend is unknown.
        ImportError: If Pathway is requested but not installed.
    """
    from visualpath.backends.base import ExecutionBackend

    if backend == "simple":
        from visualpath.backends.simple import SimpleBackend
        return SimpleBackend()
    elif backend == "pathway":
        try:
            from visualpath.backends.pathway import PathwayBackend
            return PathwayBackend()
        except ImportError as e:
            raise ImportError(
                "Pathway backend requires pathway package. "
                "Install with: pip install visualpath[pathway]"
            ) from e
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'simple' or 'pathway'.")


def process(
    video: Union[str, Path],
    extractors: Sequence[Union[str, BaseExtractor]],
    fusion: Optional[Union[str, BaseFusion]] = None,
    *,
    fps: int = DEFAULT_FPS,
    backend: BackendType = "simple",
    on_trigger: Optional[Callable[[Trigger], None]] = None,
    on_frame: Optional[Callable[[Frame, List[Observation]], None]] = None,
) -> ProcessResult:
    """Process a video with the specified extractors and fusion.

    This is the main entry point for simple video processing.

    Args:
        video: Path to video file.
        extractors: List of extractor names or instances.
        fusion: Fusion name, instance, or None for default.
        fps: Frames per second to process.
        backend: Execution backend ("simple" or "pathway").
        on_trigger: Callback when a trigger fires.
        on_frame: Callback for each processed frame.

    Returns:
        ProcessResult with triggers and statistics.

    Example:
        >>> # Simple usage
        >>> result = vp.process("video.mp4", extractors=["face"])
        >>> print(f"Found {len(result.triggers)} highlights")

        >>> # With callback
        >>> def on_trig(t):
        ...     print(f"Trigger: {t.reason} at {t.t_start_ns/1e9:.1f}s")
        >>> vp.process("video.mp4", ["face"], on_trigger=on_trig)

        >>> # Using Pathway backend
        >>> result = vp.process("video.mp4", ["face"], backend="pathway")
    """
    # Resolve extractors
    ext_instances: List[BaseExtractor] = []
    for ext in extractors:
        if isinstance(ext, str):
            instance = get_extractor(ext)
            if instance is None:
                raise ValueError(f"Unknown extractor: {ext}")
            ext_instances.append(instance)
        else:
            ext_instances.append(ext)

    # Resolve fusion
    fusion_instance: Optional[BaseFusion] = None
    if fusion is not None:
        if isinstance(fusion, str):
            fusion_instance = get_fusion(fusion)
            if fusion_instance is None:
                raise ValueError(f"Unknown fusion: {fusion}")
        else:
            fusion_instance = fusion

    # Create video source
    vb = None
    try:
        from visualbase import VideoBase
        vb = VideoBase()
        source = vb.open(str(video))
        frames = source.stream(fps=fps)
    except Exception:
        # Fallback to OpenCV
        frames = _opencv_frames(str(video), fps)

    # Process using backend
    result = ProcessResult()

    # Get the execution backend
    execution_backend = _get_backend(backend)

    # For on_frame callback, we need to wrap with frame counting
    frame_count = [0]

    def wrapped_on_trigger(trigger: Trigger) -> None:
        result.triggers.append(trigger)
        if on_trigger:
            on_trigger(trigger)

    try:
        if backend == "simple" and on_frame:
            # Simple backend with on_frame callback - use inline processing
            for ext in ext_instances:
                ext.initialize()

            try:
                for frame in frames:
                    frame_count[0] += 1

                    # Run extractors
                    observations: List[Observation] = []
                    for ext in ext_instances:
                        obs = ext.extract(frame)
                        if obs is not None:
                            observations.append(obs)

                    # Callback
                    on_frame(frame, observations)

                    # Run fusion
                    if fusion_instance and observations:
                        for obs in observations:
                            fusion_result = fusion_instance.update(obs)
                            if fusion_result.should_trigger and fusion_result.trigger:
                                result.triggers.append(fusion_result.trigger)
                                if on_trigger:
                                    on_trigger(fusion_result.trigger)
            finally:
                for ext in ext_instances:
                    ext.cleanup()
        else:
            # Use backend.run() for standard processing
            # Note: Pathway backend handles initialization/cleanup internally
            if backend == "pathway":
                # For pathway, convert frames to list to count
                frame_list = list(frames)
                frame_count[0] = len(frame_list)
                triggers = execution_backend.run(
                    iter(frame_list),
                    ext_instances,
                    fusion_instance,
                    on_trigger=wrapped_on_trigger,
                )
            else:
                # Simple backend - we need to count frames
                frame_list = list(frames)
                frame_count[0] = len(frame_list)
                triggers = execution_backend.run(
                    iter(frame_list),
                    ext_instances,
                    fusion_instance,
                    on_trigger=wrapped_on_trigger,
                )
            result.triggers = triggers

        # Calculate duration
        result.frame_count = frame_count[0]
        if result.frame_count > 0:
            result.duration_sec = result.frame_count / fps

    finally:
        if vb is not None:
            try:
                vb.disconnect()
            except Exception:
                pass

    return result


def run(
    video: Union[str, Path],
    extractors: Sequence[Union[str, BaseExtractor]],
    fusion: Optional[Union[str, BaseFusion]] = None,
    *,
    fps: int = DEFAULT_FPS,
    backend: BackendType = "simple",
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> List[Trigger]:
    """Run a video analysis pipeline (simplified version of process).

    Args:
        video: Path to video file.
        extractors: List of extractor names or instances.
        fusion: Fusion name, instance, or None.
        fps: Frames per second to process.
        backend: Execution backend ("simple" or "pathway").
        on_trigger: Callback when a trigger fires.

    Returns:
        List of triggers found.

    Example:
        >>> triggers = vp.run("video.mp4", ["face", "pose"])
        >>> for t in triggers:
        ...     print(f"{t.reason}: {t.score:.2f}")

        >>> # Using Pathway backend
        >>> triggers = vp.run("video.mp4", ["face"], backend="pathway")
    """
    result = process(
        video, extractors, fusion,
        fps=fps, backend=backend, on_trigger=on_trigger,
    )
    return result.triggers


def _opencv_frames(video_path: str, fps: int) -> Iterator[Frame]:
    """Fallback frame iterator using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(src_fps / fps))
        frame_id = 0
        read_count = 0

        while True:
            ret, data = cap.read()
            if not ret:
                break

            read_count += 1
            if read_count % skip != 0:
                continue

            t_ns = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)

            yield Frame.from_array(
                data,
                frame_id=frame_id,
                t_src_ns=t_ns,
            )
            frame_id += 1

    finally:
        cap.release()


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
    "ProcessResult",
    # Registry
    "get_extractor",
    "get_fusion",
    "list_extractors",
    "list_fusions",
    # Pipeline
    "process",
    "run",
]
