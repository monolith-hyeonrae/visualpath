"""Pipeline runner for visualpath.

This module provides the ``process_video()`` entry point for running video
analysis pipelines. It handles:
- Module resolution from registry
- Video source opening (visualbase or OpenCV fallback)
- Backend selection and FlowGraph construction
- Pipeline execution via ``backend.execute(frames, graph)``

Example:
    >>> import visualpath as vp
    >>>
    >>> result = vp.process_video("video.mp4", modules=[face_detector, smile_trigger])
    >>> print(f"Found {len(result.triggers)} triggers")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from visualbase import Frame, Trigger

from visualpath.core.extractor import Observation
from visualpath.core.module import Module

# Backend type alias
BackendType = Literal["simple", "pathway"]


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


def _resolve_modules(
    modules: Sequence[Union[str, Module]],
) -> List[Module]:
    """Resolve module names or instances to instances.

    Modules can be either extractor names (from registry) or Module instances.

    Args:
        modules: List of module names or instances.

    Returns:
        List of Module instances.

    Raises:
        ValueError: If a module name is not found.
    """
    from visualpath.api import get_extractor, get_fusion

    instances: List[Module] = []
    for mod in modules:
        if isinstance(mod, str):
            # Try extractor registry first, then fusion registry
            instance = get_extractor(mod)
            if instance is None:
                instance = get_fusion(mod)
            if instance is None:
                raise ValueError(f"Unknown module: {mod}")
            instances.append(instance)
        else:
            instances.append(mod)
    return instances


def _open_video_source(video: Union[str, Path], fps: int):
    """Open a video source, returning (frames_iterator, cleanup_fn).

    Tries visualbase first, falls back to OpenCV.

    Args:
        video: Path to video file.
        fps: Frames per second to process.

    Returns:
        Tuple of (frames_iterator, optional_cleanup_function).
    """
    vb = None
    try:
        from visualbase import VideoBase
        vb = VideoBase()
        source = vb.open(str(video))
        frames = source.stream(fps=fps)
        return frames, lambda: vb.disconnect()
    except Exception:
        if vb is not None:
            try:
                vb.disconnect()
            except Exception:
                pass
        # Fallback to OpenCV
        return _opencv_frames(str(video), fps), None


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


# Default values (imported from api for consistency)
from visualpath.api import DEFAULT_FPS


def process_video(
    video: Union[str, Path],
    modules: Sequence[Union[str, Module]],
    *,
    fps: int = DEFAULT_FPS,
    backend: BackendType = "simple",
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> ProcessResult:
    """Process a video with the specified modules.

    This is the main entry point for video processing. It constructs
    a FlowGraph from the modules, opens the video source, and runs
    the pipeline through the selected backend.

    Args:
        video: Path to video file.
        modules: List of module names or instances.
        fps: Frames per second to process.
        backend: Execution backend ("simple" or "pathway").
        on_trigger: Callback when a trigger fires.

    Returns:
        ProcessResult with triggers and statistics.

    Example:
        >>> result = vp.process_video("video.mp4", modules=[face_detector, smile_trigger])
    """
    from visualpath.flow.graph import FlowGraph

    module_instances = _resolve_modules(modules)
    graph = FlowGraph.from_modules(module_instances)

    # Build trigger callback
    def _trigger_callback(data) -> None:
        for result in data.results:
            # result is now an Observation with trigger info
            # Use the should_trigger property and trigger from metadata
            if result.should_trigger and result.trigger:
                if on_trigger:
                    on_trigger(result.trigger)

    if on_trigger:
        graph.on_trigger(_trigger_callback)

    # Open video source
    frames, cleanup_fn = _open_video_source(video, fps)

    # Execute pipeline
    execution_backend = _get_backend(backend)

    try:
        pipeline_result = execution_backend.execute(frames, graph)
    finally:
        if cleanup_fn:
            try:
                cleanup_fn()
            except Exception:
                pass

    return ProcessResult(
        triggers=pipeline_result.triggers,
        frame_count=pipeline_result.frame_count,
        duration_sec=pipeline_result.frame_count / fps if pipeline_result.frame_count > 0 else 0.0,
    )


# Alias for simpler API
run = process_video


__all__ = [
    "ProcessResult",
    "BackendType",
    "process_video",
    "run",
    "_get_backend",
]
