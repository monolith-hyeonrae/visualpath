"""Pipeline runner for visualpath.

This module provides the ``process()`` and ``run()`` entry points for
running video analysis pipelines. It handles:
- Extractor/fusion resolution from registry
- Video source opening (visualbase or OpenCV fallback)
- Backend selection and FlowGraph construction
- Pipeline execution via ``backend.execute(frames, graph)``

Example:
    >>> import visualpath as vp
    >>>
    >>> result = vp.process_video("video.mp4", extractors=["face"])
    >>> triggers = vp.run("video.mp4", ["face"], backend="pathway")
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

from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion

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


def _resolve_extractors(
    extractors: Sequence[Union[str, BaseExtractor]],
) -> List[BaseExtractor]:
    """Resolve extractor names or instances to instances.

    Args:
        extractors: List of extractor names or instances.

    Returns:
        List of BaseExtractor instances.

    Raises:
        ValueError: If an extractor name is not found.
    """
    from visualpath.api import get_extractor

    instances: List[BaseExtractor] = []
    for ext in extractors:
        if isinstance(ext, str):
            instance = get_extractor(ext)
            if instance is None:
                raise ValueError(f"Unknown extractor: {ext}")
            instances.append(instance)
        else:
            instances.append(ext)
    return instances


def _resolve_fusion(
    fusion: Optional[Union[str, BaseFusion]],
) -> Optional[BaseFusion]:
    """Resolve a fusion name or instance to an instance.

    Args:
        fusion: Fusion name, instance, or None.

    Returns:
        BaseFusion instance or None.

    Raises:
        ValueError: If a fusion name is not found.
    """
    if fusion is None:
        return None

    if isinstance(fusion, str):
        from visualpath.api import get_fusion
        instance = get_fusion(fusion)
        if instance is None:
            raise ValueError(f"Unknown fusion: {fusion}")
        return instance

    return fusion


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
    extractors: Sequence[Union[str, BaseExtractor]],
    fusion: Optional[Union[str, BaseFusion]] = None,
    *,
    fps: int = DEFAULT_FPS,
    backend: BackendType = "simple",
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> ProcessResult:
    """Process a video with the specified extractors and fusion.

    This is the main entry point for video processing. It constructs
    a FlowGraph from the extractors/fusion, opens the video source,
    and runs the pipeline through the selected backend.

    Args:
        video: Path to video file.
        extractors: List of extractor names or instances.
        fusion: Fusion name, instance, or None.
        fps: Frames per second to process.
        backend: Execution backend ("simple" or "pathway").
        on_trigger: Callback when a trigger fires.

    Returns:
        ProcessResult with triggers and statistics.

    Example:
        >>> result = vp.process_video("video.mp4", extractors=["face"])
        >>> print(f"Found {len(result.triggers)} highlights")
    """
    from visualpath.flow.graph import FlowGraph

    ext_instances = _resolve_extractors(extractors)
    fusion_instance = _resolve_fusion(fusion)

    # Build FlowGraph
    def _trigger_callback(data) -> None:
        for result in data.results:
            if result.should_trigger and result.trigger:
                if on_trigger:
                    on_trigger(result.trigger)

    graph = FlowGraph.from_pipeline(ext_instances, fusion_instance)
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
    """
    result = process_video(
        video, extractors, fusion,
        fps=fps, backend=backend, on_trigger=on_trigger,
    )
    return result.triggers


__all__ = [
    "ProcessResult",
    "BackendType",
    "process_video",
    "run",
    "_get_backend",
]
