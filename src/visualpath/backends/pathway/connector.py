"""Pathway connector for video frame input.

VideoConnectorSubject bridges visualpath's Frame iterator to Pathway's
streaming input system. Frame objects are wrapped in pw.PyObjectWrapper
for transport through the Pathway Rust engine.

Example:
    >>> from visualpath.backends.pathway.connector import VideoConnectorSubject
    >>> import pathway as pw
    >>>
    >>> subject = VideoConnectorSubject(frames=video.stream())
    >>> table = pw.io.python.read(subject, schema=FrameSchema)
"""

from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


def _check_pathway() -> None:
    """Check if Pathway is available."""
    if not PATHWAY_AVAILABLE:
        raise ImportError(
            "Pathway is not installed. Install with: pip install visualpath[pathway]"
        )


if PATHWAY_AVAILABLE:
    class VideoConnectorSubject(pw.io.python.ConnectorSubject):
        """Pathway connector subject for video frames.

        Converts a Frame iterator into Pathway streaming input.
        Each frame becomes a row in the Pathway table with:
        - frame_id: Unique frame identifier
        - t_ns: Source timestamp in nanoseconds
        - frame: The Frame object wrapped in PyObjectWrapper

        Example:
            >>> frames = video.stream(fps=10)
            >>> subject = VideoConnectorSubject(frames)
            >>> table = pw.io.python.read(
            ...     subject,
            ...     schema=FrameSchema,
            ...     autocommit_duration_ms=100,
            ... )
        """

        def __init__(self, frames: Iterator["Frame"]) -> None:
            """Initialize the connector.

            Args:
                frames: Iterator of Frame objects from video source.
            """
            super().__init__()
            self._frames = frames

        def run(self) -> None:
            """Stream frames to Pathway.

            Called by Pathway to start consuming frames.
            Wraps each Frame in PyObjectWrapper for Pathway transport.
            """
            for frame in self._frames:
                self.next(
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    frame=pw.PyObjectWrapper(frame),
                )
            self.close()


    class FrameSchema(pw.Schema):
        """Pathway schema for video frames.

        Attributes:
            frame_id: Unique frame identifier.
            t_ns: Source timestamp in nanoseconds.
            frame: The Frame object (PyObjectWrapper).
        """
        frame_id: int
        t_ns: int
        frame: pw.PyObjectWrapper


    class ObservationSchema(pw.Schema):
        """Pathway schema for observations.

        Attributes:
            frame_id: Frame identifier.
            t_ns: Timestamp in nanoseconds.
            source: Extractor name.
            observation: The Observation object (PyObjectWrapper).
        """
        frame_id: int
        t_ns: int
        source: str
        observation: pw.PyObjectWrapper


    class TriggerSchema(pw.Schema):
        """Pathway schema for triggers.

        Attributes:
            frame_id: Frame identifier.
            t_ns: Timestamp in nanoseconds.
            trigger: The Trigger object (PyObjectWrapper).
        """
        frame_id: int
        t_ns: int
        trigger: pw.PyObjectWrapper


__all__ = [
    "VideoConnectorSubject",
    "FrameSchema",
    "ObservationSchema",
    "TriggerSchema",
    "PATHWAY_AVAILABLE",
]
