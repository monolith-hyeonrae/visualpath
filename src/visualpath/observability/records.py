"""Trace record data classes for observability.

This module defines the base trace record types used throughout
the visualpath observability system.

Record Categories:
- Base: TraceRecord base class
- Timing: Component performance metrics
- Frame: Frame processing records
- Session: Session start/end records
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import time
import json


# Forward reference for TraceLevel
from visualpath.observability import TraceLevel


@dataclass
class TraceRecord:
    """Base class for all trace records.

    All trace records have:
    - record_type: String identifying the record type
    - timestamp_ns: When the record was created (monotonic)
    - min_level: Minimum trace level required to emit this record

    Subclasses should set record_type as a class variable.
    """
    record_type: str = field(default="base", init=False)
    timestamp_ns: int = field(default_factory=lambda: time.perf_counter_ns())
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = asdict(self)
        # Remove min_level from output (internal use only)
        d.pop("min_level", None)
        return d

    def to_json(self) -> str:
        """Convert record to JSON string.

        Returns:
            JSON-serialized record.
        """
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# Timing Records
# =============================================================================


@dataclass
class TimingRecord(TraceRecord):
    """Component processing time record.

    Emitted after each frame processing to track performance.
    """
    record_type: str = field(default="timing", init=False)
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    frame_id: int = 0
    component: str = ""  # e.g., "face", "pose", "fusion", "orchestrator"

    processing_ms: float = 0.0
    queue_depth: int = 0

    # Thresholds for warnings
    threshold_ms: float = 50.0  # Default warning threshold
    is_slow: bool = False


@dataclass
class FrameDropRecord(TraceRecord):
    """Record of dropped frames.

    Emitted when frames are dropped due to processing delays.
    """
    record_type: str = field(default="frame_drop", init=False)

    dropped_frame_ids: List[int] = field(default_factory=list)
    reason: str = ""  # "timeout", "backpressure", "queue_full"

    # Context
    queue_depth: int = 0
    processing_ms: float = 0.0


@dataclass
class SyncDelayRecord(TraceRecord):
    """Record of observation synchronization delay.

    Emitted when fusion waits for observations from multiple extractors.
    """
    record_type: str = field(default="sync_delay", init=False)

    frame_id: int = 0
    expected_ns: int = 0
    actual_ns: int = 0
    delay_ms: float = 0.0

    waiting_for: List[str] = field(default_factory=list)


@dataclass
class FPSRecord(TraceRecord):
    """Periodic FPS and performance summary.

    Emitted periodically (e.g., every 100 frames) with aggregate stats.
    """
    record_type: str = field(default="fps_summary", init=False)
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    # Frame range
    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0

    # FPS
    actual_fps: float = 0.0
    target_fps: float = 0.0
    fps_ratio: float = 0.0  # actual / target

    # Latency stats (ms)
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Per-component average times
    component_avg_ms: Dict[str, float] = field(default_factory=dict)

    # Issues
    dropped_frames: int = 0
    slow_frames: int = 0  # Frames exceeding threshold


# =============================================================================
# Session Records
# =============================================================================


@dataclass
class SessionStartRecord(TraceRecord):
    """Record emitted when a processing session starts."""
    record_type: str = field(default="session_start", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    session_id: str = ""
    source_path: str = ""
    target_fps: float = 0.0

    # Configuration
    extractors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    trace_level: str = ""


@dataclass
class SessionEndRecord(TraceRecord):
    """Record emitted when a processing session ends."""
    record_type: str = field(default="session_end", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    session_id: str = ""
    duration_sec: float = 0.0

    # Summary stats
    total_frames: int = 0
    total_triggers: int = 0  # Alias for compatibility
    total_events: int = 0
    total_dropped: int = 0
    avg_fps: float = 0.0


# =============================================================================
# Generic Extraction Records
# =============================================================================


@dataclass
class FrameExtractRecord(TraceRecord):
    """Generic record of frame extraction results.

    Emitted by extractors after processing each frame.
    Contains summary of what was extracted.
    """
    record_type: str = field(default="frame_extract", init=False)

    frame_id: int = 0
    t_ns: int = 0
    source: str = ""

    # Summary (NORMAL level)
    object_count: int = 0
    signals: Dict[str, float] = field(default_factory=dict)

    # Processing time
    processing_ms: float = 0.0


__all__ = [
    "TraceRecord",
    "TimingRecord",
    "FrameDropRecord",
    "SyncDelayRecord",
    "FPSRecord",
    "SessionStartRecord",
    "SessionEndRecord",
    "FrameExtractRecord",
]
