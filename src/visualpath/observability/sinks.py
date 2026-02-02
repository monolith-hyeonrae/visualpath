"""Trace output sinks for observability.

Sinks receive trace records and handle their output to various destinations:
- FileSink: JSONL file output
- ConsoleSink: Formatted console output
- MemorySink: In-memory buffer for testing/analysis
"""

import sys
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, TextIO, Callable

from visualpath.observability import Sink, TraceLevel
from visualpath.observability.records import (
    TraceRecord,
    TimingRecord,
    FrameDropRecord,
    SyncDelayRecord,
)


class FileSink(Sink):
    """Sink that writes trace records to a JSONL file.

    Each record is written as a single JSON line, suitable for
    post-processing with tools like jq.

    Args:
        path: Path to the output file.
        buffer_size: Number of records to buffer before flushing (default: 100).
        append: Whether to append to existing file (default: False).

    Example:
        >>> sink = FileSink("/tmp/trace.jsonl")
        >>> hub.add_sink(sink)
        >>> # ... processing ...
        >>> sink.close()  # Ensure final flush
    """

    def __init__(
        self,
        path: str,
        buffer_size: int = 100,
        append: bool = False,
    ):
        self._path = Path(path)
        self._buffer_size = buffer_size
        self._append = append

        self._buffer: List[str] = []
        self._file: Optional[TextIO] = None
        self._lock = threading.Lock()

        self._open_file()

    def _open_file(self) -> None:
        """Open the output file."""
        mode = "a" if self._append else "w"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, mode, encoding="utf-8")

    def write(self, record: TraceRecord) -> None:
        """Write a trace record to the file.

        Args:
            record: The trace record to write.
        """
        line = record.to_json()

        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the buffer to disk. Must be called with lock held."""
        if not self._buffer or self._file is None:
            return

        for line in self._buffer:
            self._file.write(line + "\n")
        self._file.flush()
        self._buffer.clear()

    def flush(self) -> None:
        """Flush any buffered records to disk."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            self._flush_buffer()
            if self._file is not None:
                self._file.close()
                self._file = None


class ConsoleSink(Sink):
    """Sink that writes formatted trace records to console.

    Provides human-readable output with color coding for different
    record types and severity levels.

    Args:
        stream: Output stream (default: sys.stderr).
        color: Enable ANSI color codes (default: True).
        show_timing_warnings: Show warnings for slow components (default: True).
        timing_threshold_ms: Threshold for timing warnings (default: 50.0).
        format_fn: Optional custom format function for records.

    Example:
        >>> sink = ConsoleSink()
        >>> hub.add_sink(sink)
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "gray": "\033[90m",
    }

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        color: bool = True,
        show_timing_warnings: bool = True,
        timing_threshold_ms: float = 50.0,
        format_fn: Optional[Callable[[TraceRecord], Optional[str]]] = None,
    ):
        self._stream = stream or sys.stderr
        self._color = color and self._stream.isatty()
        self._show_timing_warnings = show_timing_warnings
        self._timing_threshold_ms = timing_threshold_ms
        self._format_fn = format_fn
        self._lock = threading.Lock()

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if not self._color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def write(self, record: TraceRecord) -> None:
        """Write a formatted trace record to console.

        Args:
            record: The trace record to write.
        """
        # Use custom format function if provided
        if self._format_fn:
            line = self._format_fn(record)
        else:
            line = self._format_record(record)

        if line:
            with self._lock:
                self._stream.write(line + "\n")
                self._stream.flush()

    def _format_record(self, record: TraceRecord) -> Optional[str]:
        """Format a record for console output.

        Args:
            record: The trace record to format.

        Returns:
            Formatted string or None to skip output.
        """
        if isinstance(record, TimingRecord):
            return self._format_timing(record)
        elif isinstance(record, FrameDropRecord):
            return self._format_frame_drop(record)
        elif isinstance(record, SyncDelayRecord):
            return self._format_sync_delay(record)
        else:
            # Skip other record types by default
            return None

    def _format_timing(self, record: TimingRecord) -> Optional[str]:
        """Format timing record (only show warnings)."""
        if not self._show_timing_warnings:
            return None

        if record.processing_ms <= self._timing_threshold_ms:
            return None

        tag = self._colorize("[TIMING]", "yellow")
        component = self._colorize(record.component, "cyan")
        warning = self._colorize(
            f"{record.processing_ms:.0f}ms (> {self._timing_threshold_ms:.0f}ms threshold)",
            "red"
        )
        return f"{tag} Frame {record.frame_id}: {component} took {warning}"

    def _format_frame_drop(self, record: FrameDropRecord) -> str:
        """Format frame drop record."""
        tag = self._colorize("[DROP]", "red")
        frames = ", ".join(str(f) for f in record.dropped_frame_ids[:5])
        if len(record.dropped_frame_ids) > 5:
            frames += f" +{len(record.dropped_frame_ids) - 5} more"
        return f"{tag} Frames {frames} dropped ({record.reason})"

    def _format_sync_delay(self, record: SyncDelayRecord) -> str:
        """Format sync delay record."""
        tag = self._colorize("[SYNC]", "yellow")
        waiting = ", ".join(record.waiting_for)
        return (
            f"{tag} Frame {record.frame_id} delayed {record.delay_ms:.0f}ms "
            f"waiting for: {waiting}"
        )

    def flush(self) -> None:
        """Flush the output stream."""
        with self._lock:
            self._stream.flush()


class MemorySink(Sink):
    """Sink that stores trace records in memory.

    Useful for testing and for in-session analysis.

    Args:
        max_records: Maximum number of records to keep (default: 10000).

    Example:
        >>> sink = MemorySink()
        >>> hub.add_sink(sink)
        >>> # ... processing ...
        >>> records = sink.get_records()
        >>> timing_records = [r for r in records if isinstance(r, TimingRecord)]
    """

    def __init__(self, max_records: int = 10000):
        self._max_records = max_records
        self._records: Deque[TraceRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def write(self, record: TraceRecord) -> None:
        """Store a trace record in memory.

        Args:
            record: The trace record to store.
        """
        with self._lock:
            self._records.append(record)

    def get_records(self, record_type: Optional[str] = None) -> List[TraceRecord]:
        """Get stored records.

        Args:
            record_type: Optional filter by record type.

        Returns:
            List of trace records.
        """
        with self._lock:
            records = list(self._records)

        if record_type:
            records = [r for r in records if r.record_type == record_type]

        return records

    def get_by_frame(self, frame_id: int) -> List[TraceRecord]:
        """Get all records for a specific frame.

        Args:
            frame_id: Frame ID to filter by.

        Returns:
            List of trace records for the frame.
        """
        with self._lock:
            records = list(self._records)

        return [
            r for r in records
            if hasattr(r, "frame_id") and r.frame_id == frame_id
        ]

    def get_timing_stats(self) -> dict:
        """Get timing statistics from stored records.

        Returns:
            Dict with per-component timing stats.
        """
        from collections import defaultdict

        timing_records = [
            r for r in self.get_records()
            if isinstance(r, TimingRecord)
        ]

        if not timing_records:
            return {}

        stats: dict = defaultdict(lambda: {"times": [], "slow_count": 0})

        for record in timing_records:
            comp = record.component
            stats[comp]["times"].append(record.processing_ms)
            if record.is_slow:
                stats[comp]["slow_count"] += 1

        result = {}
        for comp, data in stats.items():
            times = data["times"]
            result[comp] = {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "max_ms": max(times),
                "min_ms": min(times),
                "slow_count": data["slow_count"],
            }

        return result

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        """Number of stored records."""
        return len(self._records)


class NullSink(Sink):
    """Sink that discards all records.

    Useful for testing or when tracing is enabled but output is not needed.
    """

    def write(self, record: TraceRecord) -> None:
        """Discard the record."""
        pass


__all__ = [
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
]
