"""Observability system for visualpath.

Provides tracing and logging infrastructure to track:
- Frame-by-frame extraction results
- Component timing and performance
- Stream synchronization issues

Trace Levels:
- OFF: No tracing (production default)
- MINIMAL: Important events only (<1% overhead)
- NORMAL: Frame summaries + state changes (~5% overhead)
- VERBOSE: Full signal details + timing (~15% overhead)

Example:
    >>> from visualpath.observability import ObservabilityHub, TraceLevel
    >>> hub = ObservabilityHub.get_instance()
    >>> hub.configure(level=TraceLevel.NORMAL)
    >>> hub.add_sink(FileSink("/tmp/trace.jsonl"))
    >>>
    >>> # In extractor code:
    >>> if hub.enabled:
    ...     hub.emit(FrameExtractRecord(...))
"""

from enum import IntEnum
from typing import List, Optional
import threading


class TraceLevel(IntEnum):
    """Observability trace levels.

    Higher levels include all lower level information.
    """
    OFF = 0       # No tracing
    MINIMAL = 1   # Important events only
    NORMAL = 2    # Frame summaries + state changes
    VERBOSE = 3   # Full details + timing graphs


class Sink:
    """Base class for trace sinks.

    Sinks receive trace records and handle their output
    (file, console, memory buffer, etc.).
    """

    def write(self, record: "TraceRecord") -> None:
        """Write a trace record.

        Args:
            record: The trace record to write.
        """
        raise NotImplementedError

    def flush(self) -> None:
        """Flush any buffered records."""
        pass

    def close(self) -> None:
        """Close the sink and release resources."""
        pass


class ObservabilityHub:
    """Central hub for observability configuration and record emission.

    Singleton pattern - use get_instance() to access.

    The hub manages:
    - Global trace level configuration
    - Registered sinks for record output
    - Fast-path enabled check for minimal overhead when off

    Thread Safety:
        The hub is thread-safe. Records can be emitted from multiple threads.

    Example:
        >>> hub = ObservabilityHub.get_instance()
        >>> hub.configure(level=TraceLevel.NORMAL)
        >>> hub.add_sink(ConsoleSink())
        >>>
        >>> # Fast check before creating records
        >>> if hub.enabled:
        ...     hub.emit(record)
    """

    _instance: Optional["ObservabilityHub"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the hub. Use get_instance() instead."""
        self._level = TraceLevel.OFF
        self._sinks: List[Sink] = []
        self._emit_lock = threading.Lock()

        # Cached state for fast checks
        self._enabled = False

    @classmethod
    def get_instance(cls) -> "ObservabilityHub":
        """Get the singleton hub instance.

        Returns:
            The global ObservabilityHub instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. For testing only."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    def configure(
        self,
        level: TraceLevel = TraceLevel.OFF,
        sinks: Optional[List[Sink]] = None,
    ) -> None:
        """Configure the observability hub.

        Args:
            level: Trace level to set.
            sinks: Optional list of sinks to add.
        """
        self._level = level
        self._enabled = level > TraceLevel.OFF

        if sinks:
            for sink in sinks:
                self.add_sink(sink)

    def add_sink(self, sink: Sink) -> None:
        """Add a sink for trace output.

        Args:
            sink: Sink to add.
        """
        with self._emit_lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        """Remove a sink.

        Args:
            sink: Sink to remove.
        """
        with self._emit_lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def emit(self, record: "TraceRecord") -> None:
        """Emit a trace record to all sinks.

        Args:
            record: The trace record to emit.
        """
        if not self._enabled:
            return

        # Check if record's minimum level is met
        if record.min_level > self._level:
            return

        with self._emit_lock:
            for sink in self._sinks:
                try:
                    sink.write(record)
                except Exception:
                    # Silently ignore sink errors to avoid affecting main processing
                    pass

    def flush(self) -> None:
        """Flush all sinks."""
        with self._emit_lock:
            for sink in self._sinks:
                try:
                    sink.flush()
                except Exception:
                    pass

    def shutdown(self) -> None:
        """Shutdown the hub and close all sinks."""
        with self._emit_lock:
            for sink in self._sinks:
                try:
                    sink.flush()
                    sink.close()
                except Exception:
                    pass
            self._sinks.clear()

        self._level = TraceLevel.OFF
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Fast check if tracing is enabled.

        Use this before creating trace records to minimize overhead
        when tracing is disabled.

        Returns:
            True if trace level > OFF.
        """
        return self._enabled

    @property
    def level(self) -> TraceLevel:
        """Current trace level.

        Returns:
            The configured trace level.
        """
        return self._level

    def is_level_enabled(self, level: TraceLevel) -> bool:
        """Check if a specific trace level is enabled.

        Args:
            level: Level to check.

        Returns:
            True if the specified level is enabled.
        """
        return self._level >= level


# Import TraceRecord and sinks after defining TraceLevel
from visualpath.observability.records import TraceRecord
from visualpath.observability.sinks import FileSink, ConsoleSink, MemorySink, NullSink

__all__ = [
    # Core
    "TraceLevel",
    "Sink",
    "ObservabilityHub",
    # Records
    "TraceRecord",
    # Sinks
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
]
