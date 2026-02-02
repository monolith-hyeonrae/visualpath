"""Tests for visualpath observability system."""

import pytest
import tempfile
import json
from pathlib import Path

from visualpath.observability import (
    TraceLevel,
    Sink,
    ObservabilityHub,
    TraceRecord,
    FileSink,
    ConsoleSink,
    MemorySink,
    NullSink,
)
from visualpath.observability.records import (
    TimingRecord,
    FrameDropRecord,
    SessionStartRecord,
    SessionEndRecord,
)


# =============================================================================
# TraceLevel Tests
# =============================================================================


class TestTraceLevel:
    """Tests for TraceLevel enum."""

    def test_level_ordering(self):
        """Test trace levels are ordered correctly."""
        assert TraceLevel.OFF < TraceLevel.MINIMAL
        assert TraceLevel.MINIMAL < TraceLevel.NORMAL
        assert TraceLevel.NORMAL < TraceLevel.VERBOSE

    def test_level_values(self):
        """Test trace level values."""
        assert TraceLevel.OFF == 0
        assert TraceLevel.MINIMAL == 1
        assert TraceLevel.NORMAL == 2
        assert TraceLevel.VERBOSE == 3


# =============================================================================
# TraceRecord Tests
# =============================================================================


class TestTraceRecord:
    """Tests for TraceRecord base class."""

    def test_base_record_creation(self):
        """Test creating a base trace record."""
        record = TraceRecord()

        assert record.record_type == "base"
        assert record.timestamp_ns > 0
        assert record.min_level == TraceLevel.NORMAL

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = TraceRecord()
        d = record.to_dict()

        assert "record_type" in d
        assert "timestamp_ns" in d
        assert "min_level" not in d  # Should be excluded

    def test_to_json(self):
        """Test converting record to JSON."""
        record = TraceRecord()
        json_str = record.to_json()

        data = json.loads(json_str)
        assert data["record_type"] == "base"


class TestTimingRecord:
    """Tests for TimingRecord."""

    def test_timing_record_creation(self):
        """Test creating a timing record."""
        record = TimingRecord(
            frame_id=42,
            component="face",
            processing_ms=25.5,
        )

        assert record.record_type == "timing"
        assert record.frame_id == 42
        assert record.component == "face"
        assert record.processing_ms == 25.5

    def test_slow_flag(self):
        """Test slow flag based on threshold."""
        record = TimingRecord(
            frame_id=1,
            component="test",
            processing_ms=100.0,
            threshold_ms=50.0,
            is_slow=True,
        )

        assert record.is_slow


# =============================================================================
# ObservabilityHub Tests
# =============================================================================


class TestObservabilityHub:
    """Tests for ObservabilityHub singleton."""

    def setup_method(self):
        """Reset hub before each test."""
        ObservabilityHub.reset_instance()

    def teardown_method(self):
        """Reset hub after each test."""
        ObservabilityHub.reset_instance()

    def test_singleton(self):
        """Test hub is a singleton."""
        hub1 = ObservabilityHub.get_instance()
        hub2 = ObservabilityHub.get_instance()

        assert hub1 is hub2

    def test_default_disabled(self):
        """Test hub is disabled by default."""
        hub = ObservabilityHub.get_instance()

        assert not hub.enabled
        assert hub.level == TraceLevel.OFF

    def test_configure(self):
        """Test configuring the hub."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)

        assert hub.enabled
        assert hub.level == TraceLevel.NORMAL

    def test_add_remove_sink(self):
        """Test adding and removing sinks."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()

        hub.add_sink(sink)
        assert sink in hub._sinks

        hub.remove_sink(sink)
        assert sink not in hub._sinks

    def test_emit_when_disabled(self):
        """Test emit does nothing when disabled."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.add_sink(sink)

        record = TraceRecord()
        hub.emit(record)

        assert len(sink) == 0

    def test_emit_when_enabled(self):
        """Test emit works when enabled."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)
        sink = MemorySink()
        hub.add_sink(sink)

        record = TraceRecord()
        hub.emit(record)

        assert len(sink) == 1

    def test_emit_respects_min_level(self):
        """Test emit respects record's min_level."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)
        sink = MemorySink()
        hub.add_sink(sink)

        # VERBOSE record should be filtered out at NORMAL level
        verbose_record = TimingRecord()
        verbose_record.min_level = TraceLevel.VERBOSE
        hub.emit(verbose_record)

        # NORMAL record should pass
        normal_record = TimingRecord()
        normal_record.min_level = TraceLevel.NORMAL
        hub.emit(normal_record)

        assert len(sink) == 1

    def test_is_level_enabled(self):
        """Test is_level_enabled method."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)

        assert hub.is_level_enabled(TraceLevel.MINIMAL)
        assert hub.is_level_enabled(TraceLevel.NORMAL)
        assert not hub.is_level_enabled(TraceLevel.VERBOSE)


# =============================================================================
# FileSink Tests
# =============================================================================


class TestFileSink:
    """Tests for FileSink."""

    def test_write_creates_file(self):
        """Test writing creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=1)

            record = TraceRecord()
            sink.write(record)
            sink.close()

            assert path.exists()
            content = path.read_text()
            assert "base" in content

    def test_buffered_writes(self):
        """Test records are buffered before writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=5)

            for i in range(3):
                sink.write(TraceRecord())

            # Should still be buffered
            content = path.read_text()
            assert content == ""

            # Flush to write
            sink.flush()
            content = path.read_text()
            assert content.count("\n") == 3

            sink.close()

    def test_jsonl_format(self):
        """Test output is valid JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=1)

            sink.write(TimingRecord(frame_id=1, component="test"))
            sink.write(TimingRecord(frame_id=2, component="test"))
            sink.close()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

            for line in lines:
                data = json.loads(line)
                assert data["record_type"] == "timing"


# =============================================================================
# MemorySink Tests
# =============================================================================


class TestMemorySink:
    """Tests for MemorySink."""

    def test_stores_records(self):
        """Test sink stores records."""
        sink = MemorySink()

        sink.write(TraceRecord())
        sink.write(TraceRecord())

        assert len(sink) == 2

    def test_max_records_limit(self):
        """Test max_records limit."""
        sink = MemorySink(max_records=3)

        for i in range(5):
            sink.write(TraceRecord())

        assert len(sink) == 3

    def test_get_records(self):
        """Test getting records."""
        sink = MemorySink()
        sink.write(TimingRecord(frame_id=1))
        sink.write(FrameDropRecord(dropped_frame_ids=[1, 2]))

        records = sink.get_records()
        assert len(records) == 2

    def test_get_records_by_type(self):
        """Test filtering records by type."""
        sink = MemorySink()
        sink.write(TimingRecord(frame_id=1))
        sink.write(FrameDropRecord(dropped_frame_ids=[1, 2]))

        timing_records = sink.get_records(record_type="timing")
        assert len(timing_records) == 1

    def test_get_by_frame(self):
        """Test getting records by frame ID."""
        sink = MemorySink()
        sink.write(TimingRecord(frame_id=1))
        sink.write(TimingRecord(frame_id=2))
        sink.write(TimingRecord(frame_id=1))

        frame_records = sink.get_by_frame(1)
        assert len(frame_records) == 2

    def test_get_timing_stats(self):
        """Test computing timing statistics."""
        sink = MemorySink()
        sink.write(TimingRecord(frame_id=1, component="face", processing_ms=10.0))
        sink.write(TimingRecord(frame_id=2, component="face", processing_ms=20.0))
        sink.write(TimingRecord(frame_id=3, component="face", processing_ms=30.0, is_slow=True))
        sink.write(TimingRecord(frame_id=1, component="pose", processing_ms=15.0))

        stats = sink.get_timing_stats()

        assert "face" in stats
        assert stats["face"]["count"] == 3
        assert stats["face"]["avg_ms"] == 20.0
        assert stats["face"]["slow_count"] == 1
        assert "pose" in stats

    def test_clear(self):
        """Test clearing records."""
        sink = MemorySink()
        sink.write(TraceRecord())
        sink.write(TraceRecord())

        sink.clear()

        assert len(sink) == 0


# =============================================================================
# NullSink Tests
# =============================================================================


class TestNullSink:
    """Tests for NullSink."""

    def test_discards_records(self):
        """Test sink discards all records."""
        sink = NullSink()

        # Should not raise
        sink.write(TraceRecord())
        sink.write(TraceRecord())
        sink.flush()
        sink.close()


# =============================================================================
# ConsoleSink Tests
# =============================================================================


class TestConsoleSink:
    """Tests for ConsoleSink."""

    def test_formats_timing_warning(self):
        """Test timing warning formatting."""
        import io
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, color=False, timing_threshold_ms=50.0)

        sink.write(TimingRecord(frame_id=1, component="face", processing_ms=100.0))
        sink.flush()

        output = stream.getvalue()
        assert "[TIMING]" in output
        assert "100ms" in output

    def test_skips_fast_timing(self):
        """Test fast timings are skipped."""
        import io
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, color=False, timing_threshold_ms=50.0)

        sink.write(TimingRecord(frame_id=1, component="face", processing_ms=25.0))
        sink.flush()

        output = stream.getvalue()
        assert output == ""

    def test_formats_frame_drop(self):
        """Test frame drop formatting."""
        import io
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, color=False)

        sink.write(FrameDropRecord(dropped_frame_ids=[1, 2, 3], reason="timeout"))
        sink.flush()

        output = stream.getvalue()
        assert "[DROP]" in output
        assert "timeout" in output
