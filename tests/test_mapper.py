"""Tests for ObservationMapper implementations."""

import pytest
import json
from typing import Optional

from visualpath.core import Observation
from visualpath.process.mapper import (
    ObservationMapper,
    DefaultObservationMapper,
    CompositeMapper,
)


# =============================================================================
# Custom Mapper for Testing
# =============================================================================


class TestSourceMapper:
    """Mapper that only handles observations from a specific source."""

    def __init__(self, source: str):
        self._source = source

    def to_message(self, observation: Observation) -> Optional[str]:
        if observation.source != self._source:
            return None
        return f"[{self._source}]:{observation.frame_id}:{observation.t_ns}"

    def from_message(self, message: str) -> Optional[Observation]:
        if not message.startswith(f"[{self._source}]"):
            return None
        parts = message.split(":")
        return Observation(
            source=self._source,
            frame_id=int(parts[1]),
            t_ns=int(parts[2]),
        )


# =============================================================================
# DefaultObservationMapper Tests
# =============================================================================


class TestDefaultObservationMapper:
    """Tests for DefaultObservationMapper."""

    def test_to_message_basic(self):
        """Test serializing a basic observation."""
        mapper = DefaultObservationMapper()
        obs = Observation(
            source="test",
            frame_id=42,
            t_ns=1234567890,
        )

        message = mapper.to_message(obs)

        assert message is not None
        data = json.loads(message)
        assert data["source"] == "test"
        assert data["frame_id"] == 42
        assert data["t_ns"] == 1234567890

    def test_to_message_with_signals(self):
        """Test serializing observation with signals."""
        mapper = DefaultObservationMapper()
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000,
            signals={"score": 0.95, "count": 5},
        )

        message = mapper.to_message(obs)

        data = json.loads(message)
        assert data["signals"]["score"] == 0.95
        assert data["signals"]["count"] == 5

    def test_to_message_with_metadata(self):
        """Test serializing observation with metadata."""
        mapper = DefaultObservationMapper()
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000,
            metadata={"key": "value"},
        )

        message = mapper.to_message(obs)

        data = json.loads(message)
        assert data["metadata"]["key"] == "value"

    def test_to_message_with_timing(self):
        """Test serializing observation with timing."""
        mapper = DefaultObservationMapper()
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000,
            timing={"detect_ms": 10.5, "process_ms": 5.2},
        )

        message = mapper.to_message(obs)

        data = json.loads(message)
        assert data["timing"]["detect_ms"] == 10.5

    def test_to_message_with_serializable_data(self):
        """Test serializing observation with JSON-serializable data."""
        mapper = DefaultObservationMapper()
        obs = Observation(
            source="test",
            frame_id=1,
            t_ns=1000,
            data=[{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
        )

        message = mapper.to_message(obs)

        data = json.loads(message)
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == 1

    def test_from_message_basic(self):
        """Test deserializing a basic message."""
        mapper = DefaultObservationMapper()
        message = json.dumps({
            "source": "test",
            "frame_id": 42,
            "t_ns": 1234567890,
        })

        obs = mapper.from_message(message)

        assert obs is not None
        assert obs.source == "test"
        assert obs.frame_id == 42
        assert obs.t_ns == 1234567890

    def test_from_message_with_all_fields(self):
        """Test deserializing message with all fields."""
        mapper = DefaultObservationMapper()
        message = json.dumps({
            "source": "test",
            "frame_id": 1,
            "t_ns": 1000,
            "signals": {"score": 0.9},
            "metadata": {"key": "value"},
            "timing": {"ms": 10.0},
            "data": [1, 2, 3],
        })

        obs = mapper.from_message(message)

        assert obs is not None
        assert obs.signals["score"] == 0.9
        assert obs.metadata["key"] == "value"
        assert obs.timing["ms"] == 10.0
        assert obs.data == [1, 2, 3]

    def test_from_message_invalid_json(self):
        """Test deserializing invalid JSON returns None."""
        mapper = DefaultObservationMapper()

        obs = mapper.from_message("not valid json")

        assert obs is None

    def test_roundtrip(self):
        """Test serializing and deserializing produces equivalent observation."""
        mapper = DefaultObservationMapper()
        original = Observation(
            source="test",
            frame_id=42,
            t_ns=1234567890,
            signals={"a": 1.0, "b": 2.0},
            metadata={"key": "value"},
        )

        message = mapper.to_message(original)
        restored = mapper.from_message(message)

        assert restored is not None
        assert restored.source == original.source
        assert restored.frame_id == original.frame_id
        assert restored.t_ns == original.t_ns
        assert restored.signals == original.signals
        assert restored.metadata == original.metadata


# =============================================================================
# CompositeMapper Tests
# =============================================================================


class TestCompositeMapper:
    """Tests for CompositeMapper."""

    def test_to_message_first_match(self):
        """Test that first matching mapper is used."""
        mapper1 = TestSourceMapper("face")
        mapper2 = TestSourceMapper("pose")
        composite = CompositeMapper([mapper1, mapper2])

        obs = Observation(source="face", frame_id=1, t_ns=1000)
        message = composite.to_message(obs)

        assert message == "[face]:1:1000"

    def test_to_message_second_match(self):
        """Test that second mapper is tried if first doesn't match."""
        mapper1 = TestSourceMapper("face")
        mapper2 = TestSourceMapper("pose")
        composite = CompositeMapper([mapper1, mapper2])

        obs = Observation(source="pose", frame_id=2, t_ns=2000)
        message = composite.to_message(obs)

        assert message == "[pose]:2:2000"

    def test_to_message_no_match(self):
        """Test that None is returned if no mapper matches."""
        mapper1 = TestSourceMapper("face")
        mapper2 = TestSourceMapper("pose")
        composite = CompositeMapper([mapper1, mapper2])

        obs = Observation(source="unknown", frame_id=1, t_ns=1000)
        message = composite.to_message(obs)

        assert message is None

    def test_to_message_with_fallback(self):
        """Test fallback to default mapper."""
        mapper1 = TestSourceMapper("face")
        fallback = DefaultObservationMapper()
        composite = CompositeMapper([mapper1, fallback])

        obs = Observation(source="unknown", frame_id=1, t_ns=1000)
        message = composite.to_message(obs)

        # Should fall back to default mapper
        assert message is not None
        data = json.loads(message)
        assert data["source"] == "unknown"

    def test_from_message_first_match(self):
        """Test that first matching mapper is used for parsing."""
        mapper1 = TestSourceMapper("face")
        mapper2 = TestSourceMapper("pose")
        composite = CompositeMapper([mapper1, mapper2])

        obs = composite.from_message("[face]:1:1000")

        assert obs is not None
        assert obs.source == "face"
        assert obs.frame_id == 1

    def test_from_message_no_match(self):
        """Test None returned if no mapper can parse."""
        mapper1 = TestSourceMapper("face")
        mapper2 = TestSourceMapper("pose")
        composite = CompositeMapper([mapper1, mapper2])

        obs = composite.from_message("invalid message format")

        assert obs is None

    def test_add_mapper_priority(self):
        """Test that added mappers get priority."""
        mapper1 = TestSourceMapper("old")
        composite = CompositeMapper([mapper1])

        # Add a new mapper that handles "face"
        mapper2 = TestSourceMapper("face")
        composite.add_mapper(mapper2)

        obs = Observation(source="face", frame_id=1, t_ns=1000)
        message = composite.to_message(obs)

        # New mapper should have priority
        assert message == "[face]:1:1000"

    def test_empty_composite(self):
        """Test empty composite returns None."""
        composite = CompositeMapper([])

        obs = Observation(source="test", frame_id=1, t_ns=1000)

        assert composite.to_message(obs) is None
        assert composite.from_message("any") is None


# =============================================================================
# Protocol Tests
# =============================================================================


class TestObservationMapperProtocol:
    """Tests for ObservationMapper protocol."""

    def test_default_mapper_implements_protocol(self):
        """Test that DefaultObservationMapper implements protocol."""
        mapper = DefaultObservationMapper()
        assert isinstance(mapper, ObservationMapper)

    def test_custom_mapper_implements_protocol(self):
        """Test that custom mapper can implement protocol."""
        mapper = TestSourceMapper("test")
        # Protocol check at runtime
        assert hasattr(mapper, "to_message")
        assert hasattr(mapper, "from_message")
