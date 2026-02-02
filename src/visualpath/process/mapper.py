"""ObservationMapper protocol for serialization.

This module defines the ObservationMapper protocol that allows
plugins to define how Observations are serialized for IPC.

Example:
    >>> from visualpath.process.mapper import ObservationMapper
    >>>
    >>> class FaceObservationMapper(ObservationMapper):
    ...     def to_message(self, observation: Observation) -> Optional[str]:
    ...         if observation.source != "face":
    ...             return None
    ...         return FaceOBS(
    ...             frame_id=observation.frame_id,
    ...             t_ns=observation.t_ns,
    ...             faces=[...],
    ...         ).to_message()
    ...
    ...     def from_message(self, message: str) -> Optional[Observation]:
    ...         # Parse FaceOBS message and convert to Observation
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable, Any

from visualpath.core.extractor import Observation


@runtime_checkable
class ObservationMapper(Protocol):
    """Protocol for mapping Observations to/from wire format.

    Implementations convert domain-specific Observations to
    serialized messages for IPC, and back.
    """

    def to_message(self, observation: Observation) -> Optional[str]:
        """Convert an Observation to a serialized message.

        Args:
            observation: The observation to serialize.

        Returns:
            Serialized message string, or None if not handled.
        """
        ...

    def from_message(self, message: str) -> Optional[Observation]:
        """Convert a serialized message back to an Observation.

        Args:
            message: The serialized message.

        Returns:
            Observation, or None if parsing failed.
        """
        ...


class DefaultObservationMapper:
    """Default mapper that serializes observations as JSON.

    This is a simple implementation that uses JSON serialization.
    Plugins should provide specialized mappers for better performance.
    """

    def to_message(self, observation: Observation) -> Optional[str]:
        """Serialize observation to JSON."""
        import json

        try:
            data = {
                "source": observation.source,
                "frame_id": observation.frame_id,
                "t_ns": observation.t_ns,
                "signals": observation.signals,
                "metadata": observation.metadata,
            }
            if observation.timing:
                data["timing"] = observation.timing
            if observation.data is not None:
                # Attempt to serialize data
                data["data"] = self._serialize_data(observation.data)
            return json.dumps(data)
        except Exception:
            return None

    def from_message(self, message: str) -> Optional[Observation]:
        """Deserialize JSON to observation."""
        import json

        try:
            data = json.loads(message)
            return Observation(
                source=data["source"],
                frame_id=data["frame_id"],
                t_ns=data["t_ns"],
                signals=data.get("signals", {}),
                data=data.get("data"),
                metadata=data.get("metadata", {}),
                timing=data.get("timing"),
            )
        except Exception:
            return None

    def _serialize_data(self, data: Any) -> Any:
        """Attempt to serialize data field."""
        import json

        # Try JSON serialization
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # Data not JSON serializable, return string representation
            return str(data)


class CompositeMapper:
    """Mapper that delegates to multiple specialized mappers.

    Tries each mapper in order until one successfully handles
    the observation or message.

    Example:
        >>> composite = CompositeMapper([
        ...     FaceObservationMapper(),
        ...     PoseObservationMapper(),
        ...     DefaultObservationMapper(),  # Fallback
        ... ])
    """

    def __init__(self, mappers: list[ObservationMapper]):
        """Initialize with a list of mappers.

        Args:
            mappers: List of mappers to try, in order.
        """
        self._mappers = mappers

    def to_message(self, observation: Observation) -> Optional[str]:
        """Try each mapper until one succeeds."""
        for mapper in self._mappers:
            result = mapper.to_message(observation)
            if result is not None:
                return result
        return None

    def from_message(self, message: str) -> Optional[Observation]:
        """Try each mapper until one succeeds."""
        for mapper in self._mappers:
            result = mapper.from_message(message)
            if result is not None:
                return result
        return None

    def add_mapper(self, mapper: ObservationMapper) -> None:
        """Add a mapper (inserted at the beginning for priority).

        Args:
            mapper: Mapper to add.
        """
        self._mappers.insert(0, mapper)
