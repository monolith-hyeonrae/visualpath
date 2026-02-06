"""Shared serialization utilities for IPC communication.

Provides JSON-safe serialization and Observation (de)serialization
used by both the worker subprocess and the launcher (main process).
"""

import json
from typing import Any, Dict, Optional

from visualpath.core.extractor import Observation


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON transmission.

    Handles dataclasses, lists, tuples, dicts, and objects with __dict__.
    Falls back to repr() or str() for unserializable types.

    Args:
        value: Any value to serialize.

    Returns:
        JSON-serializable representation.
    """
    if value is None:
        return None

    # Fast path: already JSON-serializable
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    # Handle dataclasses
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: serialize_value(getattr(value, k))
            for k in value.__dataclass_fields__
        }

    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return repr(value)

    return str(value)


def serialize_observation(obs: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Serialize an Observation for ZMQ transmission.

    Works with any object that has the standard Observation attributes
    (source, frame_id, t_ns, signals, metadata, timing, data).

    Args:
        obs: Observation to serialize.

    Returns:
        JSON-serializable dict, or None if obs is None.
    """
    if obs is None:
        return None

    result: Dict[str, Any] = {
        "source": getattr(obs, "source", "unknown"),
        "frame_id": getattr(obs, "frame_id", -1),
        "t_ns": getattr(obs, "t_ns", 0),
        "signals": getattr(obs, "signals", {}),
        "metadata": getattr(obs, "metadata", {}),
        "timing": getattr(obs, "timing", None),
    }

    if hasattr(obs, "data") and obs.data is not None:
        result["data"] = serialize_value(obs.data)

    return result


def deserialize_observation(data: Optional[Dict[str, Any]]) -> Optional[Observation]:
    """Deserialize an Observation from a ZMQ message dict.

    Args:
        data: Dict containing serialized observation data.

    Returns:
        Reconstructed Observation object, or None.
    """
    if data is None:
        return None

    return Observation(
        source=data["source"],
        frame_id=data["frame_id"],
        t_ns=data["t_ns"],
        signals=data.get("signals", {}),
        data=data.get("data"),
        metadata=data.get("metadata", {}),
        timing=data.get("timing"),
    )


__all__ = [
    "serialize_value",
    "serialize_observation",
    "deserialize_observation",
]
