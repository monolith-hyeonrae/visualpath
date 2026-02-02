"""Subprocess entry point for VenvWorker.

This module provides the subprocess side of VenvWorker's ZMQ-based
IPC mechanism. It loads an extractor via entry points and processes
frames received over ZMQ.

Usage:
    python -m visualpath.process.worker --extractor face --ipc-address ipc:///tmp/xxx.sock

Or via the entry point:
    visualpath-worker --extractor face --ipc-address ipc:///tmp/xxx.sock
"""

import argparse
import base64
import json
import logging
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _is_json_serializable(obj: Any) -> bool:
    """Check if object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON transmission.

    Args:
        value: Any value to serialize.

    Returns:
        JSON-serializable representation.
    """
    if value is None:
        return None

    if _is_json_serializable(value):
        return value

    # Handle dataclasses
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: _serialize_value(getattr(value, k))
            for k in value.__dataclass_fields__
        }

    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return repr(value)

    return str(value)


def _serialize_observation(obs: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Serialize observation for ZMQ transmission.

    Handles both visualpath.core.Observation and custom observation types
    (e.g., facemoment's Observation with faces instead of data).

    Args:
        obs: Observation to serialize.

    Returns:
        JSON-serializable dict, or None if obs is None.
    """
    if obs is None:
        return None

    # Build result from available attributes
    result = {
        "source": getattr(obs, "source", "unknown"),
        "frame_id": getattr(obs, "frame_id", -1),
        "t_ns": getattr(obs, "t_ns", 0),
        "signals": getattr(obs, "signals", {}),
        "metadata": getattr(obs, "metadata", {}),
        "timing": getattr(obs, "timing", None),
    }

    # Handle visualpath's 'data' field
    if hasattr(obs, "data"):
        result["data"] = _serialize_value(obs.data)

    # Handle facemoment's 'faces' field
    if hasattr(obs, "faces"):
        result["faces"] = _serialize_value(obs.faces)

    return result


def _deserialize_frame(data: Dict[str, Any]) -> "Frame":
    """Deserialize frame from ZMQ message.

    Args:
        data: Dict containing serialized frame data.

    Returns:
        Reconstructed Frame object.
    """
    import cv2
    from visualbase import Frame

    # Decode JPEG data
    jpeg_bytes = base64.b64decode(data["data_b64"])
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode frame image data")

    return Frame.from_array(
        img,
        frame_id=data["frame_id"],
        t_src_ns=data["t_src_ns"],
    )


def run_worker(extractor_name: str, ipc_address: str) -> int:
    """Run the worker process main loop.

    Args:
        extractor_name: Name of the extractor to load via entry points.
        ipc_address: ZMQ IPC address to bind to.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    import zmq
    from visualpath.plugin import create_extractor
    from visualpath.core import Observation  # noqa: F401 - used in type hints

    # Create ZMQ socket (REP pattern - reply to requests)
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    try:
        socket.bind(ipc_address)
        logger.info(f"Worker bound to {ipc_address}")
    except zmq.ZMQError as e:
        logger.error(f"Failed to bind to {ipc_address}: {e}")
        return 1

    # Load and initialize extractor
    try:
        extractor = create_extractor(extractor_name)
        extractor.initialize()
        logger.info(f"Loaded extractor: {extractor_name}")
    except Exception as e:
        logger.error(f"Failed to load extractor '{extractor_name}': {e}")
        socket.close()
        context.term()
        return 1

    try:
        while True:
            try:
                # Receive message (blocking)
                message = socket.recv_json()
            except zmq.ZMQError as e:
                logger.error(f"ZMQ receive error: {e}")
                break

            msg_type = message.get("type")

            if msg_type == "ping":
                # Handshake
                socket.send_json({"type": "pong", "extractor": extractor_name})
                continue

            if msg_type == "shutdown":
                # Clean shutdown
                socket.send_json({"type": "ack"})
                logger.info("Received shutdown signal")
                break

            if msg_type == "extract":
                # Process frame
                try:
                    frame = _deserialize_frame(message["frame"])
                    observation = extractor.extract(frame)
                    socket.send_json({
                        "observation": _serialize_observation(observation),
                    })
                except Exception as e:
                    logger.error(f"Extraction error: {e}")
                    socket.send_json({
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                continue

            # Unknown message type
            logger.warning(f"Unknown message type: {msg_type}")
            socket.send_json({"error": f"Unknown message type: {msg_type}"})

    finally:
        # Cleanup
        try:
            extractor.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        socket.close()
        context.term()
        logger.info("Worker shutdown complete")

    return 0


def main() -> int:
    """Main entry point for the worker subprocess."""
    parser = argparse.ArgumentParser(
        description="visualpath worker subprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extractor",
        required=True,
        help="Name of the extractor to load (via entry points)",
    )
    parser.add_argument(
        "--ipc-address",
        required=True,
        help="ZMQ IPC address to bind to (e.g., ipc:///tmp/worker.sock)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    return run_worker(args.extractor, args.ipc_address)


if __name__ == "__main__":
    sys.exit(main())
