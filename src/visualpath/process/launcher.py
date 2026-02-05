"""Worker launchers for different isolation levels.

This module provides worker launchers that execute extractors with
different isolation strategies:

- InlineWorker: Same process, same thread (IsolationLevel.INLINE)
- ThreadWorker: Same process, different thread (IsolationLevel.THREAD)
- ProcessWorker: Same venv, different process (IsolationLevel.PROCESS)
- VenvWorker: Different venv, different process (IsolationLevel.VENV)

Example:
    >>> from visualpath.process.launcher import WorkerLauncher
    >>> from visualpath.core import IsolationLevel
    >>>
    >>> # Create launcher based on isolation level
    >>> launcher = WorkerLauncher.create(
    ...     level=IsolationLevel.THREAD,
    ...     extractor=my_extractor,
    ... )
    >>> launcher.start()
    >>> result = launcher.process(frame)
    >>> launcher.stop()

For VenvWorker/ProcessWorker with ZMQ:
    >>> # Run extractor in a separate venv
    >>> worker = WorkerLauncher.create(
    ...     level=IsolationLevel.VENV,
    ...     extractor=None,  # Will be loaded in subprocess
    ...     venv_path="/path/to/venv",
    ...     extractor_name="face",  # Entry point name
    ... )
    >>> worker.start()
    >>> result = worker.process(frame)
    >>> worker.stop()
"""

import base64
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from visualpath.core.extractor import Observation
from visualpath.core.module import Module
from visualpath.core.isolation import IsolationLevel

if TYPE_CHECKING:
    from visualbase import Frame

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result from a worker processing a frame.

    Attributes:
        observation: The extracted observation, or None if extraction failed.
        error: Optional error message if extraction failed.
        timing_ms: Processing time in milliseconds.
    """
    observation: Optional[Observation]
    error: Optional[str] = None
    timing_ms: float = 0.0


class BaseWorker(ABC):
    """Abstract base class for workers.

    Workers handle frame processing with a specific isolation strategy.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the worker."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        ...

    @abstractmethod
    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame and return the result.

        Args:
            frame: The frame to process.
            deps: Optional dict of observations from dependent extractors.

        Returns:
            WorkerResult with observation or error.
        """
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the worker is running."""
        ...


class InlineWorker(BaseWorker):
    """Worker that runs extraction inline (same process, same thread).

    This is the simplest worker with zero overhead, but no isolation.
    Suitable for lightweight extractors that don't need isolation.
    """

    def __init__(self, extractor: Module):
        """Initialize the inline worker.

        Args:
            extractor: The extractor to run.
        """
        self._extractor = extractor
        self._running = False

    def start(self) -> None:
        """Start the worker (initializes extractor)."""
        self._extractor.initialize()
        self._running = True

    def stop(self) -> None:
        """Stop the worker (cleans up extractor)."""
        self._extractor.cleanup()
        self._running = False

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame inline."""
        import time
        start = time.perf_counter()

        try:
            extractor_deps = None
            if deps and self._extractor.depends:
                extractor_deps = {
                    name: deps[name]
                    for name in self._extractor.depends
                    if name in deps
                }
            # Support both Module.process() and legacy extract()
            if hasattr(self._extractor, 'process') and not hasattr(self._extractor, 'extract'):
                try:
                    obs = self._extractor.process(frame, extractor_deps)
                except TypeError:
                    obs = self._extractor.process(frame)
            else:
                try:
                    obs = self._extractor.extract(frame, extractor_deps)
                except TypeError:
                    obs = self._extractor.extract(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=obs, timing_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=None, error=str(e), timing_ms=elapsed_ms)

    @property
    def is_running(self) -> bool:
        return self._running


class ThreadWorker(BaseWorker):
    """Worker that runs extraction in a separate thread.

    Provides thread-level isolation. Useful for I/O-bound work or
    extractors that can benefit from concurrent execution.
    """

    def __init__(
        self,
        extractor: Module,
        queue_size: int = 10,
    ):
        """Initialize the thread worker.

        Args:
            extractor: The extractor to run.
            queue_size: Maximum pending frames in queue.
        """
        self._extractor = extractor
        self._queue_size = queue_size

        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False

    def start(self) -> None:
        """Start the worker thread pool."""
        self._extractor.initialize()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running = True

    def stop(self) -> None:
        """Stop the worker and clean up."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._extractor.cleanup()

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Submit frame for processing and wait for result."""
        import time

        if not self._running or self._executor is None:
            return WorkerResult(observation=None, error="Worker not running")

        start = time.perf_counter()

        try:
            extractor_deps = None
            if deps and self._extractor.depends:
                extractor_deps = {
                    name: deps[name]
                    for name in self._extractor.depends
                    if name in deps
                }

            def _do_extract():
                # Support both Module.process() and legacy extract()
                if hasattr(self._extractor, 'process') and not hasattr(self._extractor, 'extract'):
                    try:
                        return self._extractor.process(frame, extractor_deps)
                    except TypeError:
                        return self._extractor.process(frame)
                else:
                    try:
                        return self._extractor.extract(frame, extractor_deps)
                    except TypeError:
                        return self._extractor.extract(frame)

            future = self._executor.submit(_do_extract)
            obs = future.result()  # Blocking wait
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=obs, timing_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return WorkerResult(observation=None, error=str(e), timing_ms=elapsed_ms)

    def process_async(self, frame: "Frame") -> Future:
        """Submit frame for processing without waiting.

        Args:
            frame: The frame to process.

        Returns:
            Future that will contain the Observation.
        """
        if not self._running or self._executor is None:
            raise RuntimeError("Worker not running")
        return self._executor.submit(self._extractor.extract, frame)

    @property
    def is_running(self) -> bool:
        return self._running


def _check_zmq_available() -> bool:
    """Check if pyzmq is available."""
    try:
        import zmq  # noqa: F401
        return True
    except ImportError:
        return False


def _serialize_frame(frame: "Frame", jpeg_quality: int = 95) -> Dict[str, Any]:
    """Serialize frame for ZMQ transmission.

    Args:
        frame: Frame to serialize.
        jpeg_quality: JPEG compression quality (0-100).

    Returns:
        JSON-serializable dict containing frame data.
    """
    import cv2

    # Encode as JPEG for efficient transmission
    _, jpeg_data = cv2.imencode(
        '.jpg', frame.data,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )

    return {
        "frame_id": frame.frame_id,
        "t_src_ns": frame.t_src_ns,
        "width": frame.width,
        "height": frame.height,
        "data_b64": base64.b64encode(jpeg_data.tobytes()).decode('ascii'),
    }


def _deserialize_observation(data: Optional[Dict[str, Any]]) -> Optional[Observation]:
    """Deserialize observation from ZMQ message.

    Returns a visualpath Observation. Domain-specific fields like 'faces'
    are preserved in the 'data' field for downstream processing.

    Args:
        data: Dict containing serialized observation data.

    Returns:
        Reconstructed Observation object, or None.
    """
    if data is None:
        return None

    # Collect domain-specific data (faces, etc.) into the data field
    extra_data = data.get("data")
    if "faces" in data:
        # For facemoment observations, put faces in data
        if extra_data is None:
            extra_data = {"faces": data["faces"]}
        elif isinstance(extra_data, dict):
            extra_data["faces"] = data["faces"]
        else:
            extra_data = {"original": extra_data, "faces": data["faces"]}

    return Observation(
        source=data["source"],
        frame_id=data["frame_id"],
        t_ns=data["t_ns"],
        signals=data.get("signals", {}),
        data=extra_data,
        metadata=data.get("metadata", {}),
        timing=data.get("timing"),
    )


def _serialize_value_safe(value: Any) -> Any:
    """Recursively serialize a value for JSON transmission.

    Args:
        value: Any value to serialize.

    Returns:
        JSON-serializable representation.
    """
    import json

    if value is None:
        return None

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    # Handle dataclasses
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: _serialize_value_safe(getattr(value, k))
            for k in value.__dataclass_fields__
        }

    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        return [_serialize_value_safe(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {k: _serialize_value_safe(v) for k, v in value.items()}

    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return repr(value)

    return str(value)


def _serialize_observation_for_deps(obs: Optional[Observation]) -> Optional[Dict[str, Any]]:
    """Serialize an Observation for deps transmission via ZMQ.

    Args:
        obs: Observation to serialize.

    Returns:
        JSON-serializable dict, or None.
    """
    if obs is None:
        return None

    result: Dict[str, Any] = {
        "source": obs.source,
        "frame_id": obs.frame_id,
        "t_ns": obs.t_ns,
        "signals": obs.signals,
        "metadata": obs.metadata,
        "timing": obs.timing,
    }
    if obs.data is not None:
        result["data"] = _serialize_value_safe(obs.data)
    return result


class VenvWorker(BaseWorker):
    """Worker that runs extraction in a separate venv subprocess.

    Uses ZMQ for bidirectional communication:
    - Main -> Subprocess: Frame (REQ)
    - Subprocess -> Main: Observation (REP)

    This provides full dependency isolation by running the extractor
    in a different Python virtual environment with its own set of
    installed packages.

    Requirements:
    - pyzmq must be installed in both the main process and the target venv
    - The target venv must have visualpath and the extractor's dependencies

    Example:
        >>> worker = VenvWorker(
        ...     extractor=None,  # Will be loaded in subprocess
        ...     venv_path="/path/to/venv-face",
        ...     extractor_name="face",
        ... )
        >>> worker.start()
        >>> result = worker.process(frame)
        >>> worker.stop()
    """

    def __init__(
        self,
        extractor: Optional[Module],
        venv_path: str,
        extractor_name: Optional[str] = None,
        queue_size: int = 10,
        timeout_sec: float = 30.0,
        handshake_timeout_sec: float = 30.0,
        jpeg_quality: int = 95,
    ):
        """Initialize the venv worker.

        Args:
            extractor: Optional extractor instance (for backwards compatibility).
                       If provided and extractor_name is None, falls back to inline.
            venv_path: Path to the virtual environment.
            extractor_name: Entry point name of the extractor to load in subprocess.
            queue_size: Maximum pending frames in queue (reserved for future use).
            timeout_sec: Timeout for processing requests in seconds.
            handshake_timeout_sec: Timeout for initial handshake in seconds.
            jpeg_quality: JPEG quality for frame compression (0-100).
        """
        self._extractor = extractor
        self._venv_path = venv_path
        self._extractor_name = extractor_name or (extractor.name if extractor else None)
        self._queue_size = queue_size
        self._timeout_sec = timeout_sec
        self._handshake_timeout_sec = handshake_timeout_sec
        self._jpeg_quality = jpeg_quality

        self._process: Optional[subprocess.Popen] = None
        self._zmq_context: Optional[Any] = None  # zmq.Context
        self._socket: Optional[Any] = None  # zmq.Socket
        self._ipc_address: str = ""
        self._ipc_file: Optional[str] = None
        self._running = False
        self._use_zmq = False

        # Fall back to inline if ZMQ not available or no extractor_name
        self._inline: Optional[InlineWorker] = None

    def _should_use_zmq(self) -> bool:
        """Determine if ZMQ should be used."""
        if not _check_zmq_available():
            logger.warning("pyzmq not available, falling back to inline execution")
            return False

        if not self._extractor_name:
            logger.warning("No extractor_name provided, falling back to inline execution")
            return False

        # Check if venv Python exists
        venv_python = os.path.join(self._venv_path, "bin", "python")
        if not os.path.isfile(venv_python):
            logger.warning(
                f"Venv Python not found at {venv_python}, falling back to inline execution"
            )
            return False

        return True

    def start(self) -> None:
        """Start the worker subprocess and establish ZMQ connection."""
        if self._running:
            return

        self._use_zmq = self._should_use_zmq()

        if not self._use_zmq:
            # Fall back to inline execution
            if self._extractor is None:
                raise ValueError(
                    "Cannot fall back to inline execution without an extractor instance"
                )
            self._inline = InlineWorker(self._extractor)
            self._inline.start()
            self._running = True
            return

        import zmq

        # Create ZMQ context and socket
        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self._timeout_sec * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self._timeout_sec * 1000))

        # Generate unique IPC address
        self._ipc_file = tempfile.mktemp(
            prefix=f"visualpath-worker-{os.getpid()}-",
            suffix=".sock"
        )
        self._ipc_address = f"ipc://{self._ipc_file}"

        # Start subprocess
        venv_python = os.path.join(self._venv_path, "bin", "python")
        cmd = [
            venv_python,
            "-m", "visualpath.process.worker",
            "--extractor", self._extractor_name,
            "--ipc-address", self._ipc_address,
        ]

        logger.info(f"Starting worker subprocess: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            self._cleanup_zmq()
            raise RuntimeError(f"Failed to start worker subprocess: {e}") from e

        # Give subprocess time to start and bind
        time.sleep(0.1)

        # Connect to the subprocess
        try:
            self._socket.connect(self._ipc_address)
        except zmq.ZMQError as e:
            self._terminate_process()
            self._cleanup_zmq()
            raise RuntimeError(f"Failed to connect to worker: {e}") from e

        # Perform handshake with timeout
        try:
            # Temporarily set shorter timeout for handshake
            self._socket.setsockopt(zmq.RCVTIMEO, int(self._handshake_timeout_sec * 1000))

            self._socket.send_json({"type": "ping"})
            response = self._socket.recv_json()

            if response.get("type") != "pong":
                raise RuntimeError(f"Unexpected handshake response: {response}")

            logger.info(f"Worker handshake successful, extractor: {response.get('extractor')}")

            # Restore normal timeout
            self._socket.setsockopt(zmq.RCVTIMEO, int(self._timeout_sec * 1000))

        except zmq.Again:
            self._terminate_process()
            self._cleanup_zmq()
            raise RuntimeError(
                f"Worker handshake timed out after {self._handshake_timeout_sec}s"
            )
        except zmq.ZMQError as e:
            self._terminate_process()
            self._cleanup_zmq()
            raise RuntimeError(f"Worker handshake failed: {e}") from e

        self._running = True

    def stop(self) -> None:
        """Stop the worker subprocess and clean up resources."""
        if not self._running:
            return

        if self._inline is not None:
            self._inline.stop()
            self._inline = None
            self._running = False
            return

        # Send shutdown signal
        if self._socket is not None:
            try:
                import zmq
                self._socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout for shutdown
                self._socket.send_json({"type": "shutdown"})
                self._socket.recv_json()
            except Exception as e:
                logger.warning(f"Error during shutdown signal: {e}")

        self._terminate_process()
        self._cleanup_zmq()
        self._running = False

    def _terminate_process(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Worker process did not terminate, killing")
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning(f"Error terminating worker process: {e}")
            finally:
                self._process = None

    def _cleanup_zmq(self) -> None:
        """Clean up ZMQ resources."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception:
                pass
            self._zmq_context = None

        # Remove IPC socket file
        if self._ipc_file and os.path.exists(self._ipc_file):
            try:
                os.unlink(self._ipc_file)
            except Exception:
                pass
            self._ipc_file = None

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Send frame to subprocess and receive observation.

        Args:
            frame: Frame to process.
            deps: Optional dict of observations from dependent extractors.

        Returns:
            WorkerResult with observation or error.
        """
        start_time = time.perf_counter()

        if not self._running:
            return WorkerResult(
                observation=None,
                error="Worker not running",
                timing_ms=0.0,
            )

        if self._inline is not None:
            return self._inline.process(frame, deps)

        import zmq

        try:
            # Serialize and send frame
            frame_data = _serialize_frame(frame, self._jpeg_quality)
            message: Dict[str, Any] = {
                "type": "extract",
                "frame": frame_data,
            }
            # Include deps if provided
            if deps:
                message["deps"] = {
                    name: _serialize_observation_for_deps(obs)
                    for name, obs in deps.items()
                }
            self._socket.send_json(message)

            # Receive response
            response = self._socket.recv_json()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if "error" in response:
                return WorkerResult(
                    observation=None,
                    error=response["error"],
                    timing_ms=elapsed_ms,
                )

            observation = _deserialize_observation(response.get("observation"))
            return WorkerResult(
                observation=observation,
                timing_ms=elapsed_ms,
            )

        except zmq.Again:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return WorkerResult(
                observation=None,
                error=f"Worker timeout after {self._timeout_sec}s",
                timing_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return WorkerResult(
                observation=None,
                error=str(e),
                timing_ms=elapsed_ms,
            )

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running

    def get_subprocess_output(self) -> tuple[str, str]:
        """Get stdout and stderr from the subprocess.

        Useful for debugging when the worker fails.

        Returns:
            Tuple of (stdout, stderr) strings.
        """
        if self._process is None:
            return "", ""

        stdout = ""
        stderr = ""

        if self._process.stdout:
            try:
                stdout = self._process.stdout.read()
            except Exception:
                pass

        if self._process.stderr:
            try:
                stderr = self._process.stderr.read()
            except Exception:
                pass

        return stdout or "", stderr or ""


class ProcessWorker(BaseWorker):
    """Worker that runs extraction in a separate process (same venv).

    This is a convenience wrapper around VenvWorker that uses the current
    Python environment. Provides process-level isolation without dependency
    isolation.

    Requirements:
    - pyzmq must be installed

    Example:
        >>> worker = ProcessWorker(
        ...     extractor=None,
        ...     extractor_name="face",
        ... )
        >>> worker.start()
        >>> result = worker.process(frame)
        >>> worker.stop()
    """

    def __init__(
        self,
        extractor: Optional[Module] = None,
        extractor_name: Optional[str] = None,
        queue_size: int = 10,
        timeout_sec: float = 30.0,
    ):
        """Initialize the process worker.

        Args:
            extractor: Optional extractor instance (for backwards compatibility).
            extractor_name: Entry point name of the extractor.
            queue_size: Maximum pending frames in queue.
            timeout_sec: Timeout for processing requests in seconds.
        """
        # Get the current venv path
        venv_path = os.path.dirname(os.path.dirname(sys.executable))

        self._delegate = VenvWorker(
            extractor=extractor,
            venv_path=venv_path,
            extractor_name=extractor_name,
            queue_size=queue_size,
            timeout_sec=timeout_sec,
        )

    def start(self) -> None:
        """Start the worker process."""
        self._delegate.start()

    def stop(self) -> None:
        """Stop the worker process."""
        self._delegate.stop()

    def process(self, frame: "Frame", deps: Optional[Dict[str, Observation]] = None) -> WorkerResult:
        """Process a frame in the worker process."""
        return self._delegate.process(frame, deps)

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._delegate.is_running


class WorkerLauncher:
    """Factory for creating workers based on isolation level.

    Example:
        >>> # Simple usage with extractor instance
        >>> launcher = WorkerLauncher.create(
        ...     level=IsolationLevel.THREAD,
        ...     extractor=my_extractor,
        ... )
        >>> with launcher:
        ...     result = launcher.process(frame)

        >>> # VenvWorker with subprocess
        >>> launcher = WorkerLauncher.create(
        ...     level=IsolationLevel.VENV,
        ...     extractor=None,
        ...     venv_path="/path/to/venv",
        ...     extractor_name="face",
        ... )
    """

    @staticmethod
    def create(
        level: IsolationLevel,
        extractor: Optional[Module],
        venv_path: Optional[str] = None,
        extractor_name: Optional[str] = None,
        **kwargs,
    ) -> BaseWorker:
        """Create a worker for the specified isolation level.

        Args:
            level: The isolation level to use.
            extractor: The extractor to run. Can be None for PROCESS/VENV levels
                      if extractor_name is provided (loaded via entry points).
            venv_path: Path to venv (required for VENV level).
            extractor_name: Entry point name of the extractor. Required for
                           PROCESS/VENV levels when extractor is None.
            **kwargs: Additional arguments passed to the worker.

        Returns:
            A worker instance for the specified isolation level.

        Raises:
            ValueError: If required parameters are missing.
        """
        if level == IsolationLevel.INLINE:
            if extractor is None:
                raise ValueError("extractor is required for INLINE isolation level")
            return InlineWorker(extractor)

        elif level == IsolationLevel.THREAD:
            if extractor is None:
                raise ValueError("extractor is required for THREAD isolation level")
            return ThreadWorker(extractor, **kwargs)

        elif level == IsolationLevel.PROCESS:
            return ProcessWorker(
                extractor=extractor,
                extractor_name=extractor_name,
                **kwargs,
            )

        elif level == IsolationLevel.VENV:
            if venv_path is None:
                raise ValueError("venv_path is required for VENV isolation level")
            return VenvWorker(
                extractor=extractor,
                venv_path=venv_path,
                extractor_name=extractor_name,
                **kwargs,
            )

        elif level == IsolationLevel.CONTAINER:
            # Container isolation not yet implemented
            raise NotImplementedError("CONTAINER isolation level not yet implemented")

        else:
            raise ValueError(f"Unknown isolation level: {level}")
