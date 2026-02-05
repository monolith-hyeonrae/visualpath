"""IPC process wrappers for distributed execution.

Provides process wrappers for running extractors and fusion modules
as independent processes with IPC communication (A-B*-C architecture).

Architecture:
    A (Ingest) ←─── TRIG messages ←─── C (Fusion)
       ↓                                    ↑
    FIFO/video stream              OBS messages from B*
       │                                    ↑
       └─→ [B1: Extractor1] ──OBS→ ┐
           [B2: Extractor2] ──OBS→ ├─→ Fusion ──TRIG→ Ingest
           [B3: Extractor3] ──OBS→ ┘

Example (ExtractorProcess):
    >>> from visualpath.process import ExtractorProcess, DefaultObservationMapper
    >>> from visualbase.ipc.factory import TransportFactory
    >>>
    >>> mapper = DefaultObservationMapper()
    >>> reader = TransportFactory.create_video_reader("fifo", "/tmp/vid.mjpg")
    >>> sender = TransportFactory.create_message_sender("uds", "/tmp/obs.sock")
    >>>
    >>> process = ExtractorProcess(
    ...     extractor=my_extractor,
    ...     observation_mapper=mapper,
    ...     video_reader=reader,
    ...     message_sender=sender,
    ... )
    >>> process.run()

Example (FusionProcess):
    >>> from visualpath.process import FusionProcess, DefaultObservationMapper
    >>>
    >>> process = FusionProcess(
    ...     fusion=my_fusion,
    ...     observation_mapper=mapper,
    ...     obs_receiver=obs_receiver,
    ...     trig_sender=trig_sender,
    ... )
    >>> process.run()
"""

import signal
import time
import logging
from typing import Optional, Callable, List, Dict, Any
import threading
from collections import defaultdict

from visualbase.ipc.interfaces import VideoReader, MessageSender, MessageReceiver
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc.messages import TRIGMessage
from visualbase import Frame

from visualpath.core.extractor import Observation
from visualpath.core.module import Module
from visualpath.core.module import Module
from visualpath.process.mapper import ObservationMapper, DefaultObservationMapper
from visualpath.observability import ObservabilityHub

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


# Time window for observation alignment (100ms)
ALIGNMENT_WINDOW_NS = 100_000_000


class ExtractorProcess:
    """Wrapper for running an extractor as an independent process.

    Reads frames from a VideoReader input, processes them with the extractor,
    and sends OBS messages to the fusion process via MessageSender.

    Supports two initialization modes:
    1. Interface-based: Pass VideoReader and MessageSender instances directly
    2. Legacy path-based: Pass input_fifo and obs_socket paths (auto-creates FIFO/UDS)

    The observation_mapper is used to convert Observation objects to/from
    wire format. This allows domain-specific serialization to be injected.

    Args:
        extractor: The extractor instance to use.
        observation_mapper: Mapper for Observation ↔ Message conversion.
        video_reader: VideoReader instance for receiving frames.
        message_sender: MessageSender instance for sending OBS messages.
        input_fifo: (Legacy) Path to the FIFO for receiving frames.
        obs_socket: (Legacy) Path to the UDS socket for sending OBS messages.
        video_transport: Transport type for video ("fifo", "zmq"). Default: "fifo".
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        reconnect: Whether to reconnect on reader disconnect.
        on_frame: Optional callback for each processed frame.
        observability_hub: Optional custom observability hub (uses global if None).
    """

    def __init__(
        self,
        extractor: Module,
        observation_mapper: Optional[ObservationMapper] = None,
        video_reader: Optional[VideoReader] = None,
        message_sender: Optional[MessageSender] = None,
        input_fifo: Optional[str] = None,
        obs_socket: Optional[str] = None,
        video_transport: str = "fifo",
        message_transport: str = "uds",
        reconnect: bool = True,
        on_frame: Optional[Callable[[Frame, Observation], None]] = None,
        observability_hub: Optional[ObservabilityHub] = None,
    ):
        self._extractor = extractor
        self._mapper = observation_mapper or DefaultObservationMapper()
        self._reconnect = reconnect
        self._on_frame = on_frame
        self._hub = observability_hub or _hub

        # Store transport config for reconnection
        self._video_transport = video_transport
        self._message_transport = message_transport
        self._input_path = input_fifo
        self._obs_path = obs_socket

        # Interface-based or legacy path-based initialization
        if video_reader is not None:
            self._reader: Optional[VideoReader] = video_reader
            self._reader_provided = True
        elif input_fifo is not None:
            self._reader = None  # Created in run()
            self._reader_provided = False
        else:
            raise ValueError("Either video_reader or input_fifo must be provided")

        if message_sender is not None:
            self._client: Optional[MessageSender] = message_sender
            self._client_provided = True
        elif obs_socket is not None:
            self._client = None  # Created in run()
            self._client_provided = False
        else:
            raise ValueError("Either message_sender or obs_socket must be provided")

        self._running = False
        self._shutdown = threading.Event()

        # Stats
        self._frames_processed = 0
        self._obs_sent = 0
        self._errors = 0
        self._start_time: Optional[float] = None

        # Observability tracking
        self._dropped_frames: List[int] = []
        self._last_frame_id: Optional[int] = None

    def run(self) -> None:
        """Run the extractor process main loop.

        This method blocks until stop() is called or the process is
        interrupted.
        """
        # Setup signal handlers (only in main thread)
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.monotonic()

        # Initialize extractor
        logger.info(f"Initializing extractor: {self._extractor.name}")
        self._extractor.initialize()

        # Create message sender if not provided
        if self._client is None and self._obs_path is not None:
            self._client = TransportFactory.create_message_sender(
                self._message_transport, self._obs_path
            )

        # Connect to OBS socket
        if self._client is None or not self._client.connect():
            logger.error(f"Failed to connect to OBS socket: {self._obs_path}")
            return

        try:
            while self._running and not self._shutdown.is_set():
                self._run_once()

                if not self._running:
                    break

                if self._reconnect:
                    logger.info("Reader disconnected, waiting to reconnect...")
                    time.sleep(1.0)
                else:
                    break

        finally:
            self._cleanup()

    def _run_once(self) -> None:
        """Run one session (until reader disconnects)."""
        # Create reader if not provided (legacy path-based mode)
        if not self._reader_provided:
            if self._input_path is None:
                logger.error("No input path configured")
                return
            self._reader = TransportFactory.create_video_reader(
                self._video_transport, self._input_path
            )

        if self._reader is None:
            logger.error("No video reader available")
            return

        if not self._reader.open():
            logger.warning(f"Failed to open reader: {self._input_path}")
            return

        logger.info(f"Connected to reader: {self._input_path or 'provided reader'}")

        try:
            for frame in self._reader:
                if not self._running or self._shutdown.is_set():
                    break

                self._process_frame(frame)

        except Exception as e:
            logger.error(f"Error in extractor loop: {e}")
            self._errors += 1

        finally:
            if self._reader:
                self._reader.close()
                # Only clear reader if we created it (not provided)
                if not self._reader_provided:
                    self._reader = None

    def _process_frame(self, frame: Frame) -> None:
        """Process a single frame."""
        start_ns = time.perf_counter_ns() if self._hub.enabled else 0

        try:
            # Check for dropped frames
            if self._hub.enabled and self._last_frame_id is not None:
                expected_frame_id = self._last_frame_id + 1
                if frame.frame_id > expected_frame_id:
                    dropped = list(range(expected_frame_id, frame.frame_id))
                    self._dropped_frames.extend(dropped)
                    self._emit_frame_drop(dropped)
            self._last_frame_id = frame.frame_id

            # Extract features (support both Module.process() and Module.extract())
            if hasattr(self._extractor, 'process') and not hasattr(self._extractor, 'extract'):
                obs = self._extractor.process(frame)
            else:
                obs = self._extractor.extract(frame)
            self._frames_processed += 1

            if obs is not None:
                # Convert to OBS message using mapper and send
                message = self._mapper.to_message(obs)
                if message and self._client:
                    if self._client.send(message):
                        self._obs_sent += 1

                # Call frame callback if set
                if self._on_frame:
                    self._on_frame(frame, obs)

            # Emit timing record for process layer
            if self._hub.enabled:
                processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                self._emit_timing(frame.frame_id, processing_ms)

        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            self._errors += 1

    def _emit_frame_drop(self, dropped_frame_ids: List[int]) -> None:
        """Emit a frame drop record. Override in subclass for domain-specific records."""
        # Default implementation uses generic TraceRecord
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="frame_drop",
            frame_id=dropped_frame_ids[0] if dropped_frame_ids else 0,
            data={"dropped_frame_ids": dropped_frame_ids, "reason": "gap_in_sequence"},
        ))

    def _emit_timing(self, frame_id: int, processing_ms: float) -> None:
        """Emit a timing record. Override in subclass for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="timing",
            frame_id=frame_id,
            data={
                "component": f"process_{self._extractor.name}",
                "processing_ms": processing_ms,
                "threshold_ms": 100.0,
                "is_slow": processing_ms > 100.0,
            },
        ))

    def stop(self) -> None:
        """Stop the extractor process."""
        logger.info(f"Stopping extractor process: {self._extractor.name}")
        self._running = False
        self._shutdown.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._reader:
            self._reader.close()
            if not self._reader_provided:
                self._reader = None

        if self._client:
            self._client.disconnect()
            if not self._client_provided:
                self._client = None

        self._extractor.cleanup()

        # Log stats
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        fps = self._frames_processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Extractor '{self._extractor.name}' stopped: "
            f"{self._frames_processed} frames, {self._obs_sent} obs, "
            f"{fps:.1f} fps, {self._errors} errors"
        )

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "frames_processed": self._frames_processed,
            "obs_sent": self._obs_sent,
            "errors": self._errors,
            "elapsed_sec": elapsed,
            "fps": self._frames_processed / elapsed if elapsed > 0 else 0,
        }


class FusionProcess:
    """Wrapper for running fusion as an independent process.

    Receives OBS messages from extractors via MessageReceiver, converts them to
    Observations, runs the fusion engine, and sends TRIG messages to
    the ingest process via MessageSender.

    Supports two initialization modes:
    1. Interface-based: Pass MessageReceiver and MessageSender instances directly
    2. Legacy path-based: Pass obs_socket and trig_socket paths (auto-creates UDS)

    The observation_mapper is used to convert messages to Observation objects.

    Args:
        fusion: The fusion engine instance.
        observation_mapper: Mapper for Message → Observation conversion.
        obs_receiver: MessageReceiver instance for receiving OBS messages.
        trig_sender: MessageSender instance for sending TRIG messages.
        obs_socket: (Legacy) Path to the UDS socket for receiving OBS messages.
        trig_socket: (Legacy) Path to the UDS socket for sending TRIG messages.
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        alignment_window_ns: Time window for observation alignment.
        on_trigger: Optional callback for each trigger.
        observability_hub: Optional custom observability hub (uses global if None).
    """

    def __init__(
        self,
        fusion: Module,
        observation_mapper: Optional[ObservationMapper] = None,
        obs_receiver: Optional[MessageReceiver] = None,
        trig_sender: Optional[MessageSender] = None,
        obs_socket: Optional[str] = None,
        trig_socket: Optional[str] = None,
        message_transport: str = "uds",
        alignment_window_ns: int = ALIGNMENT_WINDOW_NS,
        on_trigger: Optional[Callable[[Observation], None]] = None,
        observability_hub: Optional[ObservabilityHub] = None,
    ):
        self._fusion = fusion
        self._mapper = observation_mapper or DefaultObservationMapper()
        self._alignment_window_ns = alignment_window_ns
        self._on_trigger = on_trigger
        self._hub = observability_hub or _hub

        # Store transport config
        self._message_transport = message_transport
        self._obs_path = obs_socket
        self._trig_path = trig_socket

        # Interface-based or legacy path-based initialization
        if obs_receiver is not None:
            self._obs_server: Optional[MessageReceiver] = obs_receiver
            self._obs_server_provided = True
        elif obs_socket is not None:
            self._obs_server = None  # Created in run()
            self._obs_server_provided = False
        else:
            raise ValueError("Either obs_receiver or obs_socket must be provided")

        if trig_sender is not None:
            self._trig_client: Optional[MessageSender] = trig_sender
            self._trig_client_provided = True
        elif trig_socket is not None:
            self._trig_client = None  # Created in run()
            self._trig_client_provided = False
        else:
            raise ValueError("Either trig_sender or trig_socket must be provided")

        self._running = False
        self._shutdown = threading.Event()

        # Observation buffer for alignment (keyed by frame_id)
        self._obs_buffer: Dict[int, Dict[str, str]] = defaultdict(dict)  # raw messages
        self._frame_timestamps: Dict[int, int] = {}  # frame_id -> t_ns

        # Stats
        self._obs_received = 0
        self._triggers_sent = 0
        self._errors = 0
        self._start_time: Optional[float] = None

    def run(self) -> None:
        """Run the fusion process main loop.

        This method blocks until stop() is called or the process is
        interrupted.
        """
        # Setup signal handlers (only in main thread)
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.monotonic()

        # Create OBS receiver if not provided
        if self._obs_server is None and self._obs_path is not None:
            self._obs_server = TransportFactory.create_message_receiver(
                self._message_transport, self._obs_path
            )

        # Start OBS server
        if self._obs_server is None:
            logger.error("No OBS receiver available")
            return
        self._obs_server.start()

        # Create TRIG sender if not provided
        if self._trig_client is None and self._trig_path is not None:
            self._trig_client = TransportFactory.create_message_sender(
                self._message_transport, self._trig_path
            )

        # Connect to TRIG socket
        if self._trig_client is None or not self._trig_client.connect():
            logger.error(f"Failed to connect to TRIG socket: {self._trig_path}")
            return

        logger.info("Fusion process started")
        logger.info(f"  OBS receiver: {self._obs_path or 'provided'}")
        logger.info(f"  TRIG sender: {self._trig_path or 'provided'}")

        try:
            while self._running and not self._shutdown.is_set():
                self._process_loop()

        finally:
            self._cleanup()

    def _process_loop(self) -> None:
        """Main processing loop iteration."""
        # Receive all pending OBS messages
        messages = self._obs_server.recv_all(max_messages=100)

        for msg in messages:
            self._handle_obs_message(msg)

        # Process aligned observations
        self._process_aligned_observations()

        # Small sleep to prevent busy-waiting
        if not messages:
            time.sleep(0.001)  # 1ms

    def _handle_obs_message(self, msg: str) -> None:
        """Handle a received OBS message."""
        # Try to parse basic frame info from message for buffering
        # We store raw messages and parse them later with the mapper
        self._obs_received += 1

        # Extract frame_id and source from message for buffering
        # This is a simple heuristic that works with common message formats
        frame_id, src, t_ns = self._parse_message_header(msg)
        if frame_id is None:
            logger.warning(f"Failed to parse message header: {msg[:100]}")
            return

        # Store in buffer by frame_id and source
        self._obs_buffer[frame_id][src] = msg
        self._frame_timestamps[frame_id] = t_ns

    def _parse_message_header(self, msg: str) -> tuple:
        """Parse frame_id, source, and t_ns from message.

        Override this method for custom message formats.

        Returns:
            Tuple of (frame_id, source, t_ns) or (None, None, 0) on error.
        """
        # Default implementation handles common formats:
        # 1. OBS format: "OBS src=face frame=1234 t_ns=..."
        # 2. JSON format: {"source": "face", "frame_id": 1234, "t_ns": ...}

        if msg.startswith("OBS "):
            # Parse OBS format
            parts = msg.split()
            data = {}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    data[key] = value
            try:
                return (
                    int(data.get("frame", 0)),
                    data.get("src", "unknown"),
                    int(data.get("t_ns", 0)),
                )
            except (ValueError, KeyError):
                return (None, None, 0)

        elif msg.startswith("{"):
            # Try JSON format
            import json
            try:
                data = json.loads(msg)
                return (
                    data.get("frame_id", 0),
                    data.get("source", "unknown"),
                    data.get("t_ns", 0),
                )
            except json.JSONDecodeError:
                return (None, None, 0)

        return (None, None, 0)

    def _process_aligned_observations(self) -> None:
        """Process observations that have been aligned by frame_id."""
        if not self._obs_buffer:
            return

        # Get oldest frame_id
        oldest_frame = min(self._obs_buffer.keys())

        # Current time estimate (most recent observation)
        current_t_ns = max(self._frame_timestamps.values()) if self._frame_timestamps else 0

        # Process observations older than alignment window
        frames_to_process = []
        for frame_id in list(self._obs_buffer.keys()):
            t_ns = self._frame_timestamps.get(frame_id, 0)
            delay_ns = current_t_ns - t_ns
            if delay_ns > self._alignment_window_ns:
                frames_to_process.append(frame_id)

                # Emit sync delay record if significant delay
                if self._hub.enabled and delay_ns > self._alignment_window_ns * 1.5:
                    obs_sources = list(self._obs_buffer.get(frame_id, {}).keys())
                    self._emit_sync_delay(frame_id, delay_ns, obs_sources)

        # Process in order
        for frame_id in sorted(frames_to_process):
            self._process_frame_observations(frame_id)
            del self._obs_buffer[frame_id]
            if frame_id in self._frame_timestamps:
                del self._frame_timestamps[frame_id]

    def _emit_sync_delay(
        self, frame_id: int, delay_ns: int, obs_sources: List[str]
    ) -> None:
        """Emit a sync delay record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="sync_delay",
            frame_id=frame_id,
            data={
                "expected_ns": self._alignment_window_ns,
                "actual_ns": delay_ns,
                "delay_ms": (delay_ns - self._alignment_window_ns) / 1_000_000,
                "sources": obs_sources,
            },
        ))

    def _process_frame_observations(self, frame_id: int) -> None:
        """Process all observations for a single frame."""
        start_ns = time.perf_counter_ns() if self._hub.enabled else 0

        obs_dict = self._obs_buffer.get(frame_id, {})
        if not obs_dict:
            return

        # Convert OBS messages to Observations using mapper and feed to fusion
        for src, msg in obs_dict.items():
            observation = self._mapper.from_message(msg)
            if observation:
                try:
                    result = self._fusion.update(observation)
                    if result.should_trigger and result.trigger:
                        self._send_trigger(result)

                except Exception as e:
                    logger.error(f"Fusion error: {e}")
                    self._errors += 1

        # Emit timing record for fusion process
        if self._hub.enabled:
            processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._emit_fusion_timing(frame_id, processing_ms)

    def _emit_fusion_timing(self, frame_id: int, processing_ms: float) -> None:
        """Emit fusion timing record. Override for domain-specific records."""
        from visualpath.observability.records import TraceRecord
        self._hub.emit(TraceRecord(
            record_type="timing",
            frame_id=frame_id,
            data={
                "component": "fusion_process",
                "processing_ms": processing_ms,
                "queue_depth": len(self._obs_buffer),
                "threshold_ms": 50.0,
                "is_slow": processing_ms > 50.0,
            },
        ))

    def _send_trigger(self, result: Observation) -> None:
        """Send a TRIG message.

        Args:
            result: Observation with trigger info in signals/metadata.
        """
        trigger = result.trigger  # Uses property that checks metadata
        if not trigger:
            return

        # Get score and reason from Observation signals
        score = result.trigger_score
        reason = result.trigger_reason

        trig_msg = TRIGMessage(
            label=trigger.label or "HIGHLIGHT",
            t_start_ns=trigger.clip_start_ns,
            t_end_ns=trigger.clip_end_ns,
            faces=len(result.metadata.get("faces", [])),
            score=score,
            reason=reason,
        )

        if self._trig_client and self._trig_client.send(trig_msg.to_message()):
            self._triggers_sent += 1
            logger.info(
                f"Trigger sent: {trig_msg.label} score={trig_msg.score:.2f} "
                f"reason={trig_msg.reason}"
            )

            # Call trigger callback if set
            if self._on_trigger:
                self._on_trigger(result)

    def stop(self) -> None:
        """Stop the fusion process."""
        logger.info("Stopping fusion process")
        self._running = False
        self._shutdown.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._obs_server:
            self._obs_server.stop()
            if not self._obs_server_provided:
                self._obs_server = None

        if self._trig_client:
            self._trig_client.disconnect()
            if not self._trig_client_provided:
                self._trig_client = None

        # Log stats
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            f"Fusion process stopped: "
            f"{self._obs_received} obs, {self._triggers_sent} triggers, "
            f"{self._errors} errors in {elapsed:.1f}s"
        )

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "obs_received": self._obs_received,
            "triggers_sent": self._triggers_sent,
            "errors": self._errors,
            "elapsed_sec": elapsed,
            "buffer_frames": len(self._obs_buffer),
        }
