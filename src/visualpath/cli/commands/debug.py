"""Debug command for testing pipelines with mock frames.

Usage:
    visualpath debug --frames 10 --sample 3 --debug
    visualpath debug --frames 5 --modules "analyzer,trigger" --debug
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class MockFrame:
    """Mock frame for CLI testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


def make_mock_frames(count: int, interval_ns: int = 100_000_000) -> List[MockFrame]:
    """Generate mock frames for testing."""
    return [
        MockFrame(
            frame_id=i,
            t_src_ns=i * interval_ns,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        for i in range(count)
    ]


class AnalyzerModule:
    """Unified Module that produces Observation."""

    depends: List[str] = []

    def __init__(self, name: str = "analyzer"):
        self._name = name
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None):
        from visualpath.core import Observation
        self._count += 1
        return Observation(
            source=self._name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": 0.5, "count": self._count},
        )

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def reset(self) -> None:
        self._count = 0


class TriggerModule:
    """Unified Module that produces FusionResult (trigger)."""

    def __init__(self, name: str = "trigger", threshold: float = 0.3, depends_on: str = None):
        self._name = name
        self._threshold = threshold
        self._count = 0
        self._depends_on = depends_on
        # depends will be set when we know the analyzer name
        self.depends = [depends_on] if depends_on else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trigger(self) -> bool:
        return True

    def process(self, frame, deps=None):
        from visualpath.core import FusionResult
        self._count += 1

        # Get observation from deps (try multiple keys)
        obs = None
        if deps:
            # Try specific dependency name first
            if self._depends_on and self._depends_on in deps:
                obs = deps[self._depends_on]
            else:
                # Fallback: use first observation in deps
                for v in deps.values():
                    if hasattr(v, 'signals'):  # It's an Observation
                        obs = v
                        break

        if not obs:
            return FusionResult(should_trigger=False)

        value = obs.signals.get("value", 0)

        if value > self._threshold:
            from visualbase import Trigger
            trigger = Trigger.point(
                event_time_ns=obs.t_ns,
                pre_sec=1.0,
                post_sec=1.0,
                label="cli_test",
                score=value,
            )
            return FusionResult(should_trigger=True, trigger=trigger, score=value)
        return FusionResult(should_trigger=False)

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def reset(self) -> None:
        self._count = 0


# Legacy aliases for backward compatibility
DummyExtractor = AnalyzerModule
DummyFusion = TriggerModule


def cmd_debug(
    frames: int = 5,
    sample: int = 1,
    extractor: str = "dummy",
    fusion: bool = False,
    debug: bool = False,
    backend: str = "simple",
) -> int:
    """Run debug pipeline with mock frames.

    Args:
        frames: Number of frames to process.
        sample: Sample every Nth frame.
        extractor: Extractor name (legacy) or module name.
        fusion: Enable trigger module.
        debug: Enable debug output.
        backend: Backend to use ('simple' or 'pathway').

    Returns:
        Exit code (0 for success).
    """
    from visualpath.flow import FlowGraphBuilder, GraphExecutor

    print("=" * 60)
    print("VisualPath Debug Pipeline")
    print("=" * 60)
    print(f"Frames: {frames}")
    print(f"Sample: every {sample}")
    print(f"Module: {extractor}")
    print(f"Trigger: {'enabled' if fusion else 'disabled'}")
    print(f"Debug: {debug}")
    print(f"Backend: {backend}")
    print("=" * 60)

    # Build modules
    analyzer = AnalyzerModule(extractor)
    # Set trigger dependency to actual analyzer name
    trigger = TriggerModule(depends_on=extractor) if fusion else None

    # Combine into unified modules list
    if trigger:
        modules = [analyzer, trigger]
    else:
        modules = [analyzer]

    # Build pipeline
    builder = FlowGraphBuilder().source("frames")

    if sample > 1:
        builder = builder.sample(every_nth=sample)

    # Use unified modules API
    builder = builder.path("main", modules=modules)

    graph = builder.build()

    # Generate frames
    mock_frames = make_mock_frames(frames)

    # Choose execution method
    if backend == "pathway":
        return _run_pathway(graph, mock_frames, analyzer, trigger, debug)
    else:
        return _run_simple(graph, mock_frames, analyzer, trigger, debug)


def _run_simple(graph, frames, ext, fusion_obj, debug: bool) -> int:
    """Run with SimpleBackend via GraphExecutor."""
    from visualpath.flow import GraphExecutor

    triggers_received = []

    def on_trigger(data):
        for result in data.results:
            if result.should_trigger:
                triggers_received.append(result)

    graph.on_trigger(on_trigger)

    executor = GraphExecutor(graph, debug=debug)

    print("\n[Processing frames...]")
    if not debug:
        print("(use --debug for detailed output)\n")

    with executor:
        for frame in frames:
            if debug:
                print(f"\n--- Frame {frame.frame_id} ---")
            executor.process(frame)

    _print_results(len(frames), ext, fusion_obj, triggers_received)
    return 0


def _run_pathway(graph, frames, ext, fusion_obj, debug: bool) -> int:
    """Run with PathwayBackend."""
    try:
        from visualpath.backends.pathway import PathwayBackend
    except ImportError:
        print("Error: Pathway not installed. Use 'pip install visualpath[pathway]'")
        return 1

    print("\n[Processing frames with Pathway...]")
    if debug:
        print("(Pathway debug output via ObservabilityHub)\n")

    # Configure observability for debug
    if debug:
        from visualpath.observability import ObservabilityHub, TraceLevel, MemorySink
        ObservabilityHub.reset_instance()
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.NORMAL, sinks=[sink])

    backend = PathwayBackend(autocommit_ms=10)
    result = backend.execute(iter(frames), graph)

    # Show stats
    print("\n[Pathway Stats]")
    stats = backend.get_stats()
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, dict):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val}")

    # Show debug records if enabled
    if debug:
        hub = ObservabilityHub.get_instance()
        print(f"\n[Trace Records]")
        for record in sink.get_records():
            print(f"  {record.record_type}: {record}")
        ObservabilityHub.reset_instance()

    _print_results(result.frame_count, ext, fusion_obj, result.triggers)
    return 0


def _print_results(frame_count, analyzer, trigger, triggers):
    """Print execution results."""
    print("\n" + "=" * 60)
    print("[Results]")
    print(f"  Frames processed: {frame_count}")
    print(f"  Analyzer calls: {analyzer._count}")
    if trigger:
        print(f"  Trigger calls: {trigger._count}")
        print(f"  Triggers fired: {len(triggers)}")
    print("=" * 60)
