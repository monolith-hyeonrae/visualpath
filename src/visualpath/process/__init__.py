"""Process wrappers for distributed execution.

This module provides wrappers for running extractors and fusion modules
as independent processes, with support for different isolation levels.

Components:
- Workers: Different isolation levels (inline, thread, process, venv)
- Mappers: Observation ↔ Message serialization
- IPC Processes: A-B*-C architecture (ExtractorProcess, FusionProcess)
- Orchestrator: Thread-parallel extractor execution
"""

from visualpath.process.launcher import (
    WorkerLauncher,
    BaseWorker,
    WorkerResult,
    InlineWorker,
    ThreadWorker,
    ProcessWorker,
    VenvWorker,
)
from visualpath.process.mapper import (
    ObservationMapper,
    DefaultObservationMapper,
    CompositeMapper,
)
from visualpath.process.ipc import (
    ExtractorProcess,
    FusionProcess,
    ALIGNMENT_WINDOW_NS,
)
from visualpath.process.orchestrator import ExtractorOrchestrator

__all__ = [
    # Workers
    "WorkerLauncher",
    "BaseWorker",
    "WorkerResult",
    "InlineWorker",
    "ThreadWorker",
    "ProcessWorker",
    "VenvWorker",
    # Mappers
    "ObservationMapper",
    "DefaultObservationMapper",
    "CompositeMapper",
    # IPC Processes
    "ExtractorProcess",
    "FusionProcess",
    "ALIGNMENT_WINDOW_NS",
    # Orchestrator
    "ExtractorOrchestrator",
]
