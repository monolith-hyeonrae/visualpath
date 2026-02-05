"""Path abstraction for grouping extractors.

A Path represents a logical grouping of related extractors that share
common fusion logic. Examples:
- Human/When Path: face, pose, gesture extractors with highlight fusion
- Scene/What Path: object, OCR, depth extractors with scene fusion

Paths allow:
- Grouped configuration of extractors
- Independent isolation levels per group
- Parallel execution of paths with different fusion strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from visualpath.core.extractor import Observation
from visualpath.core.module import Module
from visualpath.core.isolation import IsolationLevel, IsolationConfig

if TYPE_CHECKING:
    from visualbase import Frame


@dataclass
class PathConfig:
    """Configuration for a single Path.

    Attributes:
        name: Unique name for this path.
        extractors: List of extractor names to include.
        fusion: Name of the fusion module to use.
        default_isolation: Default isolation level for extractors in this path.
        extractor_config: Per-extractor configuration overrides.
    """

    name: str
    extractors: List[str] = field(default_factory=list)
    fusion: Optional[str] = None
    default_isolation: IsolationLevel = IsolationLevel.INLINE
    extractor_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Path:
    """A group of extractors with shared fusion logic.

    A Path manages a set of extractors and their fusion module,
    handling initialization, extraction, and cleanup.

    Example:
        >>> path = Path(
        ...     name="human",
        ...     extractors=[face_extractor, pose_extractor],
        ...     fusion=highlight_fusion,
        ... )
        >>> with path:
        ...     for frame in video:
        ...         results = path.process(frame)
        ...         for result in results:
        ...             if result.should_trigger:
        ...                 handle_trigger(result)
    """

    def __init__(
        self,
        name: str,
        extractors: List[Module],
        fusion: Optional[Module] = None,
        isolation_config: Optional[IsolationConfig] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize a Path.

        Args:
            name: Unique name for this path.
            extractors: List of extractor instances.
            fusion: Optional fusion module for combining observations.
            isolation_config: Optional isolation configuration.
            parallel: Whether to run extractors in parallel.
            max_workers: Max thread workers for parallel execution.
        """
        self._name = name
        self._extractors = extractors
        self._fusion = fusion
        self._isolation_config = isolation_config or IsolationConfig()
        self._parallel = parallel
        self._max_workers = max_workers or len(extractors)

        self._initialized = False
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def name(self) -> str:
        """Get the path name."""
        return self._name

    @property
    def extractors(self) -> List[Module]:
        """Get the list of extractors."""
        return self._extractors

    @property
    def fusion(self) -> Optional[Module]:
        """Get the fusion module."""
        return self._fusion

    def initialize(self) -> None:
        """Initialize all extractors and fusion module."""
        if self._initialized:
            return

        # Validate dependencies before initialization
        self._validate_dependencies()

        for extractor in self._extractors:
            extractor.initialize()

        if self._parallel and len(self._extractors) > 1:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        self._initialized = True

    def _validate_dependencies(self) -> None:
        """Validate that all extractor dependencies are satisfied.

        Checks:
        - All depends are provided by earlier extractors or external deps
        - No circular dependencies within this path

        Raises:
            ValueError: If dependencies are not satisfied.
        """
        available: set[str] = set()

        for extractor in self._extractors:
            # Check if all depends are available
            depends = set(extractor.depends) if extractor.depends else set()
            missing = depends - available

            if missing:
                raise ValueError(
                    f"Extractor '{extractor.name}' depends on {missing}, "
                    f"but only {available or 'nothing'} is available. "
                    f"Reorder extractors or provide external_deps."
                )

            # This extractor's output is now available for subsequent extractors
            available.add(extractor.name)

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph for extractors in this path.

        Returns:
            Dict mapping extractor names to their dependencies.
        """
        return {
            ext.name: list(ext.depends) if ext.depends else []
            for ext in self._extractors
        }

    def cleanup(self) -> None:
        """Clean up all extractors and fusion module."""
        if not self._initialized:
            return

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        for extractor in self._extractors:
            extractor.cleanup()

        self._initialized = False

    def __enter__(self) -> "Path":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def extract_all(
        self,
        frame: "Frame",
        external_deps: Optional[Dict[str, Observation]] = None,
    ) -> List[Observation]:
        """Run all extractors on a frame.

        Args:
            frame: The input frame to process.
            external_deps: Optional observations from previous extractors
                          (e.g., from upstream PathNodes in FlowGraph).

        Returns:
            List of observations from all extractors.
        """
        if not self._initialized:
            raise RuntimeError("Path not initialized. Use context manager or call initialize().")

        # Always use sequential with deps for dependency resolution
        return self._extract_with_deps(frame, external_deps)

    def _extract_with_deps(
        self,
        frame: "Frame",
        external_deps: Optional[Dict[str, Observation]] = None,
    ) -> List[Observation]:
        """Extract features with dependency resolution.

        Extractors are run in order, with each extractor receiving
        observations from its dependencies.
        """
        # Build deps dict from external deps
        deps: Dict[str, Observation] = dict(external_deps) if external_deps else {}
        observations: List[Observation] = []

        for extractor in self._extractors:
            # Build deps for this extractor
            extractor_deps = None
            if extractor.depends:
                extractor_deps = {
                    name: deps[name]
                    for name in extractor.depends
                    if name in deps
                }

            # Extract (with backward compatibility)
            obs = self._call_extract(extractor, frame, extractor_deps)
            if obs is not None:
                observations.append(obs)
                # Add to deps for subsequent extractors
                deps[extractor.name] = obs

        return observations

    def _call_extract(
        self,
        extractor: Module,
        frame: "Frame",
        deps: Optional[Dict[str, Observation]],
    ) -> Optional[Observation]:
        """Call extractor.extract with backward compatibility.

        Handles extractors that don't accept the deps parameter.
        """
        try:
            return extractor.extract(frame, deps)
        except TypeError:
            # Fallback for extractors without deps parameter
            return extractor.extract(frame)

    def _extract_sequential(self, frame: "Frame") -> List[Observation]:
        """Extract features sequentially (legacy, no deps)."""
        return self._extract_with_deps(frame, None)

    def _extract_parallel(self, frame: "Frame") -> List[Observation]:
        """Extract features in parallel using threads.

        Note: Only extractors without dependencies can run in parallel.
        Extractors with dependencies fall back to sequential execution.
        """
        if self._executor is None:
            return self._extract_sequential(frame)

        # Check if any extractor has dependencies
        has_deps = any(ext.depends for ext in self._extractors)
        if has_deps:
            # Fall back to sequential for dependency resolution
            return self._extract_with_deps(frame, None)

        futures = {
            self._executor.submit(self._call_extract, extractor, frame, None): extractor
            for extractor in self._extractors
        }

        observations = []
        for future in as_completed(futures):
            try:
                obs = future.result()
                if obs is not None:
                    observations.append(obs)
            except Exception:
                # Log but continue with other extractors
                pass

        return observations

    def process(self, frame: "Frame") -> List[Observation]:
        """Process a frame through extractors and fusion.

        Args:
            frame: The input frame to process.

        Returns:
            List of Observations from fusion processing.
        """
        observations = self.extract_all(frame)

        if self._fusion is None:
            # No fusion - return empty results
            return []

        results = []
        for obs in observations:
            # Support both Module.update() and Module.process()
            if hasattr(self._fusion, 'update'):
                result = self._fusion.update(obs)
            else:
                result = self._fusion.process(frame, {obs.source: obs})
            results.append(result)

        return results


class PathOrchestrator:
    """Orchestrates multiple Paths.

    Runs multiple paths in parallel or sequentially, collecting
    results from all paths.

    Example:
        >>> orchestrator = PathOrchestrator([human_path, scene_path])
        >>> with orchestrator:
        ...     for frame in video:
        ...         all_results = orchestrator.process_all(frame)
        ...         for path_name, results in all_results.items():
        ...             for result in results:
        ...                 if result.should_trigger:
        ...                     handle_trigger(path_name, result)
    """

    def __init__(
        self,
        paths: List[Path],
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize the orchestrator.

        Args:
            paths: List of Path instances to orchestrate.
            parallel: Whether to run paths in parallel.
            max_workers: Max thread workers for parallel path execution.
        """
        self._paths = paths
        self._parallel = parallel
        self._max_workers = max_workers or len(paths)

        self._initialized = False
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def paths(self) -> List[Path]:
        """Get the list of paths."""
        return self._paths

    def initialize(self) -> None:
        """Initialize all paths."""
        if self._initialized:
            return

        for path in self._paths:
            path.initialize()

        if self._parallel and len(self._paths) > 1:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        self._initialized = True

    def cleanup(self) -> None:
        """Clean up all paths."""
        if not self._initialized:
            return

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        for path in self._paths:
            path.cleanup()

        self._initialized = False

    def __enter__(self) -> "PathOrchestrator":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def process_all(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process a frame through all paths.

        Args:
            frame: The input frame to process.

        Returns:
            Dict mapping path names to their Observations.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Use context manager or call initialize().")

        if self._parallel and self._executor and len(self._paths) > 1:
            return self._process_parallel(frame)
        else:
            return self._process_sequential(frame)

    def _process_sequential(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process paths sequentially."""
        results = {}
        for path in self._paths:
            results[path.name] = path.process(frame)
        return results

    def _process_parallel(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Process paths in parallel."""
        if self._executor is None:
            return self._process_sequential(frame)

        futures = {
            self._executor.submit(path.process, frame): path
            for path in self._paths
        }

        results = {}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path.name] = future.result()
            except Exception:
                results[path.name] = []

        return results

    def extract_all(self, frame: "Frame") -> Dict[str, List[Observation]]:
        """Extract observations from all paths without fusion.

        Args:
            frame: The input frame to process.

        Returns:
            Dict mapping path names to their observations.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized.")

        results = {}
        for path in self._paths:
            results[path.name] = path.extract_all(frame)
        return results
