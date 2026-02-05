"""Path node for module execution.

PathNode declares modules to run on frames.
Execution is handled by the interpreter.

Supports two modes:
1. Unified modules API (preferred): PathNode(name="...", modules=[...])
2. Legacy extractors/fusion API: PathNode(name="...", extractors=[...], fusion=...)
"""

from typing import List, Optional, Union, TYPE_CHECKING

from visualpath.flow.node import FlowNode
from visualpath.flow.specs import ExtractSpec, ModuleSpec, NodeSpec

if TYPE_CHECKING:
    from visualpath.core.extractor import BaseExtractor
    from visualpath.core.fusion import BaseFusion
    from visualpath.core.module import Module
    from visualpath.core.path import Path


class PathNode(FlowNode):
    """Node that declares module execution.

    Preferred API (unified modules):
        PathNode(name="analysis", modules=[face_detector, smile_trigger])

    Legacy API (extractors + fusion):
        PathNode(name="analysis", extractors=[face_ext], fusion=smile_fusion)

    Spec types:
        - ModuleSpec when modules parameter is used
        - ExtractSpec when extractors/fusion parameters are used (legacy)

    Backend interprets the spec and runs modules on frames.
    """

    def __init__(
        self,
        path: Optional["Path"] = None,
        *,
        name: Optional[str] = None,
        # Unified modules API
        modules: Optional[List["Module"]] = None,
        # Legacy API
        extractors: Optional[List["BaseExtractor"]] = None,
        fusion: Optional["BaseFusion"] = None,
        run_fusion: bool = True,
        # Shared options
        parallel: bool = False,
        join_window_ns: int = 100_000_000,
    ):
        """Initialize a PathNode.

        Args:
            path: Existing Path instance (legacy).
            name: Name for this node.
            modules: List of unified modules (analyzers and triggers).
            extractors: List of extractors (legacy, use modules instead).
            fusion: Fusion module (legacy, use modules instead).
            run_fusion: Whether to run fusion (legacy).
            parallel: Whether independent modules can run in parallel.
            join_window_ns: Window for auto-joining parallel branches.

        Raises:
            ValueError: If neither 'path', 'name', nor 'modules' is provided.
        """
        # Determine mode: unified modules vs legacy extractors/fusion
        self._modules: Optional[tuple] = None
        self._path: Optional["Path"] = None

        if modules is not None:
            # Unified modules mode
            self._modules = tuple(modules)
            if name is None:
                # Auto-generate name from first module
                if modules:
                    name = f"path_{modules[0].name}"
                else:
                    name = "path_empty"
            self._name = name
        elif path is not None:
            # Legacy: existing Path
            self._path = path
            self._name = path.name
        elif name is not None:
            # Legacy: create Path from extractors/fusion
            from visualpath.core.path import Path
            self._path = Path(
                name=name,
                extractors=extractors or [],
                fusion=fusion,
            )
            self._name = name
        else:
            raise ValueError("Either 'modules', 'path', or 'name' must be provided")

        self._run_fusion = run_fusion
        self._parallel = parallel
        self._join_window_ns = join_window_ns

    @property
    def name(self) -> str:
        return self._name

    @property
    def modules(self) -> Optional[tuple]:
        """Get unified modules list.

        Returns:
            Tuple of modules if using unified API, None if using legacy API.
        """
        return self._modules

    @property
    def path(self) -> Optional["Path"]:
        """Get legacy Path instance.

        Returns:
            Path if using legacy API, None if using unified modules API.
        """
        return self._path

    @property
    def spec(self) -> NodeSpec:
        """Get spec for this node.

        Returns:
            ModuleSpec if using unified modules API.
            ExtractSpec if using legacy extractors/fusion API.
        """
        if self._modules is not None:
            # Unified modules mode
            return ModuleSpec(
                modules=self._modules,
                parallel=self._parallel,
                join_window_ns=self._join_window_ns,
            )
        else:
            # Legacy mode
            return ExtractSpec(
                extractors=tuple(self._path.extractors),
                fusion=self._path.fusion,
                parallel=self._parallel,
                run_fusion=self._run_fusion,
                join_window_ns=self._join_window_ns,
            )

    def initialize(self) -> None:
        """Initialize all modules."""
        if self._modules is not None:
            for module in self._modules:
                if hasattr(module, 'initialize'):
                    module.initialize()
        elif self._path is not None:
            self._path.initialize()

    def cleanup(self) -> None:
        """Cleanup all modules."""
        if self._modules is not None:
            for module in self._modules:
                if hasattr(module, 'cleanup'):
                    module.cleanup()
        elif self._path is not None:
            self._path.cleanup()
