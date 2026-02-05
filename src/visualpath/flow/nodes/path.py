"""Path node for extractor execution.

PathNode declares extractors to run on frames.
Execution is handled by the interpreter.
"""

from typing import List, Optional

from visualpath.flow.node import FlowNode
from visualpath.flow.specs import ExtractSpec
from visualpath.core.path import Path
from visualpath.core.extractor import BaseExtractor
from visualpath.core.fusion import BaseFusion


class PathNode(FlowNode):
    """Node that declares extractor execution.

    Spec: ExtractSpec(extractors, fusion, parallel, run_fusion, join_window_ns)
    Backend: runs extractors on frame, optionally applies fusion.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        name: Optional[str] = None,
        extractors: Optional[List[BaseExtractor]] = None,
        fusion: Optional[BaseFusion] = None,
        run_fusion: bool = True,
        parallel: bool = False,
        join_window_ns: int = 100_000_000,
    ):
        if path is not None:
            self._path = path
            self._name = path.name
        elif name is not None:
            self._path = Path(
                name=name,
                extractors=extractors or [],
                fusion=fusion,
            )
            self._name = name
        else:
            raise ValueError("Either 'path' or 'name' must be provided")

        self._run_fusion = run_fusion
        self._parallel = parallel
        self._join_window_ns = join_window_ns

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        return self._path

    @property
    def spec(self) -> ExtractSpec:
        return ExtractSpec(
            extractors=tuple(self._path.extractors),
            fusion=self._path.fusion,
            parallel=self._parallel,
            run_fusion=self._run_fusion,
            join_window_ns=self._join_window_ns,
        )

    def initialize(self) -> None:
        self._path.initialize()

    def cleanup(self) -> None:
        self._path.cleanup()
