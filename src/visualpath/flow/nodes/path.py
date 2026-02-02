"""Path node for extractor execution.

PathNode wraps the existing Path class, running extractors on frames
and adding observations to FlowData.
"""

from typing import List, Optional, TYPE_CHECKING

from visualpath.flow.node import FlowNode, FlowData
from visualpath.core.path import Path
from visualpath.core.extractor import BaseExtractor
from visualpath.core.fusion import BaseFusion

if TYPE_CHECKING:
    pass


class PathNode(FlowNode):
    """Node that wraps a Path for extractor execution.

    PathNode runs extractors on the frame in FlowData and adds
    the resulting observations. Optionally runs fusion to produce
    results.

    Example:
        >>> path = Path("human", extractors=[face_ext, pose_ext], fusion=highlight_fusion)
        >>> path_node = PathNode(path)
        >>> with path_node:
        ...     output = path_node.process(flow_data)
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        name: Optional[str] = None,
        extractors: Optional[List[BaseExtractor]] = None,
        fusion: Optional[BaseFusion] = None,
        run_fusion: bool = True,
    ):
        """Initialize the path node.

        Either provide an existing Path or specify components to create one.

        Args:
            path: Existing Path instance to wrap.
            name: Name for auto-created Path (required if path not provided).
            extractors: Extractors for auto-created Path.
            fusion: Fusion module for auto-created Path.
            run_fusion: Whether to run fusion and add results to FlowData.
        """
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

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @property
    def path(self) -> Path:
        """Get the wrapped Path."""
        return self._path

    def initialize(self) -> None:
        """Initialize the wrapped Path."""
        self._path.initialize()

    def cleanup(self) -> None:
        """Clean up the wrapped Path."""
        self._path.cleanup()

    def process(self, data: FlowData) -> List[FlowData]:
        """Run extractors on the frame and add observations.

        Args:
            data: Input FlowData with frame to process.

        Returns:
            Single-item list with FlowData updated with observations
            and optionally fusion results.
        """
        if data.frame is None:
            return [data]

        # Build external deps from existing observations
        external_deps = {obs.source: obs for obs in data.observations}

        # Extract observations with deps
        observations = self._path.extract_all(data.frame, external_deps)

        # Start with existing observations and add new ones
        all_observations = list(data.observations) + observations

        result_data = data.clone(
            observations=all_observations,
            path_id=self._name,  # Update path_id to this path's name
        )

        # Optionally run fusion
        if self._run_fusion and self._path.fusion is not None:
            results = []
            for obs in observations:
                result = self._path.fusion.update(obs)
                results.append(result)

            all_results = list(data.results) + results
            result_data = result_data.with_results(all_results)

        return [result_data]
