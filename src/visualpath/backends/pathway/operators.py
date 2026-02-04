"""Pathway operators for extractor execution.

This module provides operator wrappers that integrate visualpath's
extractors with Pathway's streaming operators.

Pure Python functions (create_extractor_udf, create_multi_extractor_udf)
can be used independently.

Pathway-specific functions (apply_extractors) require Pathway installed
and operate on pw.Table with PyObjectWrapper columns.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from visualbase import Frame
    from visualpath.core.extractor import BaseExtractor, Observation

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


@dataclass
class ExtractorResult:
    """Result from extractor execution in Pathway.

    Attributes:
        frame_id: Frame identifier.
        t_ns: Timestamp in nanoseconds.
        source: Extractor name.
        observation: The Observation, or None if filtered.
    """
    frame_id: int
    t_ns: int
    source: str
    observation: Optional["Observation"]


def create_extractor_udf(
    extractor: "BaseExtractor",
    deps: Optional[Dict[str, "Observation"]] = None,
):
    """Create a callable for a single extractor.

    Args:
        extractor: The extractor to wrap.
        deps: Optional pre-built deps for this extractor.

    Returns:
        A function that takes a Frame and returns list of ExtractorResult.
    """
    def extract_fn(frame: "Frame") -> List[ExtractorResult]:
        """Extract observations from a frame."""
        try:
            extractor_deps = None
            if extractor.depends and deps:
                extractor_deps = {
                    name: deps[name]
                    for name in extractor.depends
                    if name in deps
                }
                if "face_detect" in extractor.depends and "face_detect" not in extractor_deps and "face" in deps:
                    extractor_deps["face"] = deps["face"]
            try:
                observation = extractor.extract(frame, extractor_deps)
            except TypeError:
                observation = extractor.extract(frame)
            return [ExtractorResult(
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                source=extractor.name,
                observation=observation,
            )]
        except Exception:
            return []

    return extract_fn


def create_multi_extractor_udf(extractors: List["BaseExtractor"]):
    """Create a callable that runs multiple extractors on each frame.

    Extractors are run in order with dependency resolution: each
    extractor receives observations from its dependencies via the
    deps parameter.

    Args:
        extractors: List of extractors to run (order matters for deps).

    Returns:
        A function that takes a Frame and returns list of ExtractorResults.
    """
    def extract_all(frame: "Frame") -> List[ExtractorResult]:
        """Run all extractors on a frame with deps accumulation."""
        results = []
        deps: Dict[str, "Observation"] = {}
        for extractor in extractors:
            try:
                extractor_deps = None
                if extractor.depends:
                    extractor_deps = {
                        name: deps[name]
                        for name in extractor.depends
                        if name in deps
                    }
                    if "face_detect" in extractor.depends and "face_detect" not in extractor_deps and "face" in deps:
                        extractor_deps["face"] = deps["face"]
                try:
                    observation = extractor.extract(frame, extractor_deps)
                except TypeError:
                    observation = extractor.extract(frame)
                results.append(ExtractorResult(
                    frame_id=frame.frame_id,
                    t_ns=frame.t_src_ns,
                    source=extractor.name,
                    observation=observation,
                ))
                if observation is not None:
                    deps[extractor.name] = observation
            except Exception:
                pass
        return results

    return extract_all


if PATHWAY_AVAILABLE:
    def apply_extractors(
        frames_table: "pw.Table",
        extractors: List["BaseExtractor"],
    ) -> "pw.Table":
        """Apply extractors to a Pathway frames table.

        Runs all extractors on each frame via a @pw.udf, wrapping
        results in PyObjectWrapper for Pathway transport.

        Args:
            frames_table: Table with FrameSchema (frame column is PyObjectWrapper).
            extractors: List of extractors.

        Returns:
            Table with columns: frame_id, t_ns, observations (PyObjectWrapper).
        """
        raw_udf = create_multi_extractor_udf(extractors)

        @pw.udf
        def extract_all_udf(
            frame_wrapped: pw.PyObjectWrapper,
        ) -> pw.PyObjectWrapper:
            frame = frame_wrapped.value
            results = raw_udf(frame)
            return pw.PyObjectWrapper(results)

        return frames_table.select(
            frame_id=pw.this.frame_id,
            t_ns=pw.this.t_ns,
            observations=extract_all_udf(pw.this.frame),
        )


__all__ = [
    "ExtractorResult",
    "create_extractor_udf",
    "create_multi_extractor_udf",
]

if PATHWAY_AVAILABLE:
    __all__.append("apply_extractors")
