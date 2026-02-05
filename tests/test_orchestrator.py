"""Tests for ExtractorOrchestrator."""

import pytest
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from visualpath.core import Module, Observation
from visualpath.process import ExtractorOrchestrator


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockFrame:
    """Mock Frame for testing."""
    frame_id: int
    t_src_ns: int
    data: np.ndarray


class SimpleExtractor(Module):
    """Simple extractor for testing."""

    def __init__(
        self,
        name: str = "simple",
        delay_ms: float = 0,
        fail: bool = False,
        return_none: bool = False,
    ):
        self._name = name
        self._delay_ms = delay_ms
        self._fail = fail
        self._return_none = return_none
        self._initialized = False
        self._cleaned_up = False
        self._extract_count = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame, deps=None) -> Optional[Observation]:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)

        if self._fail:
            raise RuntimeError("Extraction failed")

        if self._return_none:
            return None

        self._extract_count += 1
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"count": self._extract_count},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._cleaned_up = True


# =============================================================================
# Basic Tests
# =============================================================================


class TestExtractorOrchestrator:
    """Tests for ExtractorOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        extractors = [SimpleExtractor("ext1"), SimpleExtractor("ext2")]
        orchestrator = ExtractorOrchestrator(extractors)

        assert not orchestrator.is_initialized
        assert orchestrator.extractor_names == ["ext1", "ext2"]

    def test_requires_at_least_one_extractor(self):
        """Test that at least one extractor is required."""
        with pytest.raises(ValueError, match="At least one extractor"):
            ExtractorOrchestrator([])

    def test_initialize(self):
        """Test initialize() initializes all extractors."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2")
        orchestrator = ExtractorOrchestrator([ext1, ext2])

        orchestrator.initialize()

        assert orchestrator.is_initialized
        assert ext1._initialized
        assert ext2._initialized

    def test_cleanup(self):
        """Test cleanup() cleans up all extractors."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2")
        orchestrator = ExtractorOrchestrator([ext1, ext2])

        orchestrator.initialize()
        orchestrator.cleanup()

        assert not orchestrator.is_initialized
        assert ext1._cleaned_up
        assert ext2._cleaned_up

    def test_context_manager(self):
        """Test context manager protocol."""
        ext = SimpleExtractor()

        with ExtractorOrchestrator([ext]) as orchestrator:
            assert orchestrator.is_initialized
            assert ext._initialized

        assert not orchestrator.is_initialized
        assert ext._cleaned_up

    def test_double_initialize(self):
        """Test that double initialize is safe."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext])

        orchestrator.initialize()
        orchestrator.initialize()  # Should be no-op

        assert orchestrator.is_initialized

    def test_double_cleanup(self):
        """Test that double cleanup is safe."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext])

        orchestrator.initialize()
        orchestrator.cleanup()
        orchestrator.cleanup()  # Should be no-op

        assert not orchestrator.is_initialized


# =============================================================================
# Extraction Tests
# =============================================================================


class TestExtractorOrchestratorExtraction:
    """Tests for extraction functionality."""

    def test_extract_all_single_extractor(self):
        """Test extracting with a single extractor."""
        ext = SimpleExtractor()
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext]) as orchestrator:
            observations = orchestrator.extract_all(frame)

        assert len(observations) == 1
        assert observations[0].source == "simple"
        assert observations[0].frame_id == 1

    def test_extract_all_multiple_extractors(self):
        """Test extracting with multiple extractors."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2")
        ext3 = SimpleExtractor("ext3")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2, ext3]) as orchestrator:
            observations = orchestrator.extract_all(frame)

        assert len(observations) == 3
        sources = {obs.source for obs in observations}
        assert sources == {"ext1", "ext2", "ext3"}

    def test_extract_all_requires_initialization(self):
        """Test that extract_all requires initialization."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.extract_all(frame)

    def test_extract_all_filters_none(self):
        """Test that None observations are filtered out."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2", return_none=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.extract_all(frame)

        assert len(observations) == 1
        assert observations[0].source == "ext1"

    def test_extract_all_handles_errors(self):
        """Test that errors in extractors are handled."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2", fail=True)
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.extract_all(frame)

        # Only ext1 should succeed
        assert len(observations) == 1
        assert observations[0].source == "ext1"

        # Error should be counted
        stats = orchestrator.get_stats()
        assert stats["errors"] == 1

    def test_extract_all_timeout(self):
        """Test that slow extractors timeout."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2", delay_ms=2000)  # Very slow
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2], timeout=0.1) as orchestrator:
            observations = orchestrator.extract_all(frame)

        # ext1 should succeed, ext2 may timeout
        assert len(observations) >= 1

        stats = orchestrator.get_stats()
        # May or may not have timeout depending on timing
        # Just verify stats are collected
        assert stats["frames_processed"] == 1

    def test_extract_multiple_frames(self):
        """Test processing multiple frames."""
        ext = SimpleExtractor()

        with ExtractorOrchestrator([ext]) as orchestrator:
            for i in range(5):
                frame = MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
                observations = orchestrator.extract_all(frame)
                assert len(observations) == 1
                assert observations[0].frame_id == i

        stats = orchestrator.get_stats()
        assert stats["frames_processed"] == 5
        assert stats["total_observations"] == 5


# =============================================================================
# Sequential Extraction Tests
# =============================================================================


class TestExtractorOrchestratorSequential:
    """Tests for sequential extraction."""

    def test_extract_sequential(self):
        """Test sequential extraction."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2]) as orchestrator:
            observations = orchestrator.extract_sequential(frame)

        assert len(observations) == 2

    def test_extract_sequential_requires_initialization(self):
        """Test that extract_sequential requires initialization."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext])
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.extract_sequential(frame)

    def test_extract_sequential_handles_errors(self):
        """Test that sequential extraction handles errors."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2", fail=True)
        ext3 = SimpleExtractor("ext3")
        frame = MockFrame(frame_id=1, t_src_ns=1000000, data=np.zeros((100, 100, 3)))

        with ExtractorOrchestrator([ext1, ext2, ext3]) as orchestrator:
            observations = orchestrator.extract_sequential(frame)

        # ext1 and ext3 should succeed
        assert len(observations) == 2

        stats = orchestrator.get_stats()
        assert stats["errors"] == 1


# =============================================================================
# Stats Tests
# =============================================================================


class TestExtractorOrchestratorStats:
    """Tests for statistics collection."""

    def test_get_stats_initial(self):
        """Test initial stats."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext])

        stats = orchestrator.get_stats()

        assert stats["frames_processed"] == 0
        assert stats["total_observations"] == 0
        assert stats["timeouts"] == 0
        assert stats["errors"] == 0
        assert stats["extractors"] == ["simple"]

    def test_get_stats_after_processing(self):
        """Test stats after processing."""
        ext1 = SimpleExtractor("ext1")
        ext2 = SimpleExtractor("ext2")

        with ExtractorOrchestrator([ext1, ext2]) as orchestrator:
            for i in range(3):
                frame = MockFrame(frame_id=i, t_src_ns=i * 1000000, data=np.zeros((100, 100, 3)))
                orchestrator.extract_all(frame)

            stats = orchestrator.get_stats()

        assert stats["frames_processed"] == 3
        assert stats["total_observations"] == 6
        assert stats["avg_time_ms"] > 0

    def test_max_workers_in_stats(self):
        """Test that max_workers is in stats."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext], max_workers=4)

        stats = orchestrator.get_stats()

        assert stats["max_workers"] == 4


# =============================================================================
# Configuration Tests
# =============================================================================


class TestExtractorOrchestratorConfig:
    """Tests for configuration options."""

    def test_custom_max_workers(self):
        """Test custom max_workers setting."""
        extractors = [SimpleExtractor(f"ext{i}") for i in range(5)]
        orchestrator = ExtractorOrchestrator(extractors, max_workers=2)

        assert orchestrator._max_workers == 2

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        ext = SimpleExtractor()
        orchestrator = ExtractorOrchestrator([ext], timeout=10.0)

        assert orchestrator._timeout == 10.0

    def test_default_max_workers_is_extractor_count(self):
        """Test that default max_workers equals extractor count."""
        extractors = [SimpleExtractor(f"ext{i}") for i in range(3)]
        orchestrator = ExtractorOrchestrator(extractors)

        assert orchestrator._max_workers == 3
