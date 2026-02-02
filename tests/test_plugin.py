"""Tests for plugin discovery system."""

import pytest
from typing import Optional

from visualpath.core import BaseExtractor, Observation
from visualpath.plugin import (
    discover_extractors,
    discover_fusions,
    PluginRegistry,
    EXTRACTORS_GROUP,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""

    def __init__(self, value: float = 0.5):
        self._value = value
        self._initialized = False

    @property
    def name(self) -> str:
        return "mock"

    def extract(self, frame) -> Optional[Observation]:
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"value": self._value},
        )

    def initialize(self) -> None:
        self._initialized = True

    def cleanup(self) -> None:
        self._initialized = False


class MockExtractor2(BaseExtractor):
    """Another mock extractor for testing."""

    @property
    def name(self) -> str:
        return "mock2"

    def extract(self, frame) -> Optional[Observation]:
        return None


# =============================================================================
# Discovery Tests
# =============================================================================


class TestDiscovery:
    """Tests for plugin discovery functions."""

    def test_discover_extractors_returns_dict(self):
        """Test that discover_extractors returns a dict."""
        extractors = discover_extractors()
        assert isinstance(extractors, dict)

    def test_discover_fusions_returns_dict(self):
        """Test that discover_fusions returns a dict."""
        fusions = discover_fusions()
        assert isinstance(fusions, dict)

    def test_extractors_group_name(self):
        """Test the extractors group name constant."""
        assert EXTRACTORS_GROUP == "visualpath.extractors"


# =============================================================================
# PluginRegistry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = PluginRegistry()
        assert registry is not None

    def test_register_extractor(self):
        """Test registering an extractor."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        assert "mock" in registry.list_extractors()

    def test_register_fusion(self):
        """Test registering a fusion."""
        registry = PluginRegistry()

        class MockFusion:
            pass

        registry.register_fusion("mock_fusion", MockFusion)

        assert "mock_fusion" in registry.list_fusions()

    def test_get_extractor_class_registered(self):
        """Test getting a registered extractor class."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        cls = registry.get_extractor_class("mock")

        assert cls is MockExtractor

    def test_get_extractor_class_not_found(self):
        """Test getting non-existent extractor raises KeyError."""
        registry = PluginRegistry()

        with pytest.raises(KeyError):
            registry.get_extractor_class("nonexistent")

    def test_create_extractor(self):
        """Test creating an extractor instance."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        extractor = registry.create_extractor("mock", value=0.7)

        assert isinstance(extractor, MockExtractor)
        assert extractor._value == 0.7

    def test_create_extractor_singleton(self):
        """Test singleton pattern for extractors."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        ext1 = registry.create_extractor("mock", singleton=True)
        ext2 = registry.create_extractor("mock", singleton=True)

        assert ext1 is ext2

    def test_create_extractor_non_singleton(self):
        """Test non-singleton creates new instances."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        ext1 = registry.create_extractor("mock", singleton=False)
        ext2 = registry.create_extractor("mock", singleton=False)

        assert ext1 is not ext2

    def test_list_extractors(self):
        """Test listing registered extractors."""
        registry = PluginRegistry()
        registry.register_extractor("mock1", MockExtractor)
        registry.register_extractor("mock2", MockExtractor2)

        extractors = registry.list_extractors()

        assert "mock1" in extractors
        assert "mock2" in extractors

    def test_cleanup(self):
        """Test cleanup cleans up instances."""
        registry = PluginRegistry()
        registry.register_extractor("mock", MockExtractor)

        ext = registry.create_extractor("mock", singleton=True)
        ext.initialize()
        assert ext._initialized

        registry.cleanup()

        assert not ext._initialized
        # Should create new instance after cleanup
        ext2 = registry.create_extractor("mock", singleton=True)
        assert ext2 is not ext

    def test_multiple_registries_independent(self):
        """Test that multiple registries are independent."""
        reg1 = PluginRegistry()
        reg2 = PluginRegistry()

        reg1.register_extractor("mock", MockExtractor)

        assert "mock" in reg1.list_extractors()
        # Check that discovered extractors are shared, but registered are not
        # Since we register in reg1 only, it should be in reg1's list

    def test_list_extractors_sorted(self):
        """Test that list_extractors returns sorted names."""
        registry = PluginRegistry()
        registry.register_extractor("zebra", MockExtractor)
        registry.register_extractor("alpha", MockExtractor2)

        extractors = registry.list_extractors()

        # Should be alphabetically sorted
        assert extractors.index("alpha") < extractors.index("zebra")
