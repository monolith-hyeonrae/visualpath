"""Plugin discovery system for visualpath.

This module provides infrastructure for discovering and loading
extractor plugins via Python entry points.

Plugins register themselves using the `visualpath.extractors` entry point
group in their pyproject.toml:

```toml
[project.entry-points."visualpath.extractors"]
face = "myplugin.extractors:FaceExtractor"
pose = "myplugin.extractors:PoseExtractor"
```

Example:
    >>> from visualpath.plugin import discover_extractors, load_extractor
    >>>
    >>> # Discover all available extractor plugins
    >>> extractors = discover_extractors()
    >>> for name, entry_point in extractors.items():
    ...     print(f"Found extractor: {name}")
    >>>
    >>> # Load a specific extractor
    >>> FaceExtractor = load_extractor("face")
    >>> extractor = FaceExtractor()
"""

import sys
from typing import Dict, Optional, Type, Any

from visualpath.core.extractor import BaseExtractor

# Entry point group names
EXTRACTORS_GROUP = "visualpath.extractors"
FUSIONS_GROUP = "visualpath.fusions"


def _get_entry_points(group: str) -> Dict[str, Any]:
    """Get entry points for a group.

    Uses importlib.metadata (Python 3.10+) for entry point discovery.

    Args:
        group: The entry point group name.

    Returns:
        Dict mapping entry point names to entry point objects.
    """
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group=group)
        return {ep.name: ep for ep in eps}
    else:
        # Fallback for Python 3.9
        from importlib.metadata import entry_points as get_eps
        all_eps = get_eps()
        if hasattr(all_eps, 'select'):
            eps = all_eps.select(group=group)
        else:
            eps = all_eps.get(group, [])
        return {ep.name: ep for ep in eps}


def discover_extractors() -> Dict[str, Any]:
    """Discover all available extractor plugins.

    Scans the `visualpath.extractors` entry point group for registered
    extractors.

    Returns:
        Dict mapping extractor names to their entry points.

    Example:
        >>> extractors = discover_extractors()
        >>> print(list(extractors.keys()))
        ['face', 'pose', 'gesture', 'quality']
    """
    return _get_entry_points(EXTRACTORS_GROUP)


def discover_fusions() -> Dict[str, Any]:
    """Discover all available fusion plugins.

    Scans the `visualpath.fusions` entry point group for registered
    fusion modules.

    Returns:
        Dict mapping fusion names to their entry points.
    """
    return _get_entry_points(FUSIONS_GROUP)


def load_extractor(name: str) -> Type[BaseExtractor]:
    """Load an extractor class by name.

    Args:
        name: The registered name of the extractor.

    Returns:
        The extractor class.

    Raises:
        KeyError: If no extractor with the given name is registered.
        ImportError: If the extractor cannot be loaded.

    Example:
        >>> FaceExtractor = load_extractor("face")
        >>> extractor = FaceExtractor()
    """
    extractors = discover_extractors()
    if name not in extractors:
        raise KeyError(
            f"No extractor registered with name '{name}'. "
            f"Available: {list(extractors.keys())}"
        )
    entry_point = extractors[name]
    return entry_point.load()


def load_fusion(name: str) -> Type:
    """Load a fusion class by name.

    Args:
        name: The registered name of the fusion module.

    Returns:
        The fusion class.

    Raises:
        KeyError: If no fusion with the given name is registered.
        ImportError: If the fusion cannot be loaded.
    """
    fusions = discover_fusions()
    if name not in fusions:
        raise KeyError(
            f"No fusion registered with name '{name}'. "
            f"Available: {list(fusions.keys())}"
        )
    entry_point = fusions[name]
    return entry_point.load()


def create_extractor(name: str, **kwargs) -> BaseExtractor:
    """Create an extractor instance by name.

    Convenience function that loads and instantiates an extractor.

    Args:
        name: The registered name of the extractor.
        **kwargs: Arguments passed to the extractor constructor.

    Returns:
        An initialized extractor instance.

    Example:
        >>> extractor = create_extractor("face", device="cuda:0")
    """
    ExtractorClass = load_extractor(name)
    return ExtractorClass(**kwargs)


class PluginRegistry:
    """Registry for managing loaded plugins.

    Provides caching and lifecycle management for plugins.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_extractor("face", FaceExtractor)
        >>> extractor = registry.create_extractor("face")
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._extractors: Dict[str, Type[BaseExtractor]] = {}
        self._fusions: Dict[str, Type] = {}
        self._instances: Dict[str, BaseExtractor] = {}

    def register_extractor(
        self,
        name: str,
        extractor_class: Type[BaseExtractor],
    ) -> None:
        """Register an extractor class.

        Args:
            name: Name to register under.
            extractor_class: The extractor class.
        """
        self._extractors[name] = extractor_class

    def register_fusion(self, name: str, fusion_class: Type) -> None:
        """Register a fusion class.

        Args:
            name: Name to register under.
            fusion_class: The fusion class.
        """
        self._fusions[name] = fusion_class

    def get_extractor_class(self, name: str) -> Type[BaseExtractor]:
        """Get a registered extractor class.

        Args:
            name: The extractor name.

        Returns:
            The extractor class.

        Raises:
            KeyError: If not registered.
        """
        if name in self._extractors:
            return self._extractors[name]
        # Fall back to entry point discovery
        return load_extractor(name)

    def create_extractor(
        self,
        name: str,
        singleton: bool = False,
        **kwargs,
    ) -> BaseExtractor:
        """Create an extractor instance.

        Args:
            name: The extractor name.
            singleton: If True, return cached instance.
            **kwargs: Constructor arguments.

        Returns:
            Extractor instance.
        """
        if singleton and name in self._instances:
            return self._instances[name]

        ExtractorClass = self.get_extractor_class(name)
        instance = ExtractorClass(**kwargs)

        if singleton:
            self._instances[name] = instance

        return instance

    def list_extractors(self) -> list[str]:
        """List all available extractor names.

        Returns:
            List of extractor names (registered + discovered).
        """
        discovered = set(discover_extractors().keys())
        registered = set(self._extractors.keys())
        return sorted(discovered | registered)

    def list_fusions(self) -> list[str]:
        """List all available fusion names.

        Returns:
            List of fusion names (registered + discovered).
        """
        discovered = set(discover_fusions().keys())
        registered = set(self._fusions.keys())
        return sorted(discovered | registered)

    def cleanup(self) -> None:
        """Clean up all cached instances."""
        for instance in self._instances.values():
            try:
                instance.cleanup()
            except Exception:
                pass
        self._instances.clear()
