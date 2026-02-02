"""Plugin discovery and loading system.

This module provides infrastructure for discovering and loading
extractor plugins via Python entry points.
"""

from visualpath.plugin.discovery import (
    discover_extractors,
    discover_fusions,
    load_extractor,
    load_fusion,
    create_extractor,
    PluginRegistry,
    EXTRACTORS_GROUP,
    FUSIONS_GROUP,
)

__all__ = [
    "discover_extractors",
    "discover_fusions",
    "load_extractor",
    "load_fusion",
    "create_extractor",
    "PluginRegistry",
    "EXTRACTORS_GROUP",
    "FUSIONS_GROUP",
]
