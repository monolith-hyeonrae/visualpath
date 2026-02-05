"""Isolation levels for plugin execution.

IsolationLevel controls how extractors are executed, from inline (same process)
to fully isolated containers. Each level offers different trade-offs between
performance and isolation.

Example:
    >>> from visualpath.core import IsolationLevel, IsolationConfig
    >>>
    >>> # Plugin declares recommended isolation
    >>> class HeavyMLExtractor(Module):
    ...     # Some ML modules may prefer VENV isolation
    ...     pass
    >>>
    >>> # Config can override
    >>> config = IsolationConfig(
    ...     default_level=IsolationLevel.PROCESS,
    ...     overrides={"heavy_ml": IsolationLevel.INLINE},  # For debugging
    ... )
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class IsolationLevel(IntEnum):
    """Isolation level for extractor execution.

    Higher levels provide more isolation but incur more overhead.
    Choose the appropriate level based on the extractor's requirements.

    Levels:
        INLINE: Same process, same thread. Best for simple/fast extractors.
        THREAD: Same process, different thread. Good for I/O-bound work.
        PROCESS: Same venv, different process. Provides memory isolation.
        VENV: Different venv, different process. Resolves dependency conflicts.
        CONTAINER: Full container isolation. Maximum isolation.
    """
    INLINE = 0      # Same process, same thread
    THREAD = 1      # Same process, different thread
    PROCESS = 2     # Same venv, different process
    VENV = 3        # Different venv, different process
    CONTAINER = 4   # Full container isolation

    @classmethod
    def from_string(cls, s: str) -> "IsolationLevel":
        """Parse isolation level from string.

        Args:
            s: String like "inline", "thread", "process", "venv", "container"

        Returns:
            Corresponding IsolationLevel.

        Raises:
            ValueError: If string is not a valid level name.
        """
        mapping = {
            "inline": cls.INLINE,
            "thread": cls.THREAD,
            "process": cls.PROCESS,
            "venv": cls.VENV,
            "container": cls.CONTAINER,
        }
        s_lower = s.lower()
        if s_lower not in mapping:
            raise ValueError(
                f"Unknown isolation level: {s}. "
                f"Valid levels: {', '.join(mapping.keys())}"
            )
        return mapping[s_lower]


@dataclass
class IsolationConfig:
    """Configuration for isolation levels.

    Allows setting a default isolation level and per-extractor overrides.
    Overrides take precedence over both the default and the extractor's
    recommended level.

    Attributes:
        default_level: Default isolation level for all extractors.
        overrides: Per-extractor isolation level overrides.
        venv_paths: Mapping from extractor name to venv path (for VENV level).
        container_images: Mapping from extractor name to container image (for CONTAINER level).

    Example:
        >>> config = IsolationConfig(
        ...     default_level=IsolationLevel.PROCESS,
        ...     overrides={
        ...         "face": IsolationLevel.VENV,  # Heavy ML
        ...         "quality": IsolationLevel.INLINE,  # Simple OpenCV
        ...     },
        ...     venv_paths={
        ...         "face": "/opt/venvs/face",
        ...     },
        ... )
    """

    default_level: IsolationLevel = IsolationLevel.INLINE
    overrides: Dict[str, IsolationLevel] = field(default_factory=dict)
    venv_paths: Dict[str, str] = field(default_factory=dict)
    container_images: Dict[str, str] = field(default_factory=dict)

    def get_level(
        self,
        extractor_name: str,
        recommended: Optional[IsolationLevel] = None,
    ) -> IsolationLevel:
        """Get the effective isolation level for an extractor.

        Priority (highest to lowest):
        1. Override from config
        2. Default from config
        3. Recommended from extractor (if higher than default)

        Args:
            extractor_name: Name of the extractor.
            recommended: Extractor's recommended isolation level.

        Returns:
            Effective isolation level to use.
        """
        # Check for override first
        if extractor_name in self.overrides:
            return self.overrides[extractor_name]

        # Use default, but respect recommended if higher
        if recommended is not None and recommended > self.default_level:
            return recommended

        return self.default_level

    def get_venv_path(self, extractor_name: str) -> Optional[str]:
        """Get the venv path for an extractor.

        Args:
            extractor_name: Name of the extractor.

        Returns:
            Path to venv, or None if not configured.
        """
        return self.venv_paths.get(extractor_name)

    def get_container_image(self, extractor_name: str) -> Optional[str]:
        """Get the container image for an extractor.

        Args:
            extractor_name: Name of the extractor.

        Returns:
            Container image name, or None if not configured.
        """
        return self.container_images.get(extractor_name)
