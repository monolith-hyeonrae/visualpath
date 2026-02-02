"""YAML configuration loader with environment variable substitution.

Provides functionality to load and validate pipeline configurations
from YAML files with support for environment variable expansion.

Environment variable syntax:
    ${VAR}          - Required variable, raises error if not set
    ${VAR:-default} - Optional variable with default value

Example:
    >>> config = load_yaml_config("pipeline.yaml")
    >>> for name, pipeline in config.pipelines.items():
    ...     print(f"Pipeline: {name}, extractors: {len(pipeline.extractors)}")
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import ValidationError

from visualpath.config.schema import ConfigSchema


class ConfigLoadError(Exception):
    """Error loading or validating configuration."""

    pass


# Pattern for environment variables: ${VAR} or ${VAR:-default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in a value.

    Supports:
    - ${VAR} - Required variable, raises KeyError if not set
    - ${VAR:-default} - Optional variable with default value

    Args:
        value: Value to process (string, dict, list, or other).

    Returns:
        Value with environment variables substituted.

    Raises:
        KeyError: If a required environment variable is not set.

    Examples:
        >>> os.environ["MY_VAR"] = "hello"
        >>> substitute_env_vars("${MY_VAR}")
        'hello'
        >>> substitute_env_vars("${MISSING:-default}")
        'default'
        >>> substitute_env_vars({"key": "${MY_VAR}"})
        {'key': 'hello'}
    """
    if isinstance(value, str):
        return _substitute_string(value)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    else:
        return value


def _substitute_string(s: str) -> str:
    """Substitute environment variables in a string.

    Args:
        s: String potentially containing ${VAR} or ${VAR:-default} patterns.

    Returns:
        String with variables substituted.

    Raises:
        KeyError: If a required variable is not set.
    """

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)  # None if no default specified

        value = os.environ.get(var_name)
        if value is not None:
            return value
        elif default is not None:
            return default
        else:
            raise KeyError(
                f"Environment variable '{var_name}' is not set "
                f"and no default provided"
            )

    return ENV_VAR_PATTERN.sub(replacer, s)


def load_yaml_config(
    path: Union[str, Path],
    substitute_vars: bool = True,
) -> ConfigSchema:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.
        substitute_vars: Whether to substitute environment variables.

    Returns:
        Validated ConfigSchema object.

    Raises:
        ConfigLoadError: If the file cannot be loaded or validated.
        FileNotFoundError: If the config file doesn't exist.

    Example:
        >>> config = load_yaml_config("pipeline.yaml")
        >>> print(config.version)
        '1.0'
        >>> for name, pipeline in config.pipelines.items():
        ...     print(f"Pipeline: {name}")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in {path}: {e}") from e

    if raw_data is None:
        raise ConfigLoadError(f"Empty configuration file: {path}")

    if not isinstance(raw_data, dict):
        raise ConfigLoadError(
            f"Configuration must be a dictionary, got {type(raw_data).__name__}"
        )

    # Substitute environment variables
    if substitute_vars:
        try:
            raw_data = substitute_env_vars(raw_data)
        except KeyError as e:
            raise ConfigLoadError(f"Environment variable error: {e}") from e

    # Validate with Pydantic
    try:
        config = ConfigSchema.model_validate(raw_data)
    except ValidationError as e:
        raise ConfigLoadError(f"Configuration validation failed: {e}") from e

    return config


def load_yaml_string(
    content: str,
    substitute_vars: bool = True,
) -> ConfigSchema:
    """Load and validate a YAML configuration from a string.

    Args:
        content: YAML content as a string.
        substitute_vars: Whether to substitute environment variables.

    Returns:
        Validated ConfigSchema object.

    Raises:
        ConfigLoadError: If the content cannot be parsed or validated.
    """
    try:
        raw_data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML: {e}") from e

    if raw_data is None:
        raise ConfigLoadError("Empty configuration")

    if not isinstance(raw_data, dict):
        raise ConfigLoadError(
            f"Configuration must be a dictionary, got {type(raw_data).__name__}"
        )

    if substitute_vars:
        try:
            raw_data = substitute_env_vars(raw_data)
        except KeyError as e:
            raise ConfigLoadError(f"Environment variable error: {e}") from e

    try:
        config = ConfigSchema.model_validate(raw_data)
    except ValidationError as e:
        raise ConfigLoadError(f"Configuration validation failed: {e}") from e

    return config
