"""Configuration system for visualpath.

Provides YAML-based declarative pipeline configuration with:
- Pydantic schema validation
- Environment variable substitution (${VAR} and ${VAR:-default})
- Runtime object conversion

Example YAML config:
    version: "1.0"
    pipelines:
      face_analysis:
        extractors:
          - name: face
            isolation: venv
            venv_path: "${FACE_VENV:-/opt/venvs/face}"
            config:
              device: "cuda:0"
        fusion:
          name: highlight
          config:
            cooldown_sec: 2.0
        observability:
          level: normal
          sinks:
            - type: file
              path: "${LOG_DIR:-./logs}/trace.jsonl"

Example usage:
    >>> from visualpath.config import load_yaml_config, ConfigSchema
    >>> config = load_yaml_config("pipeline.yaml")
    >>> for name, pipeline in config.pipelines.items():
    ...     print(f"Pipeline: {name}")
"""

from visualpath.config.schema import (
    ConfigSchema,
    PipelineSchema,
    ExtractorSchema,
    FusionSchema,
    ObservabilitySchema,
    SinkSchema,
)
from visualpath.config.loader import (
    load_yaml_config,
    substitute_env_vars,
    ConfigLoadError,
)

__all__ = [
    # Schema models
    "ConfigSchema",
    "PipelineSchema",
    "ExtractorSchema",
    "FusionSchema",
    "ObservabilitySchema",
    "SinkSchema",
    # Loader
    "load_yaml_config",
    "substitute_env_vars",
    "ConfigLoadError",
]
