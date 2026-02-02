"""Pydantic validation models for visualpath configuration.

Defines the schema for YAML configuration files with validation
rules and sensible defaults.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class SinkSchema(BaseModel):
    """Configuration for an observability sink.

    Attributes:
        type: Sink type (file, console, memory, null).
        path: File path for file sinks.
        options: Additional sink-specific options.
    """

    type: Literal["file", "console", "memory", "null"] = "file"
    path: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_file_sink_has_path(self) -> "SinkSchema":
        """Validate that file sinks have a path."""
        if self.type == "file" and not self.path:
            raise ValueError("File sink requires 'path' to be set")
        return self


class ObservabilitySchema(BaseModel):
    """Configuration for observability/tracing.

    Attributes:
        level: Trace level (off, minimal, normal, verbose).
        sinks: List of sink configurations.
    """

    level: Literal["off", "minimal", "normal", "verbose"] = "off"
    sinks: List[SinkSchema] = Field(default_factory=list)


class ExtractorSchema(BaseModel):
    """Configuration for an extractor in the pipeline.

    Attributes:
        name: Extractor name (must be registered via entry point).
        isolation: Isolation level for execution.
        venv_path: Path to venv (required for venv isolation).
        container_image: Container image (for container isolation).
        config: Extractor-specific configuration.
    """

    name: str
    isolation: Literal["inline", "thread", "process", "venv", "container"] = "inline"
    venv_path: Optional[str] = None
    container_image: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_isolation_requirements(self) -> "ExtractorSchema":
        """Validate isolation-specific requirements."""
        if self.isolation == "venv" and not self.venv_path:
            raise ValueError("VENV isolation requires 'venv_path' to be set")
        if self.isolation == "container" and not self.container_image:
            raise ValueError("Container isolation requires 'container_image' to be set")
        return self


class FusionSchema(BaseModel):
    """Configuration for the fusion module.

    Attributes:
        name: Fusion name (must be registered via entry point).
        config: Fusion-specific configuration.
    """

    name: str
    config: Dict[str, Any] = Field(default_factory=dict)


class PipelineSchema(BaseModel):
    """Configuration for a single pipeline.

    A pipeline consists of one or more extractors feeding into
    a fusion module.

    Attributes:
        extractors: List of extractor configurations.
        fusion: Fusion module configuration.
        observability: Optional observability settings.
    """

    extractors: List[ExtractorSchema] = Field(min_length=1)
    fusion: FusionSchema
    observability: ObservabilitySchema = Field(default_factory=ObservabilitySchema)


class ConfigSchema(BaseModel):
    """Root configuration schema.

    Attributes:
        version: Configuration file version (currently "1.0").
        pipelines: Mapping of pipeline names to their configurations.
        globals: Optional global settings applied to all pipelines.
    """

    version: str = "1.0"
    pipelines: Dict[str, PipelineSchema] = Field(min_length=1)
    globals: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate configuration version."""
        supported = {"1.0"}
        if v not in supported:
            raise ValueError(
                f"Unsupported config version: {v}. Supported: {supported}"
            )
        return v
