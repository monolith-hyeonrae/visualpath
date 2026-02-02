"""Pydantic validation models for visualpath configuration.

Defines the schema for YAML configuration files with validation
rules and sensible defaults.
"""

from typing import Dict, List, Any, Optional, Literal, Union
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


# =============================================================================
# Flow Graph Configuration Schema
# =============================================================================


class FlowNodeSchema(BaseModel):
    """Configuration for a flow graph node.

    Attributes:
        name: Unique name for this node.
        type: Node type (source, path, sampler, filter, branch, fanout, join).
        config: Node-specific configuration.
    """

    name: str
    type: Literal[
        "source",
        "path",
        "sampler",
        "rate_limiter",
        "filter",
        "observation_filter",
        "signal_filter",
        "branch",
        "fanout",
        "multi_branch",
        "join",
        "cascade_fusion",
        "collector",
    ]
    config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_node_config(self) -> "FlowNodeSchema":
        """Validate node-specific configuration requirements."""
        cfg = self.config

        if self.type == "sampler":
            if "every_nth" not in cfg:
                cfg["every_nth"] = 1

        elif self.type == "rate_limiter":
            if "min_interval_ms" not in cfg:
                raise ValueError("rate_limiter requires 'min_interval_ms'")

        elif self.type == "path":
            # path can have extractors and/or fusion
            pass

        elif self.type == "branch":
            required = ["condition", "if_true", "if_false"]
            for field in required:
                if field not in cfg:
                    raise ValueError(f"branch requires '{field}'")

        elif self.type == "fanout":
            if "paths" not in cfg or not cfg["paths"]:
                raise ValueError("fanout requires non-empty 'paths' list")

        elif self.type == "join":
            if "input_paths" not in cfg or not cfg["input_paths"]:
                raise ValueError("join requires non-empty 'input_paths' list")

        elif self.type == "signal_filter":
            required = ["signal_name", "threshold"]
            for field in required:
                if field not in cfg:
                    raise ValueError(f"signal_filter requires '{field}'")

        return self


class FlowEdgeSchema(BaseModel):
    """Configuration for a flow graph edge.

    Attributes:
        source: Name of the source node.
        target: Name of the target node.
        path_filter: Optional path_id filter for conditional routing.
    """

    source: str
    target: str
    path_filter: Optional[str] = None


class FlowGraphSchema(BaseModel):
    """Configuration for a complete flow graph.

    Attributes:
        version: Flow config version.
        entry: Name of the entry node.
        nodes: List of node configurations.
        edges: List of edge configurations.
        on_trigger: Optional trigger handler reference.
    """

    version: str = "1.0"
    entry: str
    nodes: List[FlowNodeSchema] = Field(min_length=1)
    edges: List[FlowEdgeSchema] = Field(default_factory=list)
    on_trigger: Optional[str] = None

    @model_validator(mode="after")
    def validate_graph(self) -> "FlowGraphSchema":
        """Validate graph structure."""
        node_names = {node.name for node in self.nodes}

        # Check entry node exists
        if self.entry not in node_names:
            raise ValueError(f"Entry node '{self.entry}' not found in nodes")

        # Check all edge endpoints exist
        for edge in self.edges:
            if edge.source not in node_names:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in node_names:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")

        return self

    @field_validator("version")
    @classmethod
    def validate_flow_version(cls, v: str) -> str:
        """Validate flow config version."""
        supported = {"1.0"}
        if v not in supported:
            raise ValueError(
                f"Unsupported flow config version: {v}. Supported: {supported}"
            )
        return v
