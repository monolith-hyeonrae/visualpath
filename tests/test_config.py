"""Tests for visualpath configuration system."""

import os
import tempfile
from pathlib import Path

import pytest

from visualpath.config import (
    ConfigSchema,
    PipelineSchema,
    ExtractorSchema,
    FusionSchema,
    ObservabilitySchema,
    SinkSchema,
    load_yaml_config,
    substitute_env_vars,
    ConfigLoadError,
)
from visualpath.config.loader import load_yaml_string


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""

    def test_simple_var(self):
        """Test simple variable substitution."""
        os.environ["TEST_VAR"] = "hello"
        try:
            assert substitute_env_vars("${TEST_VAR}") == "hello"
        finally:
            del os.environ["TEST_VAR"]

    def test_var_with_default(self):
        """Test variable with default value."""
        # Ensure var is not set
        os.environ.pop("MISSING_VAR", None)
        assert substitute_env_vars("${MISSING_VAR:-default}") == "default"

    def test_var_with_default_when_set(self):
        """Test that set variable takes precedence over default."""
        os.environ["SET_VAR"] = "actual"
        try:
            assert substitute_env_vars("${SET_VAR:-default}") == "actual"
        finally:
            del os.environ["SET_VAR"]

    def test_missing_required_var_raises(self):
        """Test that missing required variable raises KeyError."""
        os.environ.pop("REQUIRED_VAR", None)
        with pytest.raises(KeyError, match="REQUIRED_VAR"):
            substitute_env_vars("${REQUIRED_VAR}")

    def test_nested_dict(self):
        """Test substitution in nested dictionaries."""
        os.environ["NESTED_VAR"] = "value"
        try:
            data = {"outer": {"inner": "${NESTED_VAR}"}}
            result = substitute_env_vars(data)
            assert result == {"outer": {"inner": "value"}}
        finally:
            del os.environ["NESTED_VAR"]

    def test_list(self):
        """Test substitution in lists."""
        os.environ["LIST_VAR"] = "item"
        try:
            data = ["${LIST_VAR}", "static"]
            result = substitute_env_vars(data)
            assert result == ["item", "static"]
        finally:
            del os.environ["LIST_VAR"]

    def test_mixed_string(self):
        """Test substitution in string with multiple variables."""
        os.environ["VAR1"] = "hello"
        os.environ["VAR2"] = "world"
        try:
            result = substitute_env_vars("${VAR1} ${VAR2}!")
            assert result == "hello world!"
        finally:
            del os.environ["VAR1"]
            del os.environ["VAR2"]

    def test_non_string_passthrough(self):
        """Test that non-string values pass through unchanged."""
        assert substitute_env_vars(42) == 42
        assert substitute_env_vars(3.14) == 3.14
        assert substitute_env_vars(True) is True
        assert substitute_env_vars(None) is None

    def test_empty_default(self):
        """Test variable with empty default."""
        os.environ.pop("EMPTY_DEFAULT_VAR", None)
        assert substitute_env_vars("${EMPTY_DEFAULT_VAR:-}") == ""


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_sink_schema_file_requires_path(self):
        """Test that file sink requires path."""
        with pytest.raises(ValueError, match="path"):
            SinkSchema(type="file")  # No path

    def test_sink_schema_console_no_path(self):
        """Test that console sink doesn't require path."""
        sink = SinkSchema(type="console")
        assert sink.type == "console"
        assert sink.path is None

    def test_extractor_schema_venv_requires_path(self):
        """Test that venv isolation requires venv_path."""
        with pytest.raises(ValueError, match="venv_path"):
            ExtractorSchema(name="test", isolation="venv")

    def test_extractor_schema_venv_with_path(self):
        """Test valid venv extractor config."""
        ext = ExtractorSchema(
            name="test",
            isolation="venv",
            venv_path="/opt/venv",
        )
        assert ext.venv_path == "/opt/venv"

    def test_extractor_schema_container_requires_image(self):
        """Test that container isolation requires container_image."""
        with pytest.raises(ValueError, match="container_image"):
            ExtractorSchema(name="test", isolation="container")

    def test_config_schema_version_validation(self):
        """Test that invalid version is rejected."""
        with pytest.raises(ValueError, match="Unsupported config version"):
            ConfigSchema(
                version="2.0",
                pipelines={
                    "test": PipelineSchema(
                        extractors=[ExtractorSchema(name="dummy")],
                        fusion=FusionSchema(name="simple"),
                    )
                },
            )

    def test_config_schema_requires_pipelines(self):
        """Test that at least one pipeline is required."""
        with pytest.raises(ValueError):
            ConfigSchema(version="1.0", pipelines={})

    def test_pipeline_schema_requires_extractors(self):
        """Test that at least one extractor is required."""
        with pytest.raises(ValueError):
            PipelineSchema(
                extractors=[],
                fusion=FusionSchema(name="simple"),
            )


class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_load_simple_config(self):
        """Test loading a simple configuration."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: dummy
        isolation: inline
    fusion:
      name: simple
      config:
        threshold: 0.5
"""
        config = load_yaml_string(yaml_content)
        assert config.version == "1.0"
        assert "test" in config.pipelines
        assert len(config.pipelines["test"].extractors) == 1
        assert config.pipelines["test"].extractors[0].name == "dummy"

    def test_load_config_with_env_vars(self):
        """Test loading config with environment variable substitution."""
        os.environ["TEST_DEVICE"] = "cuda:1"
        try:
            yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: face
        isolation: inline
        config:
          device: "${TEST_DEVICE}"
    fusion:
      name: simple
"""
            config = load_yaml_string(yaml_content)
            assert config.pipelines["test"].extractors[0].config["device"] == "cuda:1"
        finally:
            del os.environ["TEST_DEVICE"]

    def test_load_config_with_defaults(self):
        """Test loading config with default environment variables."""
        os.environ.pop("MISSING_PATH", None)
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: dummy
    fusion:
      name: simple
    observability:
      level: normal
      sinks:
        - type: file
          path: "${MISSING_PATH:-./default.log}"
"""
        config = load_yaml_string(yaml_content)
        assert config.pipelines["test"].observability.sinks[0].path == "./default.log"

    def test_load_from_file(self):
        """Test loading config from a file."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: dummy
    fusion:
      name: simple
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = load_yaml_config(f.name)
                assert config.version == "1.0"
            finally:
                os.unlink(f.name)

    def test_load_missing_file_raises(self):
        """Test that loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises(self):
        """Test that invalid YAML raises ConfigLoadError."""
        with pytest.raises(ConfigLoadError, match="Invalid YAML"):
            load_yaml_string("{ invalid yaml [")

    def test_load_empty_config_raises(self):
        """Test that empty config raises ConfigLoadError."""
        with pytest.raises(ConfigLoadError, match="Empty"):
            load_yaml_string("")

    def test_load_non_dict_raises(self):
        """Test that non-dict config raises ConfigLoadError."""
        with pytest.raises(ConfigLoadError, match="dictionary"):
            load_yaml_string("- item1\n- item2")

    def test_load_config_validation_error(self):
        """Test that validation errors are wrapped in ConfigLoadError."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors: []
    fusion:
      name: simple
"""
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_yaml_string(yaml_content)


class TestFullConfigParsing:
    """Integration tests for complete configuration parsing."""

    def test_full_pipeline_config(self):
        """Test parsing a complete pipeline configuration."""
        yaml_content = """
version: "1.0"
globals:
  env: production

pipelines:
  face_analysis:
    extractors:
      - name: face
        isolation: venv
        venv_path: /opt/venvs/face
        config:
          device: cuda:0
          min_face_size: 50

      - name: quality
        isolation: process
        config:
          blur_threshold: 100.0

    fusion:
      name: highlight
      config:
        threshold: 0.7
        cooldown_sec: 2.0

    observability:
      level: verbose
      sinks:
        - type: file
          path: /var/log/trace.jsonl
        - type: console
"""
        config = load_yaml_string(yaml_content)

        assert config.version == "1.0"
        assert config.globals["env"] == "production"

        pipeline = config.pipelines["face_analysis"]
        assert len(pipeline.extractors) == 2

        face_ext = pipeline.extractors[0]
        assert face_ext.name == "face"
        assert face_ext.isolation == "venv"
        assert face_ext.venv_path == "/opt/venvs/face"
        assert face_ext.config["device"] == "cuda:0"

        quality_ext = pipeline.extractors[1]
        assert quality_ext.name == "quality"
        assert quality_ext.isolation == "process"

        fusion = pipeline.fusion
        assert fusion.name == "highlight"
        assert fusion.config["cooldown_sec"] == 2.0

        obs = pipeline.observability
        assert obs.level == "verbose"
        assert len(obs.sinks) == 2
        assert obs.sinks[0].type == "file"
        assert obs.sinks[1].type == "console"
