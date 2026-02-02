"""Tests for visualpath CLI."""

import os
import sys
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest

from visualpath.cli import main, create_parser


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_parser_creation(self):
        """Test that parser is created correctly."""
        parser = create_parser()
        assert parser.prog == "visualpath"

    def test_help_no_error(self):
        """Test that help doesn't raise an error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_shows_help(self):
        """Test that no arguments shows help and exits 0."""
        result = main([])
        assert result == 0

    def test_version_flag(self):
        """Test --version flag."""
        result = main(["--version"])
        assert result == 0

    def test_version_command(self):
        """Test version command."""
        result = main(["version"])
        assert result == 0


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self, capsys):
        """Test that version command outputs expected info."""
        main(["version"])
        captured = capsys.readouterr()
        assert "visualpath" in captured.out
        assert "Python" in captured.out
        assert "Optional dependencies:" in captured.out


class TestPluginsCommand:
    """Tests for the plugins command."""

    def test_plugins_list(self, capsys):
        """Test plugins list command."""
        result = main(["plugins", "list"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Available Extractors:" in captured.out
        assert "Available Fusions:" in captured.out

    def test_plugins_list_shows_dummy(self, capsys):
        """Test that dummy extractor is listed."""
        main(["plugins", "list"])
        captured = capsys.readouterr()
        # The dummy extractor is registered in pyproject.toml
        assert "dummy" in captured.out


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_simple_config(self, capsys):
        """Test validating a simple configuration."""
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
                result = main(["validate", "-c", f.name])
                assert result == 0
                captured = capsys.readouterr()
                assert "Configuration is valid" in captured.out
            finally:
                os.unlink(f.name)

    def test_validate_missing_file(self, capsys):
        """Test validating a missing file."""
        result = main(["validate", "-c", "/nonexistent/config.yaml"])
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_validate_invalid_config(self, capsys):
        """Test validating an invalid configuration."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors: []
    fusion:
      name: simple
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = main(["validate", "-c", f.name])
                assert result == 1
                captured = capsys.readouterr()
                assert "failed" in captured.err.lower()
            finally:
                os.unlink(f.name)

    def test_validate_with_check_plugins(self, capsys):
        """Test validate with --check-plugins flag."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: dummy
    fusion:
      name: nonexistent_fusion
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = main(["validate", "-c", f.name, "--check-plugins"])
                assert result == 1
                captured = capsys.readouterr()
                assert "not found" in captured.err
            finally:
                os.unlink(f.name)


class TestRunCommand:
    """Tests for the run command."""

    def test_run_dry_run(self, capsys):
        """Test run command with --dry-run flag."""
        yaml_content = """
version: "1.0"
pipelines:
  test:
    extractors:
      - name: dummy
        isolation: inline
        config:
          delay_ms: 10
    fusion:
      name: simple
      config:
        threshold: 0.5
    observability:
      level: normal
      sinks:
        - type: console
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = main(["run", "-c", f.name, "--dry-run"])
                assert result == 0
                captured = capsys.readouterr()
                assert "Dry run" in captured.out
                assert "test" in captured.out  # Pipeline name
                assert "dummy" in captured.out  # Extractor name
            finally:
                os.unlink(f.name)

    def test_run_missing_file(self, capsys):
        """Test run command with missing file."""
        result = main(["run", "-c", "/nonexistent/config.yaml"])
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_run_specific_pipeline_dry_run(self, capsys):
        """Test running a specific pipeline with dry run."""
        yaml_content = """
version: "1.0"
pipelines:
  pipeline1:
    extractors:
      - name: dummy
    fusion:
      name: simple
  pipeline2:
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
                result = main([
                    "run", "-c", f.name,
                    "--pipeline", "pipeline1",
                    "--dry-run"
                ])
                assert result == 0
                captured = capsys.readouterr()
                assert "pipeline1" in captured.out
                # pipeline2 should not appear in the output
            finally:
                os.unlink(f.name)

    def test_run_nonexistent_pipeline(self, capsys):
        """Test running a nonexistent pipeline."""
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
                result = main([
                    "run", "-c", f.name,
                    "--pipeline", "nonexistent"
                ])
                assert result == 1
                captured = capsys.readouterr()
                assert "not found" in captured.err
            finally:
                os.unlink(f.name)


class TestCLIEntryPoint:
    """Tests for CLI as entry point."""

    def test_main_returns_int(self):
        """Test that main always returns an integer."""
        result = main([])
        assert isinstance(result, int)

    def test_main_with_invalid_command(self):
        """Test that invalid subcommand shows help."""
        # argparse handles unknown commands
        with pytest.raises(SystemExit):
            main(["unknown_command"])
