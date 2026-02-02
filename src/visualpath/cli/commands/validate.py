"""Validate command for visualpath CLI."""

import sys
from pathlib import Path

from visualpath.config import load_yaml_config, ConfigLoadError
from visualpath.plugin.discovery import discover_extractors, discover_fusions


def cmd_validate(config_path: str, check_plugins: bool = False) -> int:
    """Validate a configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        check_plugins: Whether to verify plugin availability.

    Returns:
        Exit code (0 for success, 1 for validation errors).
    """
    path = Path(config_path)

    if not path.exists():
        print(f"Error: Configuration file not found: {path}", file=sys.stderr)
        return 1

    print(f"Validating: {path}")

    try:
        config = load_yaml_config(path)
    except ConfigLoadError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1

    print(f"  Version: {config.version}")
    print(f"  Pipelines: {len(config.pipelines)}")

    errors = []

    for pipeline_name, pipeline in config.pipelines.items():
        print(f"\n  Pipeline '{pipeline_name}':")
        print(f"    Extractors: {len(pipeline.extractors)}")

        for ext in pipeline.extractors:
            isolation_info = f" ({ext.isolation})"
            if ext.isolation == "venv" and ext.venv_path:
                isolation_info += f" -> {ext.venv_path}"
            print(f"      - {ext.name}{isolation_info}")

        print(f"    Fusion: {pipeline.fusion.name}")
        print(f"    Observability: {pipeline.observability.level}")

        if pipeline.observability.sinks:
            print(f"    Sinks: {len(pipeline.observability.sinks)}")
            for sink in pipeline.observability.sinks:
                sink_info = sink.type
                if sink.path:
                    sink_info += f" -> {sink.path}"
                print(f"      - {sink_info}")

    # Check plugins if requested
    if check_plugins:
        print("\nChecking plugin availability...")

        available_extractors = set(discover_extractors().keys())
        available_fusions = set(discover_fusions().keys())

        for pipeline_name, pipeline in config.pipelines.items():
            for ext in pipeline.extractors:
                if ext.name not in available_extractors:
                    errors.append(
                        f"Extractor '{ext.name}' not found "
                        f"(pipeline: {pipeline_name})"
                    )

            if pipeline.fusion.name not in available_fusions:
                errors.append(
                    f"Fusion '{pipeline.fusion.name}' not found "
                    f"(pipeline: {pipeline_name})"
                )

        if errors:
            print("\nPlugin errors:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        else:
            print("  All plugins available")

    print("\nConfiguration is valid.")
    return 0
