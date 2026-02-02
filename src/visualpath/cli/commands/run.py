"""Run command for visualpath CLI."""

import sys
from pathlib import Path
from typing import Optional

from visualpath.config import load_yaml_config, ConfigLoadError
from visualpath.core.isolation import IsolationLevel
from visualpath.observability import TraceLevel


def cmd_run(
    config_path: str,
    pipeline_name: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Run a pipeline from configuration.

    Args:
        config_path: Path to the YAML configuration file.
        pipeline_name: Specific pipeline to run (None for all).
        dry_run: If True, validate and show what would run without executing.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    path = Path(config_path)

    if not path.exists():
        print(f"Error: Configuration file not found: {path}", file=sys.stderr)
        return 1

    try:
        config = load_yaml_config(path)
    except ConfigLoadError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Determine which pipelines to run
    if pipeline_name:
        if pipeline_name not in config.pipelines:
            print(
                f"Error: Pipeline '{pipeline_name}' not found. "
                f"Available: {list(config.pipelines.keys())}",
                file=sys.stderr,
            )
            return 1
        pipelines_to_run = {pipeline_name: config.pipelines[pipeline_name]}
    else:
        pipelines_to_run = config.pipelines

    if dry_run:
        return _dry_run(pipelines_to_run)
    else:
        return _execute(pipelines_to_run)


def _dry_run(pipelines: dict) -> int:
    """Show what would be executed without running.

    Args:
        pipelines: Dict of pipeline name to PipelineSchema.

    Returns:
        Exit code (always 0 for dry run).
    """
    print("Dry run - showing execution plan:")
    print("=" * 50)

    for name, pipeline in pipelines.items():
        print(f"\nPipeline: {name}")
        print("-" * 40)

        # Extractors
        print("Extractors:")
        for i, ext in enumerate(pipeline.extractors, 1):
            print(f"  {i}. {ext.name}")
            print(f"     Isolation: {ext.isolation}")
            if ext.isolation == "venv":
                print(f"     Venv: {ext.venv_path}")
            elif ext.isolation == "container":
                print(f"     Image: {ext.container_image}")
            if ext.config:
                print(f"     Config: {ext.config}")

        # Fusion
        print(f"\nFusion: {pipeline.fusion.name}")
        if pipeline.fusion.config:
            print(f"  Config: {pipeline.fusion.config}")

        # Observability
        print(f"\nObservability: {pipeline.observability.level}")
        if pipeline.observability.sinks:
            print("  Sinks:")
            for sink in pipeline.observability.sinks:
                sink_desc = sink.type
                if sink.path:
                    sink_desc += f" -> {sink.path}"
                print(f"    - {sink_desc}")

    print("\n" + "=" * 50)
    print("Dry run complete. No pipelines were executed.")
    return 0


def _execute(pipelines: dict) -> int:
    """Execute the pipelines.

    Args:
        pipelines: Dict of pipeline name to PipelineSchema.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    from visualpath.plugin.discovery import (
        discover_extractors,
        discover_fusions,
        load_extractor,
        load_fusion,
    )
    from visualpath.observability import ObservabilityHub, FileSink, ConsoleSink

    # Verify all plugins are available first
    available_extractors = set(discover_extractors().keys())
    available_fusions = set(discover_fusions().keys())

    errors = []
    for name, pipeline in pipelines.items():
        for ext in pipeline.extractors:
            if ext.name not in available_extractors:
                errors.append(f"Extractor '{ext.name}' not found (pipeline: {name})")

        if pipeline.fusion.name not in available_fusions:
            errors.append(f"Fusion '{pipeline.fusion.name}' not found (pipeline: {name})")

    if errors:
        print("Plugin errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Execute each pipeline
    for name, pipeline in pipelines.items():
        print(f"Running pipeline: {name}")

        # Configure observability
        hub = ObservabilityHub.get_instance()
        level_map = {
            "off": TraceLevel.OFF,
            "minimal": TraceLevel.MINIMAL,
            "normal": TraceLevel.NORMAL,
            "verbose": TraceLevel.VERBOSE,
        }
        trace_level = level_map.get(pipeline.observability.level, TraceLevel.OFF)

        sinks = []
        for sink_config in pipeline.observability.sinks:
            if sink_config.type == "file" and sink_config.path:
                sinks.append(FileSink(sink_config.path))
            elif sink_config.type == "console":
                sinks.append(ConsoleSink())

        hub.configure(level=trace_level, sinks=sinks)

        try:
            # Load extractors
            extractors = []
            for ext_config in pipeline.extractors:
                ExtractorClass = load_extractor(ext_config.name)
                extractor = ExtractorClass(**ext_config.config)
                extractors.append((extractor, ext_config))
                print(f"  Loaded extractor: {ext_config.name}")

            # Load fusion
            FusionClass = load_fusion(pipeline.fusion.name)
            fusion = FusionClass(**pipeline.fusion.config)
            print(f"  Loaded fusion: {pipeline.fusion.name}")

            # Note: Actual pipeline execution requires a video source
            # which is not provided in the config. This is a minimal
            # implementation that demonstrates the loading process.
            print(f"  Pipeline '{name}' loaded successfully.")
            print("  Note: Actual execution requires a video source input.")

        except Exception as e:
            print(f"Error running pipeline '{name}': {e}", file=sys.stderr)
            return 1
        finally:
            hub.shutdown()

    return 0
