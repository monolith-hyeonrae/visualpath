"""CLI entry point for visualpath.

Provides command-line interface for:
- Running pipelines from YAML config
- Validating configuration files
- Listing available plugins
- Displaying version information

Usage:
    visualpath run -c config.yaml
    visualpath run -c config.yaml --dry-run
    visualpath validate -c config.yaml
    visualpath plugins list
    visualpath version
"""

import argparse
import sys
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="visualpath",
        description="Video analysis pipeline platform",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a pipeline from configuration",
    )
    run_parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--pipeline",
        help="Specific pipeline to run (default: all)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show what would run without executing",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )
    validate_parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    validate_parser.add_argument(
        "--check-plugins",
        action="store_true",
        help="Also verify that referenced plugins are available",
    )

    # plugins command
    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage plugins",
    )
    plugins_subparsers = plugins_parser.add_subparsers(
        dest="plugins_command",
        help="Plugin commands",
    )
    plugins_subparsers.add_parser(
        "list",
        help="List available plugins",
    )

    # version command
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    # debug command
    debug_parser = subparsers.add_parser(
        "debug",
        help="Run debug pipeline with mock frames",
    )
    debug_parser.add_argument(
        "-n", "--frames",
        type=int,
        default=5,
        help="Number of frames to process (default: 5)",
    )
    debug_parser.add_argument(
        "-s", "--sample",
        type=int,
        default=1,
        help="Sample every Nth frame (default: 1, no sampling)",
    )
    debug_parser.add_argument(
        "-e", "--extractor",
        type=str,
        default="dummy",
        help="Extractor name (default: dummy)",
    )
    debug_parser.add_argument(
        "-f", "--fusion",
        action="store_true",
        help="Enable fusion (triggers)",
    )
    debug_parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug output (show internal operations)",
    )
    debug_parser.add_argument(
        "-b", "--backend",
        choices=["simple", "pathway"],
        default="simple",
        help="Backend to use (default: simple)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle --version flag at top level
    if args.version:
        from visualpath.cli.commands.version import cmd_version
        return cmd_version()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handlers
    if args.command == "run":
        from visualpath.cli.commands.run import cmd_run
        return cmd_run(
            config_path=args.config,
            pipeline_name=args.pipeline,
            dry_run=args.dry_run,
        )

    elif args.command == "validate":
        from visualpath.cli.commands.validate import cmd_validate
        return cmd_validate(
            config_path=args.config,
            check_plugins=args.check_plugins,
        )

    elif args.command == "plugins":
        if args.plugins_command == "list":
            from visualpath.cli.commands.plugins import cmd_plugins_list
            return cmd_plugins_list()
        else:
            # Show plugins help
            parser.parse_args(["plugins", "--help"])
            return 0

    elif args.command == "version":
        from visualpath.cli.commands.version import cmd_version
        return cmd_version()

    elif args.command == "debug":
        from visualpath.cli.commands.debug import cmd_debug
        return cmd_debug(
            frames=args.frames,
            sample=args.sample,
            extractor=args.extractor,
            fusion=args.fusion,
            debug=args.debug,
            backend=args.backend,
        )

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
