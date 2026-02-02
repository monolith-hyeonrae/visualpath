"""Version command for visualpath CLI."""

import sys
from importlib.metadata import version, PackageNotFoundError


def cmd_version() -> int:
    """Display version information.

    Returns:
        Exit code (always 0).
    """
    # Get visualpath version
    try:
        vp_version = version("visualpath")
    except PackageNotFoundError:
        vp_version = "development"

    print(f"visualpath {vp_version}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check optional dependencies
    print("\nOptional dependencies:")

    deps = [
        ("pyyaml", "YAML config support"),
        ("pydantic", "Config validation"),
        ("pyzmq", "IPC for distributed workers"),
    ]

    for pkg, desc in deps:
        try:
            pkg_version = version(pkg)
            status = f"v{pkg_version}"
        except PackageNotFoundError:
            status = "not installed"
        print(f"  {pkg}: {status} ({desc})")

    return 0
