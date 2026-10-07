"""LocLM entry point.

This module provides the main entry point for the `loclm` CLI command.
"""

from loclm.cli.app import app


def run() -> None:
    """Main entry point for the loclm CLI."""
    app()


if __name__ == "__main__":
    run()
