"""Shared CLI test fixtures."""

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def disable_colors() -> Generator[None, None, None]:
    """Disable ANSI colors for CLI assertions."""
    from llama_cli.colors import Colors

    original = Colors.enabled
    Colors.enabled = False
    yield
    Colors.enabled = original
