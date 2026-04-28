"""Shared test helpers.

Provides utilities for common test setup and deterministic assertions:
- make_server_config: Create a minimal ServerConfig with optional overrides
- assert_dicts_equal: Compare dicts with optional key ordering
- assert_sorted_identically: Compare sorted lists for equality
- normalize_output_for_diff: Normalize output strings for consistent diffing
"""

from typing import Any

from llama_manager.config import ServerConfig


def make_server_config(**overrides: object) -> ServerConfig:
    """Create a minimal ServerConfig for tests."""
    defaults: dict[str, object] = {
        "model": "/models/test.gguf",
        "alias": "test-slot",
        "device": "SYCL0",
        "port": 8080,
        "ctx_size": 4096,
        "ubatch_size": 512,
        "threads": 4,
        "server_bin": "dummy-llama-server",
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)  # type: ignore[arg-type]


def assert_dicts_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    message: str = "",
) -> None:
    """Assert two dictionaries are equal, handling key ordering.

    This helper is useful for testing configs and results where key
    ordering may vary but semantic equality is required.

    Args:
        actual: The actual dictionary from the code under test.
        expected: The expected dictionary.
        message: Optional custom error message.

    Raises:
        AssertionError: If dicts differ in keys or values.

    """
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        error_parts = []
        if missing:
            error_parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra keys: {sorted(extra)}")
        raise AssertionError(
            f"Dict key mismatch. {', '.join(error_parts)}" + (f": {message}" if message else "")
        )

    for key in sorted(actual_keys):
        if actual[key] != expected[key]:
            error_msg = (
                f"Value mismatch for key '{key}': expected {expected[key]!r}, got {actual[key]!r}"
            )
            if message:
                error_msg += f" ({message})"
            raise AssertionError(error_msg)


def assert_sorted_identically(
    actual: list[Any],
    expected: list[Any],
    key_name: str | None = None,
    message: str = "",
) -> None:
    """Assert two sorted lists are identical.

    This is useful for testing deterministic sorting behavior where
    the order should be stable and reproducible.

    Args:
        actual: The actual sorted list from the code under test.
        expected: The expected sorted list.
        key_name: Optional descriptive key name for mismatch diagnostics.
        message: Optional custom error message.

    Raises:
        AssertionError: If lists differ in length or element ordering.

    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"List length mismatch: expected {len(expected)}, got {len(actual)}"
            + (f": {message}" if message else "")
        )

    for i, (act_item, exp_item) in enumerate(zip(actual, expected, strict=True)):
        if act_item != exp_item:
            error_msg = f"Item at index {i} mismatch: expected {exp_item!r}, got {act_item!r}"
            if key_name:
                error_msg += f" (key={key_name})"
            if message:
                error_msg += f" ({message})"
            raise AssertionError(error_msg)


def normalize_output_for_diff(output: str) -> str:
    """Normalize output string for consistent diff comparison.

    This helper right-strips each line, normalizes line endings,
    and removes trailing blank lines for cleaner diffs.

    Useful when comparing console output, error messages, or logs
    where formatting may vary slightly.

    Args:
        output: The raw output string to normalize.

    Returns:
        Normalized string with consistent formatting.

    """
    # Split into lines
    lines = output.splitlines()

    # Strip each line and remove trailing blank lines
    stripped_lines = [line.rstrip() for line in lines]
    while stripped_lines and not stripped_lines[-1]:
        stripped_lines.pop()

    # Rejoin with consistent line endings
    return "\n".join(stripped_lines)
