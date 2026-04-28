"""Reusable assertion helpers for tests."""

from typing import Any


def assert_dicts_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    message: str = "",
) -> None:
    """Assert two dictionaries are equal, with useful key diagnostics."""
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
    """Assert two sorted lists are identical."""
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
    """Normalize output strings for consistent diff comparison."""
    stripped_lines = [line.rstrip() for line in output.splitlines()]
    while stripped_lines and not stripped_lines[-1]:
        stripped_lines.pop()
    return "\n".join(stripped_lines)


def assert_json_has_keys(data: dict[str, Any], required_keys: set[str]) -> None:
    """Assert JSON-like dict contains all required keys."""
    missing = required_keys - set(data)
    if missing:
        raise AssertionError(f"Missing required JSON keys: {sorted(missing)}")

