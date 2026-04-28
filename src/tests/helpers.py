"""Compatibility re-exports for shared test helpers.

New tests should import from ``tests.support`` modules directly.
"""

from tests.support.assertions import (
    assert_dicts_equal,
    assert_json_has_keys,
    assert_sorted_identically,
    normalize_output_for_diff,
)
from tests.support.factories import make_server_config

__all__ = [
    "assert_dicts_equal",
    "assert_json_has_keys",
    "assert_sorted_identically",
    "make_server_config",
    "normalize_output_for_diff",
]
