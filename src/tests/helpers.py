from __future__ import annotations

"""Compatibility re-exports for shared test helpers.

New tests should import from ``tests.support`` modules directly.
"""


from tests.support.helpers import (
    assert_dicts_equal,
    assert_json_has_keys,
    assert_sorted_identically,
    make_server_config,
    normalize_output_for_diff,
)

__all__ = [
    "assert_dicts_equal",
    "assert_json_has_keys",
    "assert_sorted_identically",
    "make_server_config",
    "normalize_output_for_diff",
]
