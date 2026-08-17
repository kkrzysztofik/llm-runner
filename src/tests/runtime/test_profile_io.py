"""Unit tests for llama_manager.common.profile_io."""

from pathlib import Path

import pytest

from llama_manager.common.profile_io import write_profile_toml


def test_write_profile_toml_rejects_unsupported_value_type(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    with pytest.raises(TypeError, match="unsupported profile value type: set"):
        write_profile_toml(path, {"profiles": [{"name": "p", "bad": {"a", "b"}}]})
    assert not path.exists()
