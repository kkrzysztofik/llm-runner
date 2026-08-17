import json

import pytest

from llama_manager.config.reasoning_effort import (
    REASONING_EFFORT_DEFAULT,
    REASONING_EFFORT_JSON_CONFLICT,
    REASONING_EFFORT_VALUES,
    chat_template_kwargs_has_reasoning_effort,
    merge_chat_template_kwargs,
    resolve_reasoning_effort,
)
from tests.support.helpers import make_server_config


def test_values_are_official_template_safe() -> None:
    assert frozenset({"xhigh", "medium", "low"}) == REASONING_EFFORT_VALUES
    assert REASONING_EFFORT_DEFAULT == "medium"
    assert "high" not in REASONING_EFFORT_VALUES


def test_resolve_reasoning_effort_valid() -> None:
    assert resolve_reasoning_effort({"reasoning_effort": "xhigh"}) == "xhigh"
    assert resolve_reasoning_effort({"reasoning_effort": "medium"}) == "medium"
    assert resolve_reasoning_effort({"reasoning_effort": "low"}) == "low"


@pytest.mark.parametrize("raw", ["high", "auto", "bogus", 1, None])
def test_resolve_reasoning_effort_invalid_defaults_medium(raw: object) -> None:
    assert resolve_reasoning_effort({"reasoning_effort": raw}) == "medium"
    assert resolve_reasoning_effort({}) == "medium"


def test_has_reasoning_effort_detects_key() -> None:
    assert chat_template_kwargs_has_reasoning_effort("") is False
    assert chat_template_kwargs_has_reasoning_effort("{}") is False
    assert chat_template_kwargs_has_reasoning_effort('{"preserve_thinking":true}') is False
    assert chat_template_kwargs_has_reasoning_effort('{"reasoning_effort":"low"}') is True


def test_has_reasoning_effort_invalid_json_is_false() -> None:
    assert chat_template_kwargs_has_reasoning_effort("not json") is False


def test_merge_empty_kwargs_emits_medium() -> None:
    merged = merge_chat_template_kwargs("", "medium")
    assert json.loads(merged) == {"reasoning_effort": "medium"}


def test_merge_preserves_existing_keys() -> None:
    merged = merge_chat_template_kwargs('{"preserve_thinking":true}', "low")
    assert json.loads(merged) == {"preserve_thinking": True, "reasoning_effort": "low"}


def test_merge_conflict_raises() -> None:
    with pytest.raises(ValueError, match="Thinking level"):
        merge_chat_template_kwargs('{"reasoning_effort":"xhigh"}', "medium")
    assert "reasoning_effort" in REASONING_EFFORT_JSON_CONFLICT


def test_merge_non_object_json_raises() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        merge_chat_template_kwargs("[1]", "medium")


def test_server_config_defaults_to_medium() -> None:
    cfg = make_server_config(alias="test", server_bin="/usr/bin/llama-server")
    assert cfg.reasoning_effort == "medium"


def test_server_config_invalid_reasoning_effort_normalizes() -> None:
    cfg = make_server_config(
        alias="test",
        server_bin="/usr/bin/llama-server",
        reasoning_effort="high",
    )
    assert cfg.reasoning_effort == "medium"
