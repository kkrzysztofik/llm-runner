"""Qwen 3.8 thinking-level (reasoning_effort) helpers."""

from __future__ import annotations

import json

REASONING_EFFORT_VALUES = frozenset({"xhigh", "medium", "low"})
REASONING_EFFORT_DEFAULT = "medium"
REASONING_EFFORT_JSON_CONFLICT = (
    "Remove reasoning_effort from chat template kwargs JSON and use the Thinking level Select"
)


def resolve_reasoning_effort(data: dict[str, object]) -> str:
    raw = data.get("reasoning_effort")
    if isinstance(raw, str) and raw in REASONING_EFFORT_VALUES:
        return raw
    return REASONING_EFFORT_DEFAULT


def chat_template_kwargs_has_reasoning_effort(ctk: str) -> bool:
    if not ctk or not ctk.strip():
        return False
    try:
        parsed: object = json.loads(ctk)
    except TypeError, ValueError:
        return False
    return isinstance(parsed, dict) and "reasoning_effort" in parsed


def merge_chat_template_kwargs(kwargs_json: str, reasoning_effort: str) -> str:
    if reasoning_effort not in REASONING_EFFORT_VALUES:
        reasoning_effort = REASONING_EFFORT_DEFAULT
    raw = kwargs_json.strip() if kwargs_json else ""
    parsed: object
    if not raw:
        parsed = {}
    else:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("chat_template_kwargs must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    if "reasoning_effort" in parsed:
        raise ValueError(REASONING_EFFORT_JSON_CONFLICT)
    parsed["reasoning_effort"] = reasoning_effort
    return json.dumps(parsed, separators=(",", ":"))
