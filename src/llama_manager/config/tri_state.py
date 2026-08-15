"""Tri-state resolution for auto/on/off llama-server options."""

TRI_STATE_VALUES = frozenset({"auto", "on", "off"})


def resolve_tri_state(data: dict[str, object], key: str, *, default: str = "auto") -> str:
    raw = data.get(key)
    if isinstance(raw, str) and raw in TRI_STATE_VALUES:
        return raw
    return default


def resolve_reasoning_preserve(data: dict[str, object]) -> str:
    return resolve_tri_state(data, "reasoning_preserve")


def resolve_fit(data: dict[str, object]) -> str:
    return resolve_tri_state(data, "fit")
