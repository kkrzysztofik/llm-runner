"""Load-mode resolution for llama-server model memory mapping."""

LOAD_MODE_VALUES = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})


def resolve_load_mode(data: dict[str, object]) -> str:
    raw = data.get("load_mode")
    if isinstance(raw, str) and raw in LOAD_MODE_VALUES:
        return raw
    if "mmap" not in data and "mlock" not in data:
        return "auto"
    mmap = bool(data.get("mmap", True))
    mlock = bool(data.get("mlock", False))
    if mmap and mlock:
        return "mmap+mlock"
    if mmap:
        return "mmap"
    if mlock:
        return "mlock"
    return "none"
