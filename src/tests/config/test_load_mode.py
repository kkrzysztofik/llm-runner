from llama_manager.config.load_mode import resolve_load_mode


def test_resolve_explicit_load_mode() -> None:
    assert resolve_load_mode({"load_mode": "dio"}) == "dio"


def test_resolve_legacy_mmap_mlock() -> None:
    assert resolve_load_mode({"mmap": True, "mlock": False}) == "mmap"
    assert resolve_load_mode({"mmap": True, "mlock": True}) == "mmap+mlock"
    assert resolve_load_mode({"mmap": False, "mlock": True}) == "mlock"
    assert resolve_load_mode({"mmap": False, "mlock": False}) == "none"


def test_resolve_missing_defaults_auto() -> None:
    assert resolve_load_mode({}) == "auto"
