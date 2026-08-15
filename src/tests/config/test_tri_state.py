from llama_manager.config.tri_state import (
    resolve_fit,
    resolve_reasoning_preserve,
    resolve_tri_state,
)


def test_resolve_tri_state_valid() -> None:
    assert resolve_tri_state({"fit": "on"}, "fit") == "on"
    assert resolve_tri_state({"fit": "off"}, "fit") == "off"
    assert resolve_tri_state({"fit": "auto"}, "fit") == "auto"


def test_resolve_tri_state_invalid_defaults_auto() -> None:
    assert resolve_tri_state({"fit": "bogus"}, "fit") == "auto"
    assert resolve_tri_state({"fit": 1}, "fit") == "auto"
    assert resolve_tri_state({}, "fit") == "auto"


def test_resolve_reasoning_preserve() -> None:
    assert resolve_reasoning_preserve({"reasoning_preserve": "on"}) == "on"
    assert resolve_reasoning_preserve({"reasoning_preserve": "invalid"}) == "auto"


def test_resolve_fit() -> None:
    assert resolve_fit({"fit": "off"}) == "off"
    assert resolve_fit({"fit": "yes"}) == "auto"
