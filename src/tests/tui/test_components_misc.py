"""Unit tests for smaller TUI component modules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.widgets import Input, Select

from llama_cli.tui.components import alerts, panels
from llama_cli.tui.components.modal import AddSlotModal
from llama_cli.tui.components.profile_status import ProfileStatusPanelRenderer


def test_alerts_module_exports_expected_symbols() -> None:
    assert "ProfileStatusPanelRenderer" in alerts.__all__
    assert "SystemHealthRenderer" in alerts.__all__
    assert alerts.ProfileStatusPanelRenderer is ProfileStatusPanelRenderer


def test_panels_module_exports_expected_symbols() -> None:
    assert "SlotStatusResolver" in panels.__all__
    assert "GPUStatsPanel" in panels.__all__
    assert hasattr(panels, "ServerColumnPanel")


def test_profile_status_renderer_returns_none_when_empty() -> None:
    renderer = ProfileStatusPanelRenderer()
    assert renderer.render({}, {}) is None


def test_profile_status_renderer_renders_all_status_variants() -> None:
    renderer = ProfileStatusPanelRenderer()
    panel = renderer.render(
        profile_status={
            "slot-a": "running",
            "slot-b": "done",
            "slot-c": "failed",
            "slot-d": "custom-status",
        },
        profile_flavor={"slot-a": "fast", "slot-b": "balanced", "slot-c": "deep"},
    )

    assert panel is not None
    assert panel.title == "Profile Status"
    body = panel.renderable.plain
    assert "Profiling slot-a: fast [running...]" in body
    assert "Profile slot-b: balanced [done]" in body
    assert "Profile slot-c: deep [failed]" in body
    assert "Profile slot-d: unknown [custom-status]" in body


def test_add_slot_modal_rejects_empty_profile_options() -> None:
    with pytest.raises(ValueError, match="profile_options must not be empty"):
        AddSlotModal([])


def test_add_slot_modal_collect_values_valid_and_strips_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value=" 8081 ")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    values = modal._collect_values()

    assert values == {"profile": "qwen", "port": "8081"}


def test_add_slot_modal_collect_values_blank_profile_maps_to_empty() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value=Select.BLANK)
    port = SimpleNamespace(value="")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    values = modal._collect_values()

    assert values == {"profile": "", "port": ""}


def test_add_slot_modal_collect_values_rejects_non_numeric_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value="abc")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    modal.notify = MagicMock()  # type: ignore[method-assign]

    assert modal._collect_values() is None
    modal.notify.assert_called_once_with("Port must be a number", severity="error")


def test_add_slot_modal_collect_values_rejects_out_of_range_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value="70000")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    modal.notify = MagicMock()  # type: ignore[method-assign]

    assert modal._collect_values() is None
    modal.notify.assert_called_once_with("Port must be 1-65535", severity="error")


def test_add_slot_modal_action_cancel_dismisses_none() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]

    modal.action_cancel()

    modal.dismiss.assert_called_once_with(None)


def test_add_slot_modal_on_mount_focuses_profile_select() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    select_widget = MagicMock()
    modal.query_one = MagicMock(return_value=select_widget)  # type: ignore[method-assign]

    modal.on_mount()

    select_widget.focus.assert_called_once_with()
    modal.query_one.assert_called_once_with("#slot-profile", Select)


def test_add_slot_modal_on_button_pressed_paths() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]

    modal.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="cancel-slot")))
    modal.dismiss.assert_called_once_with(None)

    modal.dismiss.reset_mock()
    modal._collect_values = MagicMock(return_value={"profile": "qwen", "port": "8080"})  # type: ignore[method-assign]
    modal.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="submit-slot")))
    modal.dismiss.assert_called_once_with({"profile": "qwen", "port": "8080"})

    modal.dismiss.reset_mock()
    modal._collect_values = MagicMock(return_value=None)  # type: ignore[method-assign]
    modal.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="submit-slot")))
    modal.dismiss.assert_not_called()


def test_add_slot_modal_on_input_submitted_only_for_slot_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]
    modal._collect_values = MagicMock(return_value={"profile": "qwen", "port": "8080"})  # type: ignore[method-assign]

    modal.on_input_submitted(SimpleNamespace(input=SimpleNamespace(id="other-input")))
    modal.dismiss.assert_not_called()

    modal.on_input_submitted(SimpleNamespace(input=SimpleNamespace(id="slot-port")))
    modal.dismiss.assert_called_once_with({"profile": "qwen", "port": "8080"})
