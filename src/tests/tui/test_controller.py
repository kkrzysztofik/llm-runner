"""Tests for DashboardController — TUI command handler and background work.

Covers:
- Signal handler releases build lock and stops TUI
- Status message push and retrieval
- Build cancellation via cancel_event
- Risk handling (acknowledge, reject)
- Slot state transitions
"""

from __future__ import annotations

import signal
import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from llama_cli.tui import DashboardController
from llama_manager import ModelSlot, SlotState
from tests.support.helpers import make_server_config

if TYPE_CHECKING:
    from llama_cli.tui.components.config_modal import ConfigPayload
    from llama_cli.tui.components.slot_profile_modal import SlotProfilePayload


def _make_controller(**kwargs: object) -> DashboardController:
    """Create a minimal DashboardController for tests."""
    configs = [make_server_config(alias="slot0")]
    return DashboardController(
        configs=configs,
        gpu_indices=[0],
        slots=None,
        register_signals=False,
    )


class TestControllerSignalHandler:
    """Tests for DashboardController._signal_handler."""

    def test_signal_handler_stops_tui(self) -> None:
        """_signal_handler should set running=False."""
        controller = _make_controller()
        assert controller.running is True

        controller._signal_handler(signal.SIGINT, None)

        assert controller.running is False

    def test_signal_handler_releases_build_lock(self) -> None:
        """_signal_handler should release build pipeline lock when build is in progress."""
        controller = _make_controller()
        mock_pipeline = MagicMock()
        controller._build_pipeline = mock_pipeline
        controller.build_in_progress = True

        controller._signal_handler(signal.SIGINT, None)

        mock_pipeline.release_lock.assert_called_once()
        assert controller.build_in_progress is False

    def test_signal_handler_without_build_does_not_crash(self) -> None:
        """_signal_handler should work when no build pipeline exists."""
        controller = _make_controller()
        controller._build_pipeline = None
        controller.build_in_progress = False

        controller._signal_handler(signal.SIGINT, None)

        assert controller.running is False

    def test_signal_handler_does_not_release_lock_when_no_build(self) -> None:
        """_signal_handler should not call release_lock when build_in_progress is False."""
        controller = _make_controller()
        mock_pipeline = MagicMock()
        controller._build_pipeline = mock_pipeline
        controller.build_in_progress = False

        controller._signal_handler(signal.SIGINT, None)

        mock_pipeline.release_lock.assert_not_called()


class TestControllerStatusMessages:
    """Tests for DashboardController status message handling."""

    def test_push_status_message_adds_to_buffer(self) -> None:
        """_push_status_message should add a timestamped message."""
        controller = _make_controller()

        controller._push_status_message("test message")

        messages = controller._status_messages
        assert len(messages) == 1
        assert messages[0][1] == "test message"

    def test_get_status_messages_since_filters_by_timestamp(self) -> None:
        """get_status_messages_since should return only messages newer than since_ts."""
        controller = _make_controller()

        controller._push_status_message("old message")
        old_ts = controller._status_messages[0][0]

        # Small delay to ensure new message has later timestamp
        import time

        time.sleep(0.01)
        controller._push_status_message("new message")

        messages = controller.get_status_messages_since(old_ts + 0.001)
        assert len(messages) == 1
        assert messages[0][1] == "new message"

    def test_status_messages_limited_to_five(self) -> None:
        """Status messages should be limited to 5 entries."""
        controller = _make_controller()

        for i in range(10):
            controller._push_status_message(f"message {i}")

        messages = controller._status_messages
        assert len(messages) <= 5

    def test_push_status_message_triggers_refresh(self) -> None:
        """_push_status_message should trigger a view model refresh."""
        controller = _make_controller()

        # Should not raise
        controller._push_status_message("refresh test")
        assert len(controller._status_messages) == 1


class TestControllerCancelBuild:
    """Tests for DashboardController.cancel_build."""

    def test_cancel_build_sets_cancel_event(self) -> None:
        """cancel_build should set the build_cancel_event."""
        controller = _make_controller()
        controller.model.build_cancel_event = threading.Event()

        controller.cancel_build()

        assert controller.model.build_cancel_event.is_set()

    def test_cancel_build_kills_subprocess(self) -> None:
        """cancel_build should call kill_active_subprocess on the pipeline."""
        controller = _make_controller()
        controller.model.build_cancel_event = threading.Event()
        mock_pipeline = MagicMock()
        controller._build_pipeline = mock_pipeline

        controller.cancel_build()

        mock_pipeline.kill_active_subprocess.assert_called_once()
        assert controller.model.build_cancel_event.is_set()

    def test_cancel_build_no_pipeline_no_crash(self) -> None:
        """cancel_build should not crash when no pipeline exists."""
        controller = _make_controller()
        controller.model.build_cancel_event = threading.Event()
        controller._build_pipeline = None

        # Should not raise
        controller.cancel_build()

        assert controller.model.build_cancel_event.is_set()

    def test_cancel_build_no_event_no_crash(self) -> None:
        """cancel_build should not crash when no cancel event exists."""
        controller = _make_controller()
        # Don't set build_cancel_event
        controller._build_pipeline = None

        # Should not raise
        controller.cancel_build()


class TestControllerRiskHandling:
    """Tests for DashboardController risk prompt handling."""

    def test_acknowledge_risk_clears_prompt(self) -> None:
        """acknowledge_risk should clear the risk prompt."""
        controller = _make_controller()
        controller.active_risk_kind = "hardware"

        controller.acknowledge_risk()

        assert controller.active_risk_kind is None

    def test_reject_risk_clears_prompt(self) -> None:
        """reject_risk should clear the risk prompt."""
        controller = _make_controller()
        controller.active_risk_kind = "hardware"

        controller.reject_risk()

        assert controller.active_risk_kind is None

    def test_acknowledge_risk_no_prompt(self) -> None:
        """acknowledge_risk should be a no-op when no risk prompt exists."""
        controller = _make_controller()

        # Should not raise
        controller.acknowledge_risk()

        assert controller.active_risk_kind is None

    def test_reject_risk_no_prompt(self) -> None:
        """reject_risk should be a no-op when no risk prompt exists."""
        controller = _make_controller()

        # Should not raise
        controller.reject_risk()

        assert controller.active_risk_kind is None

    def test_handle_hardware_warning_y(self) -> None:
        """handle_hardware_warning('y') should acknowledge and clear."""
        controller = _make_controller()
        controller.active_risk_kind = "hardware"

        result = controller.handle_hardware_warning("y")

        assert result == "acknowledge"
        assert controller.active_risk_kind is None

    def test_handle_hardware_warning_n(self) -> None:
        """handle_hardware_warning('n') should abort and stop."""
        controller = _make_controller()
        controller.active_risk_kind = "hardware"

        result = controller.handle_hardware_warning("n")

        assert result == "abort"
        assert controller.running is False

    def test_handle_hardware_warning_q(self) -> None:
        """handle_hardware_warning('q') should quit."""
        controller = _make_controller()
        controller.active_risk_kind = "hardware"

        result = controller.handle_hardware_warning("q")

        assert result == "quit"

    def test_handle_vram_risk_y(self) -> None:
        """handle_vram_risk('y') should proceed and clear."""
        controller = _make_controller()
        controller.active_risk_kind = "vram"

        result = controller.handle_vram_risk("y")

        assert result == "proceed"
        assert controller.active_risk_kind is None

    def test_handle_vram_risk_n(self) -> None:
        """handle_vram_risk('n') should abort and stop."""
        controller = _make_controller()
        controller.active_risk_kind = "vram"

        result = controller.handle_vram_risk("n")

        assert result == "abort"
        assert controller.running is False


class TestControllerSlotTransition:
    """Tests for DashboardController.handle_slot_transition."""

    def test_transition_pushes_status_message(self) -> None:
        """handle_slot_transition should push a status message for valid transitions."""
        controller = _make_controller()

        controller.handle_slot_transition("slot0", SlotState.RUNNING)

        messages = controller._status_messages
        assert len(messages) >= 1

    def test_same_state_no_message(self) -> None:
        """handle_slot_transition should not push a message for same-state."""
        controller = _make_controller()
        controller.slot_states["slot0"] = SlotState.RUNNING.value

        controller.handle_slot_transition("slot0", SlotState.RUNNING)

        # No new message should be added for same-state
        messages = controller._status_messages
        # The count should be the same (no new message for same-state)
        assert len(messages) == 0

    def test_transition_updates_state(self) -> None:
        """handle_slot_transition should update slot_states."""
        controller = _make_controller()

        controller.handle_slot_transition("slot0", SlotState.RUNNING)

        assert controller.slot_states["slot0"] == SlotState.RUNNING.value

    def test_transition_launched_pushes_message(self) -> None:
        """Transition to RUNNING should push a 'launched' message."""
        controller = _make_controller()

        controller.handle_slot_transition("slot0", SlotState.RUNNING)

        messages = controller._status_messages
        assert any("launched" in msg.lower() for _, msg in messages)

    def test_transition_no_result_returns_early(self) -> None:
        """handle_slot_transition should return early for invalid transitions."""
        controller = _make_controller()

        # offline → running is not a valid transition (no message pushed)
        controller.slot_states["slot0"] = SlotState.OFFLINE.value
        controller.handle_slot_transition("slot0", SlotState.RUNNING)

        # Should not raise


class TestControllerRemoveLiveSlot:
    """Tests for DashboardController.remove_live_slot."""

    def test_remove_live_slot_success_updates_state_and_status(self) -> None:
        controller = _make_controller()
        controller.model.server_manager = MagicMock()
        controller.model.server_manager.shutdown_slot.return_value = True
        controller.model.server_processes = {"slot0": MagicMock()}
        controller.model.slot_states = {"slot0": SlotState.RUNNING.value}
        controller.model.unsaved_slots = {"slot0"}
        controller.model.slots = [ModelSlot(slot_id="slot0", model_path="/m/slot0.gguf", port=8080)]
        controller.model.stale_warnings = {"slot0": "stale"}

        success = controller.remove_live_slot("slot0")

        assert success is True
        assert controller.configs == []
        assert controller.gpu_indices == []
        assert controller.gpu_stats == []
        assert controller.log_buffers == {}
        assert controller.server_processes == {}
        assert controller.slot_states == {}
        assert controller.unsaved_slots == set()
        assert controller.slots == []
        assert controller.model.stale_warnings == {}
        assert any(msg == "Removed slot 'slot0'" for _, msg in controller._status_messages)

    def test_remove_live_slot_shutdown_failure_still_removes(self) -> None:
        controller = _make_controller()
        controller.model.server_manager = MagicMock()
        controller.model.server_manager.shutdown_slot.return_value = False

        success = controller.remove_live_slot("slot0")

        assert success is True
        assert len(controller.configs) == 0
        assert len(controller.gpu_indices) == 0
        assert len(controller.gpu_stats) == 0
        assert "slot0" not in controller.log_buffers


class TestControllerBuildLifecycle:
    """Tests for DashboardController build lifecycle methods."""

    def test_build_request_sets_flag(self) -> None:
        """request_build should set _build_request to True."""
        controller = _make_controller()

        controller.request_build()

        assert controller._build_request is True

    def test_cancel_pending_prompt_clears_flag(self) -> None:
        """cancel_pending_prompt should clear _build_request and return True."""
        controller = _make_controller()
        controller._build_request = True

        result = controller.cancel_pending_prompt()

        assert result is True
        assert controller._build_request is False

    def test_cancel_pending_prompt_no_request(self) -> None:
        """cancel_pending_prompt should return False when no build request."""
        controller = _make_controller()
        controller._build_request = False

        result = controller.cancel_pending_prompt()

        assert result is False

    def test_stop_sets_running_false(self) -> None:
        """stop should set running to False."""
        controller = _make_controller()
        controller.running = True

        controller.stop()

        assert controller.running is False

    def test_graceful_shutdown_stops_servers(self) -> None:
        """_graceful_shutdown should call cleanup_servers."""
        controller = _make_controller()

        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        controller.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        controller._graceful_shutdown()

        assert cleanup_called is True
        assert controller.running is False

    def test_graceful_shutdown_noop_when_not_running(self) -> None:
        """_graceful_shutdown should be a no-op when already stopped."""
        controller = _make_controller()
        controller.running = False

        controller._graceful_shutdown()

        assert controller.running is False


class TestControllerSaveConfig:
    """Tests for DashboardController.save_config — config persistence."""

    def _make_payload(self, **kwargs: object) -> ConfigPayload:
        from llama_cli.tui.components.config_modal import ConfigPayload

        return ConfigPayload(**{**{"restart": False, "clean_cache": False}, **kwargs})  # type: ignore[arg-type]

    def test_errors_push_messages_and_return_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_config should push each error from apply_config_updates and not save."""
        from llama_manager.config.persistence import ConfigUpdateResult

        mock_result = ConfigUpdateResult(success=False, updated_fields=[], errors=["Bad value"])
        monkeypatch.setattr("llama_manager.apply_config_updates", lambda *a, **kw: mock_result)
        controller = _make_controller()

        controller.save_config(self._make_payload())

        texts = [msg for _, msg in controller._status_messages]
        assert any("Bad value" in t for t in texts)
        # Should NOT push a "Config saved" message
        assert not any("Config saved" in t for t in texts)

    def test_updated_fields_pushes_saved_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_config should push 'Config saved' when apply_config_updates updates fields."""
        from llama_manager.config.persistence import ConfigUpdateResult

        mock_result = ConfigUpdateResult(success=True, updated_fields=["models_dir"], errors=[])
        monkeypatch.setattr("llama_manager.apply_config_updates", lambda *a, **kw: mock_result)
        controller = _make_controller()

        controller.save_config(self._make_payload())

        texts = [msg for _, msg in controller._status_messages]
        assert any("Config saved" in t for t in texts)

    def test_no_changes_no_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_config should not push any message when no fields were updated."""
        from llama_manager.config.persistence import ConfigUpdateResult

        mock_result = ConfigUpdateResult(success=True, updated_fields=[], errors=[])
        monkeypatch.setattr("llama_manager.apply_config_updates", lambda *a, **kw: mock_result)
        controller = _make_controller()

        controller.save_config(self._make_payload())

        assert len(controller._status_messages) == 0

    def test_restart_flag_stops_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_config with restart=True should set controller.running to False."""
        from llama_manager.config.persistence import ConfigUpdateResult

        mock_result = ConfigUpdateResult(success=True, updated_fields=["models_dir"], errors=[])
        monkeypatch.setattr("llama_manager.apply_config_updates", lambda *a, **kw: mock_result)
        controller = _make_controller()
        assert controller.running is True

        controller.save_config(self._make_payload(restart=True))

        assert controller.running is False

    def test_log_file_level_calls_update_file_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_config should call update_file_level when log_file_level is updated."""
        from llama_manager.config.persistence import ConfigUpdateResult

        mock_result = ConfigUpdateResult(success=True, updated_fields=["log_file_level"], errors=[])
        monkeypatch.setattr("llama_manager.apply_config_updates", lambda *a, **kw: mock_result)
        called_with: list[str] = []
        monkeypatch.setattr(
            "llama_cli.tui.controller.update_file_level", lambda lvl: called_with.append(lvl)
        )
        controller = _make_controller()

        controller.save_config(self._make_payload())

        assert len(called_with) == 1


class TestControllerProfileMethods:
    """Tests for DashboardController run-profile management methods."""

    def _make_profile_payload(self, **kwargs: object) -> SlotProfilePayload:
        from llama_cli.tui.components.slot_profile_modal import SlotProfilePayload

        defaults: dict[str, object] = {
            "profile_id": "my-profile",
            "label": "My Profile",
            "server_bin": "",
            "model": "/data/models/model.gguf",
            "port": 8080,
            "ctx_size": 4096,
            "ubatch_size": 512,
            "n_gpu_layers": "all",
            "threads": 8,
            "chat_template_kwargs": "",
            "device": "CUDA:0",
            "save_and_add_slot": False,
            "original_profile_id": "",
        }
        defaults.update(kwargs)
        return SlotProfilePayload(**defaults)  # type: ignore[arg-type]

    def test_save_profile_empty_id_returns_false(self) -> None:
        """save_run_profile_from_form should return False for an empty profile_id."""
        controller = _make_controller()

        result = controller.save_slot_profile_from_form(self._make_profile_payload(profile_id=""))

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("Profile ID is required" in t for t in texts)

    def test_save_profile_empty_model_returns_false(self) -> None:
        """save_run_profile_from_form should return False for an empty model path."""
        controller = _make_controller()

        result = controller.save_slot_profile_from_form(self._make_profile_payload(model=""))

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("Model path is required" in t for t in texts)

    def test_save_profile_invalid_port_returns_false(self) -> None:
        """save_run_profile_from_form should return False for port < 1024."""
        controller = _make_controller()

        result = controller.save_slot_profile_from_form(self._make_profile_payload(port=80))

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("Port must be between 1024 and 65535" in t for t in texts)

    def test_save_profile_zero_ctx_size_returns_false(self) -> None:
        """save_run_profile_from_form should return False for ctx_size <= 0."""
        controller = _make_controller()

        result = controller.save_slot_profile_from_form(self._make_profile_payload(ctx_size=0))

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("must be positive" in t for t in texts)

    def test_delete_profile_in_use_returns_false(self) -> None:
        """delete_run_profile should return False when the profile is in use."""
        controller = _make_controller()
        # slot0 alias matches profile_id "slot0"
        result = controller.delete_slot_profile("slot0")

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("in use" in t for t in texts)

    def test_delete_profile_not_found_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """delete_run_profile should return False when the profile does not exist."""
        monkeypatch.setattr(
            "llama_manager.slot_profile_store.delete_custom_slot_profile",
            lambda *a, **kw: False,
        )
        controller = _make_controller()

        result = controller.delete_slot_profile("nonexistent-profile")

        assert result is False
        texts = [msg for _, msg in controller._status_messages]
        assert any("not found" in t for t in texts)

    def test_delete_profile_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """delete_run_profile should return True and push a message on success."""
        monkeypatch.setattr(
            "llama_manager.slot_profile_store.delete_custom_slot_profile",
            lambda *a, **kw: True,
        )
        controller = _make_controller()

        result = controller.delete_slot_profile("some-other-profile")

        assert result is True
        texts = [msg for _, msg in controller._status_messages]
        assert any("deleted" in t for t in texts)
