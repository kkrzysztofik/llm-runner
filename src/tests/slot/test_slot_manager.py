"""Characterization tests for llama_manager.slot_manager.

These tests lock in the behaviour extracted from TUIApp controller methods:
add_slot_from_form, _upsert_profile_slot, _register_and_start_slot,
_remove_slot_runtime_state, _normalize_slot_port, _device_class_for_config,
and _gpu_index_for_config.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

from llama_manager.config import Config, ModelSlot, ServerConfig, SlotState
from llama_manager.gpu_stats import GPUStats
from llama_manager.log_buffer import LogBuffer
from llama_manager.slot_manager import (
    add_slot_from_form,
    device_class_for_config,
    gpu_index_for_config,
    normalize_slot_port,
    register_and_start_slot,
    remove_slot_runtime_state,
    upsert_profile_slot,
)


def _make_config(**overrides: object) -> ServerConfig:
    defaults: dict[str, object] = {
        "model": "/models/test.gguf",
        "alias": "test-slot",
        "device": "SYCL0",
        "port": 8080,
        "ctx_size": 4096,
        "ubatch_size": 512,
        "threads": 4,
        "server_bin": "dummy-llama-server",
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)  # type: ignore[arg-type]


def _make_state(**overrides: object) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "log_buffers": {},
        "server_processes": {},
        "slot_states": {},
        "unsaved_slots": set(),
        "slots": [],
    }
    defaults.update(overrides)
    return defaults


def _make_collector(_idx: int) -> Callable[[], dict[str, Any]]:
    return lambda: {"device": "test"}


# =============================================================================
# normalize_slot_port
# =============================================================================


class TestNormalizeSlotPort:
    def test_valid_port_unchanged(self) -> None:
        port, warning = normalize_slot_port("8080")
        assert port == 8080
        assert warning is None

    def test_port_below_range_resets_to_default(self) -> None:
        port, warning = normalize_slot_port("80")
        assert port == 8080
        assert warning == "Invalid port 80, using 8080"

    def test_port_above_range_resets_to_default(self) -> None:
        port, warning = normalize_slot_port("99999")
        assert port == 8080
        assert warning == "Invalid port 99999, using 8080"

    def test_non_numeric_port_resets_to_default(self) -> None:
        port, warning = normalize_slot_port("abc")
        assert port == 8080
        assert warning == "Invalid port, using 8080"

    def test_empty_port_resets_to_default(self) -> None:
        port, warning = normalize_slot_port("")
        assert port == 8080
        assert warning == "Invalid port, using 8080"


# =============================================================================
# device_class_for_config
# =============================================================================


class TestDeviceClassForConfig:
    def test_sycl_device(self) -> None:
        cfg = _make_config(device="SYCL0")
        assert device_class_for_config(cfg) == "sycl"

    def test_sycl_lowercase(self) -> None:
        cfg = _make_config(device="sycl1")
        assert device_class_for_config(cfg) == "sycl"

    def test_cuda_device(self) -> None:
        cfg = _make_config(device="CUDA0")
        assert device_class_for_config(cfg) == "cuda"

    def test_cpu_fallback(self) -> None:
        cfg = _make_config(device="CPU")
        assert device_class_for_config(cfg) == "cuda"


# =============================================================================
# gpu_index_for_config
# =============================================================================


class TestGpuIndexForConfig:
    def test_sycl_returns_one(self) -> None:
        cfg = _make_config(device="SYCL0")
        assert gpu_index_for_config(cfg) == 1

    def test_cuda_returns_zero(self) -> None:
        cfg = _make_config(device="CUDA0")
        assert gpu_index_for_config(cfg) == 0

    def test_custom_mapping(self) -> None:
        cfg = _make_config(device="SYCL0")
        mapping = {"sycl": 2, "cuda": 3}
        assert gpu_index_for_config(cfg, device_mapping=mapping) == 2

    def test_unknown_device_returns_zero(self) -> None:
        cfg = _make_config(device="CPU")
        assert gpu_index_for_config(cfg) == 0


# =============================================================================
# remove_slot_runtime_state
# =============================================================================


class TestRemoveSlotRuntimeState:
    def test_removes_all_state_for_alias(self) -> None:
        state = _make_state(
            log_buffers={"a": MagicMock(), "b": MagicMock()},
            server_processes={"a": MagicMock(), "b": MagicMock()},
            slot_states={"a": SlotState.RUNNING.value, "b": SlotState.IDLE.value},
            unsaved_slots={"a", "b"},
            slots=[
                ModelSlot(slot_id="a", model_path="/m/a.gguf", port=8080),
                ModelSlot(slot_id="b", model_path="/m/b.gguf", port=8081),
            ],
        )
        remove_slot_runtime_state("a", state)

        assert "a" not in state["log_buffers"]
        assert "a" not in state["server_processes"]
        assert "a" not in state["slot_states"]
        assert "a" not in state["unsaved_slots"]
        assert all(slot.slot_id != "a" for slot in state["slots"])

        # b should be untouched
        assert "b" in state["log_buffers"]
        assert "b" in state["server_processes"]
        assert "b" in state["slot_states"]
        assert "b" in state["unsaved_slots"]
        assert any(slot.slot_id == "b" for slot in state["slots"])

    def test_idempotent_when_alias_missing(self) -> None:
        state = _make_state()
        remove_slot_runtime_state("missing", state)
        assert state == _make_state()


# =============================================================================
# register_and_start_slot
# =============================================================================


class TestRegisterAndStartSlot:
    def test_registers_and_starts(self) -> None:
        cfg = _make_config(alias="summary")
        server_manager = MagicMock()
        server_manager.start_servers.return_value = [MagicMock()]

        state = _make_state()
        updated_state, messages = register_and_start_slot(cfg, server_manager, state)

        assert updated_state is state
        assert "summary" in state["log_buffers"]
        assert isinstance(state["log_buffers"]["summary"], LogBuffer)
        assert "summary" in state["unsaved_slots"]
        assert len(state["slots"]) == 1
        assert state["slots"][0].slot_id == "summary"
        assert state["server_processes"]["summary"] is server_manager.start_servers.return_value[0]
        assert state["slot_states"]["summary"] == SlotState.RUNNING.value
        assert any("launched successfully" in m for m in messages)

    def test_no_process_when_start_servers_returns_empty(self) -> None:
        cfg = _make_config(alias="summary")
        server_manager = MagicMock()
        server_manager.start_servers.return_value = []

        state = _make_state()
        updated_state, messages = register_and_start_slot(cfg, server_manager, state)

        assert "summary" not in state["server_processes"]
        assert state["slot_states"]["summary"] == SlotState.CRASHED.value
        assert any("failed to start" in m for m in messages)

    def test_second_registration_no_first_launch_message(self) -> None:
        cfg = _make_config(alias="summary")
        server_manager = MagicMock()
        server_manager.start_servers.return_value = [MagicMock()]

        state = _make_state(slot_states={"summary": SlotState.RUNNING.value})
        updated_state, messages = register_and_start_slot(cfg, server_manager, state)

        # Transition from RUNNING -> RUNNING is unmapped, so no message
        assert not any("launched successfully" in m for m in messages)


# =============================================================================
# upsert_profile_slot
# =============================================================================


class TestUpsertProfileSlot:
    def test_adds_new_slot_when_no_existing_device(self) -> None:
        cfg = _make_config(alias="new", device="SYCL0")
        configs: list[ServerConfig] = []
        gpu_indices: list[int] = []
        gpu_stats: list[GPUStats] = []
        server_manager = MagicMock()
        server_manager.start_servers.return_value = [MagicMock()]
        state = _make_state()

        success, messages, state = upsert_profile_slot(
            cfg,
            "profile-id",
            configs,
            gpu_indices,
            gpu_stats,
            server_manager,
            state,
            _make_collector,
        )

        assert success is True
        assert len(configs) == 1
        assert configs[0].alias == "new"
        assert len(gpu_indices) == 1
        assert gpu_indices[0] == 1
        assert len(gpu_stats) == 1
        assert "Added profile 'profile-id'" in messages[-1]

    def test_replaces_existing_slot_on_same_device(self) -> None:
        old_cfg = _make_config(alias="old", device="SYCL0")
        new_cfg = _make_config(alias="new", device="SYCL0", port=8090)
        configs = [old_cfg]
        gpu_indices = [1]
        gpu_stats = [GPUStats(1, collector=_make_collector(1))]
        server_manager = MagicMock()
        server_manager.shutdown_slot.return_value = True
        server_manager.start_servers.return_value = [MagicMock()]
        state = _make_state(
            log_buffers={"old": MagicMock()},
            server_processes={"old": MagicMock()},
            slot_states={"old": SlotState.RUNNING.value},
            unsaved_slots={"old"},
            slots=[ModelSlot(slot_id="old", model_path="/m/old.gguf", port=8080)],
        )

        success, messages, state = upsert_profile_slot(
            new_cfg,
            "profile-id",
            configs,
            gpu_indices,
            gpu_stats,
            server_manager,
            state,
            _make_collector,
        )

        assert success is True
        assert len(configs) == 1
        assert configs[0].alias == "new"
        assert "old" not in state["log_buffers"]
        assert "old" not in state["slot_states"]
        assert "Replaced 'old'" in messages[-1]
        server_manager.shutdown_slot.assert_called_once_with("old")

    def test_aborts_when_shutdown_fails(self) -> None:
        old_cfg = _make_config(alias="old", device="SYCL0")
        new_cfg = _make_config(alias="new", device="SYCL0")
        configs = [old_cfg]
        gpu_indices = [1]
        gpu_stats = [GPUStats(1, collector=_make_collector(1))]
        server_manager = MagicMock()
        server_manager.shutdown_slot.return_value = False
        state = _make_state()

        success, messages, state = upsert_profile_slot(
            new_cfg,
            "profile-id",
            configs,
            gpu_indices,
            gpu_stats,
            server_manager,
            state,
            _make_collector,
        )

        assert success is False
        assert configs[0].alias == "old"
        assert "shutdown verification failed" in messages[-1]
        server_manager.start_servers.assert_not_called()


# =============================================================================
# add_slot_from_form
# =============================================================================


class TestAddSlotFromForm:
    def test_rejects_empty_profile(self) -> None:
        success, messages, state = add_slot_from_form(
            values={"profile": "   ", "port": ""},
            config=Config(),
            configs=[],
            gpu_indices=[],
            gpu_stats=[],
            server_manager=MagicMock(),
            state=_make_state(),
            make_collector=_make_collector,
        )
        assert success is False
        assert any("Profile is required" in m for m in messages)

    @patch("llama_manager.slot_manager.resolve_profile_config")
    @patch("llama_manager.slot_manager.create_default_profile_registry")
    def test_creates_slot_from_profile(self, mock_registry_cls: MagicMock, mock_resolve: MagicMock) -> None:
        cfg = _make_config(alias="summary-fast", device="SYCL0", port=8090)
        mock_resolve.return_value = cfg
        registry = MagicMock()
        registry.profile_ids = ["summary-fast"]
        mock_registry_cls.return_value = registry

        server_manager = MagicMock()
        server_manager.start_servers.return_value = [MagicMock()]
        state = _make_state()
        configs: list[ServerConfig] = []

        success, messages, state = add_slot_from_form(
            values={"profile": "summary-fast", "port": "8090"},
            config=Config(),
            configs=configs,
            gpu_indices=[],
            gpu_stats=[],
            server_manager=server_manager,
            state=state,
            make_collector=_make_collector,
        )

        assert success is True
        assert len(configs) == 1
        assert configs[0].alias == "summary-fast"
        assert configs[0].port == 8090
        assert "summary-fast" in state["log_buffers"]
        mock_resolve.assert_called_once_with(
            registry, "summary-fast", override_config={"port": 8090}
        )

    @patch("llama_manager.slot_manager.resolve_profile_config")
    @patch("llama_manager.slot_manager.create_default_profile_registry")
    def test_invalid_port_includes_warning(self, mock_registry_cls: MagicMock, mock_resolve: MagicMock) -> None:
        cfg = _make_config(alias="summary-fast", device="SYCL0")
        mock_resolve.return_value = cfg
        registry = MagicMock()
        registry.profile_ids = ["summary-fast"]
        mock_registry_cls.return_value = registry

        server_manager = MagicMock()
        server_manager.start_servers.return_value = [MagicMock()]
        state = _make_state()
        configs: list[ServerConfig] = []

        success, messages, state = add_slot_from_form(
            values={"profile": "summary-fast", "port": "80"},
            config=Config(),
            configs=configs,
            gpu_indices=[],
            gpu_stats=[],
            server_manager=server_manager,
            state=state,
            make_collector=_make_collector,
        )

        assert success is True
        assert any("Invalid port 80, using 8080" in m for m in messages)
        mock_resolve.assert_called_once_with(
            registry, "summary-fast", override_config={"port": 8080}
        )

    @patch("llama_manager.slot_manager.resolve_profile_config")
    @patch("llama_manager.slot_manager.create_default_profile_registry")
    def test_rejects_unknown_profile(self, mock_registry_cls: MagicMock, mock_resolve: MagicMock) -> None:
        mock_resolve.side_effect = ValueError("unknown")
        registry = MagicMock()
        registry.profile_ids = ["summary-fast", "summary-balanced"]
        mock_registry_cls.return_value = registry

        state = _make_state()
        configs: list[ServerConfig] = []

        success, messages, state = add_slot_from_form(
            values={"profile": "unknown", "port": ""},
            config=Config(),
            configs=configs,
            gpu_indices=[],
            gpu_stats=[],
            server_manager=MagicMock(),
            state=state,
            make_collector=_make_collector,
        )

        assert success is False
        assert any("Unknown profile 'unknown'" in m for m in messages)
        assert len(configs) == 0
