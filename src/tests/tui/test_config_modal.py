"""Tests for ConfigModal — global config editor for the TUI dashboard.

Covers:
- Reading input widget values with whitespace stripping
- Save button returns ConfigPayload with collected values
- Save & Restart sets restart=True
- Cancel dismisses with None
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select

from llama_cli.tui.components.config_modal import ConfigModal, ConfigPayload
from llama_manager.config.defaults import (
    BuildPipelineConfig,
    Config,
    DeploymentConfig,
    PathsConfig,
    ServerDefaultsConfig,
    SmokeConfig,
)

# Explicit mapping from test's default_* field names to server_defaults attrs.
_DEFAULT_TO_SERVER_DEFAULTS: dict[str, str] = {
    "default_profile_port": "port",
    "default_profile_ctx_size": "ctx_size",
    "default_profile_ubatch_size": "ubatch_size",
    "default_profile_threads": "threads",
    "default_profile_n_gpu_layers": "n_gpu_layers_profile",
    "default_bind_address": "bind_address",
    "default_batch_size": "batch_size",
    "default_poll_ms": "poll_ms",
    "default_n_predict": "n_predict",
    "default_parallel": "parallel",
    "default_threads_batch": "threads_batch",
    "default_profile_cache_type_k": "cache_type_k",
    "default_profile_cache_type_v": "cache_type_v",
    "default_reasoning_mode": "reasoning_mode",
    "default_reasoning_format": "reasoning_format",
    "default_reasoning_budget": "reasoning_budget",
    "default_use_jinja": "use_jinja",
    "default_profile_chat_template_kwargs": "chat_template_kwargs",
    "default_mmproj": "mmproj",
    "default_spec_type": "spec_type",
    "default_spec_ngram_size_n": "spec_ngram_size_n",
    "default_draft_min": "draft_min",
    "default_draft_max": "draft_max",
    "default_spec_draft_n_max": "spec_draft_n_max",
    "default_spec_draft_p_min": "spec_draft_p_min",
    "default_spec_draft_cache_type_k": "spec_draft_cache_type_k",
    "default_spec_draft_cache_type_v": "spec_draft_cache_type_v",
    "default_spec_draft_device": "spec_draft_device",
}

# Fields that belong in each sub-dataclass (after stripping prefix).
_PATHS_FIELDS = frozenset(
    {
        "llama_cpp_root",
        "models_dir",
        "llama_server_bin_intel",
        "llama_server_bin_nvidia",
        "xdg_cache_base",
        "xdg_state_base",
        "xdg_data_base",
    }
)
_DEPLOYMENT_FIELDS = frozenset(
    {
        "host",
        "model_summary_balanced",
        "model_summary_fast",
        "model_qwen35",
        "model_qwen35_both",
        "summary_balanced_port",
        "summary_fast_port",
        "qwen35_port",
        "summary_balanced_chat_template_kwargs",
        "summary_fast_chat_template_kwargs",
    }
)
_BUILD_FIELDS = frozenset(
    {
        "source_flavor",
        "git_remote",
        "git_branch",
        "retry_attempts",
        "retry_delay",
        "max_reports",
        "output_truncate_bytes",
        "args_default",
        "toolchain_timeout_seconds",
    }
)
_SMOKE_FIELDS = frozenset(
    {
        "listen_timeout_s",
        "http_request_timeout_s",
        "inter_slot_delay_s",
        "max_tokens",
        "prompt",
        "skip_models_discovery",
        "api_key",
        "first_token_timeout_s",
        "total_chat_timeout_s",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "profile_staleness_days",
        "server_binary_version",
        "gguf_metadata_prefix_cap_bytes",
        "gguf_metadata_parse_timeout_s",
        "tui_launch_timeout_s",
        "tui_refresh_interval_ms",
        "probe_latency_threshold_s",
        "lock_stale_threshold_s",
        "log_file_level",
        "log_stderr_level",
    }
)


def _make_config(**overrides: object) -> Config:
    """Create a Config with nested sub-dataclasses, overridden by *overrides*.

    Test overrides use flat prefixed keys (e.g. ``build_git_remote``,
    ``smoke_listen_timeout_s``, ``default_batch_size``) which are routed
    into the correct sub-dataclass.  Sub-dataclasses are always created
    (with defaults) and overridden via ``dataclasses.replace()``.
    """
    paths_kw: dict[str, object] = {}
    deployment_kw: dict[str, object] = {}
    build_kw: dict[str, object] = {}
    smoke_kw: dict[str, object] = {}
    server_defaults_kw: dict[str, object] = {}
    top_level_kw: dict[str, Any] = {}

    # Test defaults — overridden by caller-supplied kwargs.
    defaults: dict[str, object] = {
        "llama_cpp_root": "/opt/llama.cpp",
        "models_dir": "/data/models",
        "llama_server_bin_intel": "/opt/llama.cpp/build/bin/llama-server",
        "llama_server_bin_nvidia": "/opt/llama.cpp/build_cuda/bin/llama-server",
        "host": "127.0.0.1",
        "build_git_remote": "https://github.com/ggerganov/llama.cpp",
        "build_git_branch": "master",
        "smoke_listen_timeout_s": 30,
        "smoke_http_request_timeout_s": 10,
        "smoke_first_token_timeout_s": 60,
        "smoke_total_chat_timeout_s": 300,
        "profile_staleness_days": 7,
        "build_retry_attempts": 3,
        "build_retry_delay": 5,
        "server_binary_version": "v1.0.0",
        "xdg_cache_base": "/tmp/llm-runner-cache",
        "xdg_state_base": "/tmp/llm-runner-state",
        "xdg_data_base": "/tmp/llm-runner-data",
        "build_max_reports": 5,
        "build_output_truncate_bytes": 1048576,
        "build_toolchain_timeout_seconds": 600,
        "smoke_inter_slot_delay_s": 2,
        "smoke_max_tokens": 16,
        "smoke_prompt": "hello",
        "smoke_skip_models_discovery": False,
        "smoke_api_key": "",
        "gguf_metadata_prefix_cap_bytes": 4096,
        "gguf_metadata_parse_timeout_s": 5,
        "tui_launch_timeout_s": 30,
        "tui_refresh_interval_ms": 250,
        "probe_latency_threshold_s": 1.0,
        "lock_stale_threshold_s": 300,
    }
    defaults.update(overrides)

    for k, v in defaults.items():
        # default_* -> server_defaults (with explicit name mapping)
        if k.startswith("default_") and k in _DEFAULT_TO_SERVER_DEFAULTS:
            server_defaults_kw[_DEFAULT_TO_SERVER_DEFAULTS[k]] = v
        # build_* -> build (strip prefix)
        elif k.startswith("build_"):
            stripped = k[len("build_") :]
            if stripped in _BUILD_FIELDS:
                build_kw[stripped] = v
            else:
                top_level_kw[k] = v
        # smoke_* -> smoke (strip prefix)
        elif k.startswith("smoke_"):
            stripped = k[len("smoke_") :]
            if stripped in _SMOKE_FIELDS:
                smoke_kw[stripped] = v
            else:
                top_level_kw[k] = v
        # Known paths fields
        elif k in _PATHS_FIELDS:
            paths_kw[k] = v
        # Known deployment fields
        elif k in _DEPLOYMENT_FIELDS:
            deployment_kw[k] = v
        # Known top-level fields
        elif k in _TOP_LEVEL_FIELDS:
            top_level_kw[k] = v
        # Anything else -> server_defaults (best-effort)
        else:
            server_defaults_kw[k] = v

    # Always create sub-dataclasses (with defaults), overriding via replace().
    cfg = Config(
        paths=replace(PathsConfig(), **paths_kw) if paths_kw else PathsConfig(),
        deployment=replace(DeploymentConfig(), **deployment_kw)
        if deployment_kw
        else DeploymentConfig(),
        build=replace(BuildPipelineConfig(), **build_kw) if build_kw else BuildPipelineConfig(),
        smoke=replace(SmokeConfig(), **smoke_kw) if smoke_kw else SmokeConfig(),
        server_defaults=replace(ServerDefaultsConfig(), **server_defaults_kw)
        if server_defaults_kw
        else ServerDefaultsConfig(),
        **top_level_kw,
    )
    return cfg


class ConfigModalHostApp(App[None]):
    """Minimal Textual app that hosts ConfigModal for testing."""

    pass


class TestConfigModalCollectValues:
    """Tests for ConfigModal._collect_values — reads input widget values."""

    @pytest.mark.anyio
    async def test_all_fields_collected(self) -> None:
        """_collect_values should read all Input widgets and strip whitespace."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            payload = modal._collect_values()

        assert payload.llama_cpp_root == "/opt/llama.cpp"
        assert payload.models_dir == "/data/models"
        assert payload.llama_server_bin_intel == "/opt/llama.cpp/build/bin/llama-server"
        assert payload.llama_server_bin_nvidia == "/opt/llama.cpp/build_cuda/bin/llama-server"
        assert payload.host == "127.0.0.1"
        assert payload.build_source_flavor == "upstream"
        assert payload.build_git_remote == "https://github.com/ggerganov/llama.cpp"
        assert payload.build_git_branch == "master"
        assert payload.smoke_listen_timeout_s == "30"
        assert payload.smoke_http_request_timeout_s == "10"
        assert payload.smoke_first_token_timeout_s == "60"  # noqa: S105
        assert payload.smoke_total_chat_timeout_s == "300"
        assert payload.restart is False

    @pytest.mark.anyio
    async def test_whitespace_stripped(self) -> None:
        """_collect_values should strip leading/trailing whitespace from inputs."""
        config = _make_config(llama_cpp_root="  /path/with/spaces  ")
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            input_widget = modal.query_one("#cfg-llama_cpp_root", Input)
            input_widget.value = "  /trimmed/path  "
            input_widget.blur()
            payload = modal._collect_values()

        assert payload.llama_cpp_root == "/trimmed/path"

    @pytest.mark.anyio
    async def test_empty_inputs_yield_empty_strings(self) -> None:
        """_collect_values should return empty strings when inputs are empty."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            input_widget = modal.query_one("#cfg-llama_cpp_root", Input)
            input_widget.value = ""
            input_widget.blur()
            payload = modal._collect_values()

        assert payload.llama_cpp_root == ""

    @pytest.mark.anyio
    async def test_numeric_fields_return_strings(self) -> None:
        """Numeric config fields should be returned as strings."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            payload = modal._collect_values()

        assert isinstance(payload.smoke_listen_timeout_s, str)
        assert payload.smoke_listen_timeout_s == "30"

    @pytest.mark.anyio
    async def test_profile_defaults_collected(self) -> None:
        """Profile/server default fields should be collected from the Config modal."""
        config = _make_config(
            default_batch_size=1024,
            default_poll_ms=0,
            default_parallel=4,
            default_profile_cache_type_k="f16",
            default_reasoning_mode="off",
            default_use_jinja=True,
            default_spec_type="ngram-mod",
        )
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            payload = modal._collect_values()

        assert payload.default_batch_size == "1024"
        assert payload.default_poll_ms == "0"
        assert payload.default_parallel == "4"
        assert payload.default_profile_cache_type_k == "f16"
        assert payload.default_reasoning_mode == "off"
        assert payload.default_use_jinja is True
        assert payload.default_spec_type == "ngram-mod"


class TestConfigModalSave:
    """Tests for ConfigModal save button — returns ConfigPayload."""

    def test_save_button_returns_payload(self) -> None:
        """Save button should return ConfigPayload with collected values."""
        config = _make_config()
        modal = ConfigModal(config)
        payload = ConfigPayload(llama_cpp_root="/saved")
        modal._collect_values = MagicMock(return_value=payload)  # type: ignore[method-assign]
        modal.dismiss = MagicMock()  # type: ignore[method-assign]

        modal.on_button_pressed(
            cast(Any, SimpleNamespace(button=SimpleNamespace(id="save-config")))
        )

        modal.dismiss.assert_called_once_with(payload)  # type: ignore[attr-defined]

    @pytest.mark.anyio
    async def test_save_populates_all_fields(self) -> None:
        """Save should populate all ConfigPayload fields via _collect_values."""
        config = _make_config(llama_cpp_root="/custom/root")
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            payload = modal._collect_values()

        assert payload.llama_cpp_root == "/custom/root"
        assert payload.models_dir == "/data/models"
        assert payload.host == "127.0.0.1"
        assert payload.restart is False

    @pytest.mark.anyio
    async def test_save_with_modified_values(self) -> None:
        """Save should reflect modified input values."""
        config = _make_config(models_dir="/original/models")
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            models_input = modal.query_one("#cfg-models_dir", Input)
            models_input.value = "/new/models/path"
            models_input.blur()

            payload = modal._collect_values()

        assert payload.models_dir == "/new/models/path"


class TestConfigModalSaveRestart:
    """Tests for Save & Restart button — sets restart=True."""

    @pytest.mark.anyio
    async def test_save_restart_sets_flag(self) -> None:
        """Save & Restart should set restart=True on the payload."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            values = modal._collect_values()
            values.restart = True

        assert values.restart is True
        assert values.llama_cpp_root == "/opt/llama.cpp"

    @pytest.mark.anyio
    async def test_save_restart_preserves_values(self) -> None:
        """Save & Restart should preserve all collected values."""
        config = _make_config(host="0.0.0.0")
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            values = modal._collect_values()
            values.restart = True

        assert values.restart is True
        assert values.host == "0.0.0.0"


class TestConfigModalCancel:
    """Tests for Cancel button — dismisses with None."""

    def test_cancel_button_dismisses(self) -> None:
        """Cancel button should dismiss the modal."""
        config = _make_config()
        modal = ConfigModal(config)
        modal.dismiss = MagicMock()  # type: ignore[method-assign]

        modal.on_button_pressed(
            cast(Any, SimpleNamespace(button=SimpleNamespace(id="cancel-config")))
        )

        modal.dismiss.assert_called_once_with(None)  # type: ignore[attr-defined]

    def test_escape_key_dismisses(self) -> None:
        """Escape key should dismiss via action_cancel."""
        binding = next(
            binding
            for binding in (cast(Any, item) for item in ConfigModal.BINDINGS)
            if binding.key == "escape"
        )

        assert binding.action == "cancel"

    def test_ctrl_c_dismisses(self) -> None:
        """Ctrl+C should dismiss via action_cancel."""
        binding = next(
            binding
            for binding in (cast(Any, item) for item in ConfigModal.BINDINGS)
            if binding.key == "ctrl+c"
        )

        assert binding.action == "cancel"

    def test_action_cancel_calls_dismiss_none(self) -> None:
        """action_cancel should call self.dismiss(None)."""
        config = _make_config()
        modal = ConfigModal(config)
        modal.dismiss = MagicMock()  # type: ignore[method-assign]

        modal.action_cancel()

        modal.dismiss.assert_called_once_with(None)  # type: ignore[attr-defined]


class TestConfigModalComposition:
    """Tests for ConfigModal screen composition."""

    @pytest.mark.anyio
    async def test_modal_has_title(self) -> None:
        """ConfigModal should compose with a title label."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            title_widget = modal.query_one("Label#config-title")
            assert title_widget is not None

    @pytest.mark.anyio
    async def test_modal_has_all_input_fields(self) -> None:
        """ConfigModal should compose Input widgets for all config fields."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            inputs = list(modal.query(Input))
            assert len(inputs) >= 10  # At least 10 input fields

    @pytest.mark.anyio
    async def test_modal_has_three_action_buttons(self) -> None:
        """ConfigModal should compose Cancel, Save, and Save & Restart buttons."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            cancel = modal.query_one("#cancel-config", Button)
            save = modal.query_one("#save-config", Button)
            save_restart = modal.query_one("#save-restart-config", Button)

            assert cancel is not None
            assert save is not None
            assert save_restart is not None

    @pytest.mark.anyio
    async def test_source_flavor_uses_select(self) -> None:
        """Source flavor should be a Select with upstream and beellama choices."""
        config = _make_config(build_source_flavor="beellama")
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            flavor_select = modal.query_one("#cfg-build_source_flavor", Select)

        assert flavor_select.value == "beellama"
        assert {option[1] for option in flavor_select._options} == {
            "upstream",
            "beellama",
        }

    @pytest.mark.anyio
    async def test_first_field_has_focus(self) -> None:
        """on_mount should focus the first input field."""
        config = _make_config()
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            focused = modal.focused
            assert focused is not None
            assert focused.id == "cfg-llama_cpp_root"
