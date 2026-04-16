"""Tests for TUI application module (tui_app.py).

Covers:
- TUIApp initialization
- Layout building and rendering
- Build pipeline integration
- Risk acknowledgment
- Status panel building
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.tui_app import TUIApp
from llama_manager import ServerConfig
from llama_manager.build_pipeline import BuildBackend, BuildProgress, BuildResult


def _make_config(
    alias: str = "test",
    port: int = 8080,
    device: str = "CUDA",
) -> ServerConfig:
    """Helper to create a ServerConfig for tests."""
    return ServerConfig(
        model="/path/to/model.gguf",
        alias=alias,
        device=device,
        port=port,
        ctx_size=4096,
        ubatch_size=512,
        threads=4,
    )


def _layout_keys(layout: Any) -> list[str]:
    """Extract layout keys for assertions (avoids pyright 'in' operator issues)."""
    return list(getattr(layout, "names", []))


# =============================================================================
# TUIApp initialization
# =============================================================================


class TestTUIAppInit:
    """Tests for TUIApp.__init__."""

    def test_init_basic(self) -> None:
        """TUIApp should initialize with basic config."""
        configs = [_make_config()]
        app = TUIApp(configs=configs, gpu_indices=[0])

        assert len(app.configs) == 1
        assert app.gpu_indices == [0]
        assert app.running is True
        assert app.width == 80
        assert app.height == 24
        assert app.launch_result is None
        assert app.risk_panel is None
        assert app.risks_acknowledged is False

    def test_init_multiple_configs(self) -> None:
        """TUIApp should create log buffers for all configs."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        app = TUIApp(configs=configs, gpu_indices=[0, 1])

        assert len(app.log_buffers) == 2
        assert "model1" in app.log_buffers
        assert "model2" in app.log_buffers

    def test_init_no_slots(self) -> None:
        """TUIApp should initialize with empty slots when not provided."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        assert app.slots == []

    def test_init_with_slots(self) -> None:
        """TUIApp should accept slots parameter."""
        from llama_manager import ModelSlot

        slots = [ModelSlot(slot_id="test", model_path="/path/to/model.gguf", port=8080)]
        app = TUIApp(configs=[_make_config()], gpu_indices=[0], slots=slots)
        assert len(app.slots) == 1

    def test_init_default_width_height(self) -> None:
        """TUIApp should have default 80x24 dimensions."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        assert app.width == 80
        assert app.height == 24


# =============================================================================
# stop
# =============================================================================


class TestTUIAppStop:
    """Tests for TUIApp.stop."""

    def test_stop_sets_running_false(self) -> None:
        """stop should set running to False."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.running = True
        app.stop()
        assert app.running is False


# =============================================================================
# on_resize
# =============================================================================


class TestTUIAppOnResize:
    """Tests for TUIApp.on_resize."""

    def test_on_resize_updates_dimensions(self) -> None:
        """on_resize should update width and height."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        mock_event = MagicMock()
        mock_event.width = 120
        mock_event.height = 40
        app.on_resize(mock_event)  # type: ignore[arg-type]
        assert app.width == 120
        assert app.height == 40


# =============================================================================
# build_layout
# =============================================================================


class TestTUIAppBuildLayout:
    """Tests for TUIApp.build_layout."""

    def test_build_layout_with_wide_screen(self) -> None:
        """build_layout should split content into row when width >= 80."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 120
        layout = app.build_layout()
        keys = _layout_keys(layout)
        assert "main" in keys
        assert "alerts" in keys
        assert "content" in keys
        assert "left" in keys
        assert "right" in keys

    def test_build_layout_with_narrow_screen(self) -> None:
        """build_layout should split content into column when width < 80."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 60
        layout = app.build_layout()
        keys = _layout_keys(layout)
        assert "left" in keys
        assert "right" in keys

    def test_build_layout_has_alerts_panel(self) -> None:
        """build_layout should create alerts panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        layout = app.build_layout()
        keys = _layout_keys(layout)
        assert "alerts" in keys


# =============================================================================
# render
# =============================================================================


class TestTUIAppRender:
    """Tests for TUIApp.render."""

    def test_render_no_configs(self) -> None:
        """render should handle no configs gracefully."""
        app = TUIApp(configs=[], gpu_indices=[])
        layout = app.render()
        keys = _layout_keys(layout)
        assert "right" in keys

    def test_render_single_config(self) -> None:
        """render should render single config with placeholder."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        layout = app.render()
        keys = _layout_keys(layout)
        assert "left" in keys
        assert "right" in keys

    def test_render_two_configs(self) -> None:
        """render should render both configs side by side."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        app = TUIApp(configs=configs, gpu_indices=[0, 1])
        layout = app.render()
        keys = _layout_keys(layout)
        assert "left" in keys
        assert "right" in keys

    def test_render_with_status_panel(self) -> None:
        """render should include status panel in alerts."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.status_panel = MagicMock()
        layout = app.render()
        keys = _layout_keys(layout)
        assert "alerts" in keys


# =============================================================================
# _build_status_panel
# =============================================================================


class TestBuildStatusPanel:
    """Tests for TUIApp._build_status_panel."""

    def test_status_panel_success_clears_panel(self) -> None:
        """_build_status_panel should clear status panel on success."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="success")
        app._build_status_panel(launch_result)
        assert app.status_panel is None

    def test_status_panel_blocked(self) -> None:
        """_build_status_panel should create blocked panel."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(
            status="blocked",
            errors=MagicMock(errors=[]),
        )
        app._build_status_panel(launch_result)
        assert app.status_panel is not None

    def test_status_panel_degraded(self) -> None:
        """_build_status_panel should create degraded panel."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(
            status="degraded",
            launched=["slot1"],
            warnings=["partial success"],
        )
        app._build_status_panel(launch_result)
        assert app.status_panel is not None


# =============================================================================
# Risk panels
# =============================================================================


class TestRiskPanels:
    """Tests for risk panel building."""

    def test_build_risk_panel_required(self) -> None:
        """_build_risk_panel_required should set risk_panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_required()
        assert app.risk_panel is not None
        assert app.risks_acknowledged is False

    def test_build_risk_panel_acknowledged(self) -> None:
        """_build_risk_panel_acknowledged should set risk_panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_acknowledged()
        assert app.risk_panel is not None
        assert app.risks_acknowledged is True

    def test_update_risk_panel_state_with_risks(self) -> None:
        """_update_risk_panel_state should set acknowledged panel when has_risks."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._update_risk_panel_state(has_risks=True)
        assert app.risks_acknowledged is True

    def test_update_risk_panel_state_without_risks(self) -> None:
        """_update_risk_panel_state should clear risk_panel when no risks."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.risk_panel = MagicMock()
        app._update_risk_panel_state(has_risks=False)
        assert app.risk_panel is None
        assert app.risks_acknowledged is False


# =============================================================================
# _handle_build_progress
# =============================================================================


class TestHandleBuildProgress:
    """Tests for TUIApp._handle_build_progress."""

    def test_handle_progress_retry(self) -> None:
        """_handle_build_progress should create retrying status panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="retrying",
            message="Retrying...",
            progress_percent=50,
            retries_remaining=2,
        )
        app._handle_build_progress(progress)

        assert app._build_progress is progress
        assert app.status_panel is not None

    def test_handle_progress_failure(self) -> None:
        """_handle_build_progress should create failed status panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="failed",
            message="Build failed",
            progress_percent=0,
        )
        app._handle_build_progress(progress)

        assert app.status_panel is not None

    def test_handle_progress_success_clears(self) -> None:
        """_handle_build_progress should clear status panel on success."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_in_progress = True
        app.status_panel = MagicMock()

        progress = BuildProgress(
            stage="build",
            status="success",
            message="Build successful",
            progress_percent=100,
        )
        app._handle_build_progress(progress)

        assert app.status_panel is None


# =============================================================================
# _build_placeholder_panel
# =============================================================================


class TestBuildPlaceholderPanel:
    """Tests for TUIApp._build_placeholder_panel."""

    def test_placeholder_panel(self) -> None:
        """_build_placeholder_panel should create a dim placeholder."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        panel = app._build_placeholder_panel()
        assert panel is not None


# =============================================================================
# _handle_launch_result
# =============================================================================


class TestHandleLaunchResult:
    """Tests for TUIApp._handle_launch_result."""

    def test_handle_blocked_result(self, capsys) -> None:
        """_handle_launch_result should raise SystemExit for blocked result."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="blocked", errors=MagicMock(errors=[]))

        with pytest.raises(SystemExit):
            app._handle_launch_result(launch_result)

    def test_handle_degraded_result(self, capsys) -> None:
        """_handle_launch_result should print warning for degraded result."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="degraded", warnings=["slot1 blocked"])

        # Should not raise, just print warning
        app._handle_launch_result(launch_result)

        captured = capsys.readouterr()
        assert "degraded" in captured.err.lower()


# =============================================================================
# build_llama_cpp
# =============================================================================


class TestBuildLlamaCpp:
    """Tests for TUIApp.build_llama_cpp."""

    def test_build_llama_cpp_success(self, tmp_path: Path) -> None:
        """build_llama_cpp should return True on success."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.run.return_value = BuildResult(success=True)
            mock_pipeline.dry_run = False

            result = app.build_llama_cpp(backend="sycl", dry_run=False)

            assert result is True
            assert app._build_in_progress is False

    def test_build_llama_cpp_failure(self, tmp_path: Path) -> None:
        """build_llama_cpp should return False on failure."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.run.return_value = BuildResult(success=False, error_message="failed")
            mock_pipeline.dry_run = False

            result = app.build_llama_cpp(backend="sycl", dry_run=False)

            assert result is False
            assert app._build_in_progress is False

    def test_build_llama_cpp_dry_run(self, tmp_path: Path) -> None:
        """build_llama_cpp should set dry_run on pipeline."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.run.return_value = BuildResult(success=True)

            app.build_llama_cpp(backend="sycl", dry_run=True)

            assert mock_pipeline.dry_run is True

    def test_build_llama_cpp_cuda_backend(self, tmp_path: Path) -> None:
        """build_llama_cpp should use correct backend for CUDA."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.run.return_value = BuildResult(success=True)

            app.build_llama_cpp(backend="cuda", dry_run=False)

            # Verify pipeline was created with CUDA backend
            call_args = mock_pipeline.call_args
            build_config = call_args[0][0]
            assert build_config.backend == BuildBackend.CUDA


# =============================================================================
# _signal_handler
# =============================================================================


class TestSignalHandler:
    """Tests for signal handlers."""

    def test_signal_handler_stops(self) -> None:
        """_signal_handler should call stop()."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.running = True

        app._signal_handler(2, None)

        assert app.running is False

    def test_signal_handler_build_releases_lock(self) -> None:
        """_signal_handler_build should release build lock."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_in_progress = True

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.release_lock.return_value = None

            with pytest.raises(SystemExit) as exc_info:
                app._signal_handler_build(2, None)

            assert exc_info.value.code == 130
            mock_pipeline.release_lock.assert_called_once()
