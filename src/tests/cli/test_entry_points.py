"""Tests for CLI entry points."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestPackageEntryPoint:
    """Tests for the llm-runner package entry point."""

    def test_cli_main_calls_main_with_sys_argv(self) -> None:
        """cli_main should pass sys.argv[1:] to main."""
        from llama_cli import server_runner

        test_args = ["llm-runner", "tui", "summary-balanced"]
        with (
            patch("sys.argv", test_args),
            patch.object(server_runner, "main", return_value=0) as mock_main,
            pytest.raises(SystemExit) as exc,
        ):
            server_runner.cli_main()

        assert exc.value.code == 0
        mock_main.assert_called_once_with(["tui", "summary-balanced"])


class TestRunTuiPrereqs:
    """Tests for TUI prerequisite validation in the canonical CLI module."""

    def test_check_prereqs_calls_require_executable(self) -> None:
        """_run_tui should validate the resolved Intel binary."""
        from llama_cli import server_runner

        server_cfg = MagicMock()
        server_cfg.server_bin = "/fake/intel-server"
        server_cfg.device = "SYCL0"
        server_cfg.port = 8080
        server_cfg.alias = "summary-balanced"
        server_cfg.model = "/fake/model.gguf"
        parsed = Namespace(tui_mode="summary-balanced", acknowledge_risky=False)

        with (
            patch.object(server_runner, "Config", return_value=MagicMock()),
            patch.object(
                server_runner,
                "_build_tui_mode_configs",
                return_value={
                    "summary-balanced": (
                        [server_cfg.server_bin],
                        [server_runner.INTEL_SERVER_NAME],
                        [server_cfg],
                        [1],
                    )
                },
            ),
            patch.object(server_runner, "require_executable", return_value=None) as mock_require,
            patch.object(server_runner, "_validate_tui_configs") as mock_validate,
            patch("llama_cli.tui.DashboardController") as mock_tui_app,
        ):
            mock_tui_app.return_value.run.return_value = None
            result = server_runner._run_tui(parsed)

        assert result == 0
        mock_require.assert_called_once_with("/fake/intel-server", "Intel llama-server")
        mock_validate.assert_called_once_with([server_cfg])
        mock_tui_app.assert_called_once_with([server_cfg], [1])

    def test_check_prereqs_with_nvidia(self) -> None:
        """_run_tui should validate every resolved server binary."""
        from llama_cli import server_runner

        intel_cfg = MagicMock()
        intel_cfg.server_bin = "/fake/intel-server"
        intel_cfg.device = "SYCL0"
        nvidia_cfg = MagicMock()
        nvidia_cfg.server_bin = "/fake/nvidia-server"
        nvidia_cfg.device = "CUDA0"
        parsed = Namespace(tui_mode="both", acknowledge_risky=False)

        with (
            patch.object(server_runner, "Config", return_value=MagicMock()),
            patch.object(
                server_runner,
                "_build_tui_mode_configs",
                return_value={
                    "both": (
                        [intel_cfg.server_bin, nvidia_cfg.server_bin],
                        [server_runner.INTEL_SERVER_NAME, server_runner.NVIDIA_SERVER_NAME],
                        [intel_cfg, nvidia_cfg],
                        [1, 0],
                    )
                },
            ),
            patch.object(server_runner, "require_executable", return_value=None) as mock_require,
            patch.object(server_runner, "_validate_tui_configs"),
            patch("llama_cli.tui.DashboardController") as mock_tui_app,
        ):
            mock_tui_app.return_value.run.return_value = None
            result = server_runner._run_tui(parsed)

        assert result == 0
        calls = mock_require.call_args_list
        assert len(calls) == 2
        assert calls[0][0] == ("/fake/intel-server", "Intel llama-server")
        assert calls[1][0] == ("/fake/nvidia-server", "NVIDIA llama-server")
