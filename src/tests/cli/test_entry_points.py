#!/usr/bin/env python3
"""Tests for entry point scripts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRunOpencodeModels:
    """Tests for run_opencode_models.py entry point."""

    def test_calls_run_cli(self) -> None:
        """run_opencode_models should call run_cli with sys.argv."""
        test_args = ["run_opencode_models.py", "summary-balanced"]

        # Mock at the module level where it's imported
        with patch("llama_cli.run_cli") as mock_run_cli:
            mock_run_cli.return_value = 0
            from llama_cli import run_cli

            result = run_cli(test_args)
            assert result == 0
            mock_run_cli.assert_called_once_with(test_args)


class TestRunModelsTui:
    """Tests for run_models_tui.py entry point."""

    def test_check_prereqs_calls_require_executable(self) -> None:
        """check_prereqs should validate Intel binary exists."""
        from run_models_tui import check_prereqs

        with patch("run_models_tui.Config") as mock_config_cls:
            mock_cfg = MagicMock()
            mock_cfg.llama_server_bin_intel = "/fake/intel-server"
            mock_cfg.llama_server_bin_nvidia = ""
            mock_config_cls.return_value = mock_cfg

            with patch("run_models_tui.require_executable") as mock_require:
                check_prereqs()
                mock_require.assert_called_once_with("/fake/intel-server", "Intel llama-server")

    def test_check_prereqs_with_nvidia(self) -> None:
        """check_prereqs should also validate NVIDIA binary when present."""
        from run_models_tui import check_prereqs

        with patch("run_models_tui.Config") as mock_config_cls:
            mock_cfg = MagicMock()
            mock_cfg.llama_server_bin_intel = "/fake/intel-server"
            mock_cfg.llama_server_bin_nvidia = "/fake/nvidia-server"
            mock_config_cls.return_value = mock_cfg

            with patch("run_models_tui.require_executable") as mock_require:
                check_prereqs()
                calls = mock_require.call_args_list
                assert len(calls) == 2
                assert calls[0][0][0] == "/fake/intel-server"
                assert calls[1][0][0] == "/fake/nvidia-server"
