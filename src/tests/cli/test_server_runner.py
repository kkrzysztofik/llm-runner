import argparse
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli import server_runner
from llama_cli.cli_parser import parse_args, parse_tui_args
from llama_cli.commands.dry_run import dry_run
from llama_cli.server_runner import (
    _build_target_configs,
    _normalize_main_args,
    _print_validation_error,
    _resolve_port,
    _run_dry_run_mode,
)
from llama_cli.tui import DashboardController
from llama_manager.config import Config, ErrorCode, ErrorDetail, MultiValidationError, ServerConfig
from llama_manager.orchestration import LaunchResult, ServerManager

# =============================================================================
# _resolve_port
# =============================================================================


class TestResolvePort:
    """Tests for _resolve_port helper."""

    def test_resolve_port_from_list(self) -> None:
        """_resolve_port should return port from list when available."""
        port = _resolve_port([8080, 8081], 0, 8080)
        assert port == 8080

    def test_resolve_port_from_list_secondary(self) -> None:
        """_resolve_port should return secondary port from list."""
        port = _resolve_port([8080, 8081], 1, 8081)
        assert port == 8081

    def test_resolve_port_default_when_empty(self) -> None:
        """_resolve_port should return default when list is empty."""
        port = _resolve_port([], 0, 8080)
        assert port == 8080

    def test_resolve_port_default_when_index_out_of_range(self) -> None:
        """_resolve_port should return default when index > list length."""
        port = _resolve_port([8080], 1, 8081)
        assert port == 8081

    def test_resolve_port_default_when_list_shorter(self) -> None:
        """_resolve_port should return default for missing ports."""
        port = _resolve_port([8080], 2, 9999)
        assert port == 9999


# =============================================================================
# _normalize_main_args
# =============================================================================


class TestNormalizeMainArgs:
    """Tests for _normalize_main_args helper."""

    def test_normalize_none_args(self) -> None:
        """_normalize_main_args with None should return sys.argv[1:]."""
        result = _normalize_main_args(None)
        assert result == sys.argv[1:]

    def test_normalize_empty_args(self) -> None:
        """_normalize_main_args with empty list should return empty list."""
        result = _normalize_main_args([])
        assert result == []

    def test_normalize_valid_mode(self) -> None:
        """_normalize_main_args should pass through valid modes unchanged."""
        for mode in [
            "dry-run",
            "doctor",
            "build",
            "setup",
        ]:
            result = _normalize_main_args([mode])
            assert result == [mode]

    def test_normalize_flag_args(self) -> None:
        """_normalize_main_args should pass through flag args unchanged."""
        result = _normalize_main_args(["--port", "8080"])
        assert result == ["--port", "8080"]

    def test_normalize_skip_first_when_program_name(self) -> None:
        """_normalize_main_args should drop a leading console-script program name."""
        result = _normalize_main_args(["llm-runner", "summary-balanced"])
        assert result == ["summary-balanced"]

    def test_normalize_preserves_bare_run_group(self) -> None:
        """Bare run-group names must not be stripped (they are not program names)."""
        result = _normalize_main_args(["summary-balanced"])
        assert result == ["summary-balanced"]

    def test_normalize_non_flag_non_mode_passes_through(self) -> None:
        """Unknown bare tokens should be preserved for parse_args to reject."""
        result = _normalize_main_args(["8080"])
        assert result == ["8080"]


# =============================================================================
# _build_target_configs
# =============================================================================


class TestBuildTargetConfigs:
    """Tests for _build_target_configs helper."""

    def test_build_configs_summary_balanced(self) -> None:
        """_build_target_configs should create summary-balanced config."""
        cfg = Config()
        configs = _build_target_configs("summary-balanced", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "summary-balanced"

    def test_build_configs_qwen35(self) -> None:
        """_build_target_configs should create qwen35 config."""
        cfg = Config()
        configs = _build_target_configs("qwen35", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "qwen35-coding"

    def test_build_configs_both(self) -> None:
        """_build_target_configs should create two configs for 'both'."""
        cfg = Config()
        configs = _build_target_configs("both", [], cfg)
        assert len(configs) == 2
        assert configs[0].alias == "summary-balanced"
        assert configs[1].alias == "qwen35-coding"

    def test_build_configs_with_ports(self) -> None:
        """_build_target_configs should use provided ports."""
        cfg = Config()
        configs = _build_target_configs("qwen35", [9999], cfg)
        assert len(configs) == 1
        assert configs[0].port == 9999

    def test_build_configs_unknown_mode(self) -> None:
        """_build_target_configs should return empty list for unknown mode."""
        cfg = Config()
        configs = _build_target_configs("unknown-mode", [], cfg)
        assert configs == []

    def test_build_configs_summary_fast(self) -> None:
        """_build_target_configs should create summary-fast config."""
        cfg = Config()
        configs = _build_target_configs("summary-fast", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "summary-fast"

    def test_build_configs_rejects_custom_slot_profile_as_mode(self) -> None:
        """_build_target_configs should only accept fixed launch modes."""
        cfg = Config()
        configs = _build_target_configs("custom-group", [9090], cfg)
        assert configs == []


# =============================================================================
# _run_dry_run_mode
# =============================================================================


class TestRunDryRunMode:
    """Tests for _run_dry_run_mode helper."""

    def test_dry_run_missing_mode(self) -> None:
        """_run_dry_run_mode should return 1 when dry_run_mode is missing."""
        parsed = argparse.Namespace(mode="dry-run", ports=[], acknowledge_risky=False)
        exit_code = _run_dry_run_mode(parsed, acknowledged=False)
        assert exit_code == 1

    @patch("llama_cli.commands.dry_run.dry_run")
    def test_dry_run_with_valid_mode(self, mock_dry_run: MagicMock) -> None:
        """_run_dry_run_mode should call dry_run with correct args."""
        parsed = argparse.Namespace(
            mode="dry-run",
            dry_run_mode="both",
            ports=[8080, 8081],
            acknowledge_risky=True,
        )
        exit_code = _run_dry_run_mode(parsed, acknowledged=True)
        assert exit_code == 0
        mock_dry_run.assert_called_once_with("both", "8080", "8081", acknowledged=True)

    @patch("llama_cli.commands.dry_run.dry_run")
    def test_dry_run_with_single_port(self, mock_dry_run: MagicMock) -> None:
        """_run_dry_run_mode should handle single port."""
        parsed = argparse.Namespace(
            mode="dry-run",
            dry_run_mode="summary-balanced",
            ports=[8080],
            acknowledge_risky=False,
        )
        exit_code = _run_dry_run_mode(parsed, acknowledged=False)
        assert exit_code == 0
        mock_dry_run.assert_called_once_with("summary-balanced", "8080", "", acknowledged=False)


# =============================================================================
# main — dispatch paths
# =============================================================================


class TestPrintValidationError:
    """Tests for _print_validation_error helper."""

    def test_print_validation_error_exits_code_1(self) -> None:
        """_print_validation_error should raise SystemExit(1)."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked="port out of range",
            how_to_fix="use port between 1 and 65535",
        )
        with pytest.raises(SystemExit) as exc_info:
            _print_validation_error(error)
        assert exc_info.value.code == 1

    def test_print_validation_error_writes_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_print_validation_error should write all error fields to stderr."""
        error = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_exists",
            why_blocked="model file missing",
            how_to_fix="download model to specified path",
        )
        with pytest.raises(SystemExit):
            _print_validation_error(error)

        captured = capsys.readouterr()
        # Should print 4 lines: error_code, failed_check, why_blocked, how_to_fix
        lines = captured.err.strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            assert (
                "error:" in line
                or "failed_check:" in line
                or "why_blocked:" in line
                or "how_to_fix:" in line
            )

    def test_print_validation_error_error_code_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_print_validation_error should include error_code in first line."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked="port out of range",
            how_to_fix="use port between 1 and 65535",
        )
        with pytest.raises(SystemExit):
            from llama_cli.server_runner import _print_validation_error

            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "error: PORT_INVALID" in captured.err

    def test_print_validation_error_failed_check_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_print_validation_error should include failed_check in output."""
        error = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_exists",
            why_blocked="model file missing",
            how_to_fix="download model",
        )
        with pytest.raises(SystemExit):
            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "failed_check: model_exists" in captured.err

    def test_print_validation_error_why_blocked_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_print_validation_error should include why_blocked in output."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked="port out of range",
            how_to_fix="use valid port",
        )
        with pytest.raises(SystemExit):
            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "why_blocked: port out of range" in captured.err

    def test_print_validation_error_how_to_fix_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_print_validation_error should include how_to_fix in output."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked="port out of range",
            how_to_fix="use port between 1 and 65535",
        )
        with pytest.raises(SystemExit):
            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "how_to_fix: use port between 1 and 65535" in captured.err


# =============================================================================
# main — dispatch paths
# =============================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    def test_main_no_args_launches_tui(self) -> None:
        """main with no args should launch the TUI in standalone mode."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=[]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
            patch("llama_cli.server_runner._run_tui", return_value=0) as mock_run_tui,
        ):
            mock_parse.return_value = argparse.Namespace(
                mode="tui",
                tui_mode=None,
                port=None,
                port2=None,
                acknowledge_risky=False,
            )

            from llama_cli.server_runner import main

            result = main()

            assert result == 0
            mock_run_tui.assert_called_once()

    def test_main_build_dispatch(self, tmp_path: Path) -> None:
        """main should dispatch build args to the build CLI."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=["build", "sycl"]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="build",
                backend="sycl",
                build_args=["sycl"],
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.build.main", return_value=0) as mock_build:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_build.assert_called_once_with(["sycl"])

    def test_main_build_dry_run_dispatch(self, tmp_path: Path) -> None:
        """main should preserve build CLI flags."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["build", "sycl", "--dry-run"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="build",
                backend="sycl",
                build_args=["sycl", "--dry-run"],
                dry_run=True,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.build.main", return_value=0) as mock_build:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_build.assert_called_once_with(["sycl", "--dry-run"])

    def test_main_build_both_dispatch(self) -> None:
        """main should dispatch 'build both' through the build CLI."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=["build", "both"]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="build",
                backend="both",
                build_args=["both"],
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.build.main", return_value=0) as mock_build:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_build.assert_called_once_with(["both"])

    def test_main_setup_dispatch(self) -> None:
        """main should dispatch to setup_main for 'setup' mode."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=["setup", "check"]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="setup",
                backend="sycl",
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.server_runner.setup_main", return_value=0) as mock_setup:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_setup.assert_called_once()

    def test_main_doctor_dispatch(self) -> None:
        """main should dispatch to doctor_main for 'doctor' mode."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=["doctor"]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="doctor",
                backend="sycl",
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.doctor.main", return_value=0) as mock_doctor:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_doctor.assert_called_once()

    def test_main_profile_dispatch(self) -> None:
        """main should dispatch to profile_main for 'profile' mode."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["profile", "slot0", "balanced", "--json"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="profile",
                backend="sycl",
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
                smoke_mode=None,
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.profile.main", return_value=0) as mock_profile:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_profile.assert_called_once()

    def test_main_smoke_dispatch(self) -> None:
        """main should dispatch to run_smoke with reconstructed args for 'smoke' mode."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=[
                    "smoke",
                    "both",
                    "slot0",
                    "--api-key",
                    "key123",
                    "--model-id",
                    "m1",
                    "--max-tokens",
                    "16",
                    "--prompt",
                    "hello",
                    "--delay",
                    "2",
                    "--timeout",
                    "30",
                    "--json",
                ],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="smoke",
                smoke_mode="both",
                slot_id="slot0",
                api_key="key123",
                model_id="m1",
                max_tokens=16,
                prompt="hello",
                delay=2,
                timeout=30,
                json=True,
                backend="sycl",
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.smoke.run_smoke", return_value=0) as mock_smoke:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                mock_smoke.assert_called_once()
                # Verify smoke_args are reconstructed correctly
                smoke_args = mock_smoke.call_args[0][0]
                assert smoke_args == [
                    "both",
                    "slot0",
                    "--api-key",
                    "key123",
                    "--model-id",
                    "m1",
                    "--max-tokens",
                    "16",
                    "--prompt",
                    "hello",
                    "--delay",
                    "2",
                    "--timeout",
                    "30",
                    "--json",
                ]

    def test_main_smoke_dispatch_minimal_args(self) -> None:
        """main should dispatch to run_smoke with minimal args when optional flags absent."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=["smoke", "both"]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="smoke",
                smoke_mode="both",
                slot_id=None,
                api_key=None,
                model_id=None,
                max_tokens=None,
                prompt=None,
                delay=None,
                timeout=None,
                json=False,
                backend="sycl",
                dry_run=False,
                ports=[],
                acknowledge_risky=False,
            )
            mock_parse.return_value = parsed

            with patch("llama_cli.commands.smoke.run_smoke", return_value=0) as mock_smoke:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                smoke_args = mock_smoke.call_args[0][0]
                assert smoke_args == ["both"]

    def test_main_direct_mode_exits_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject bare run-group names (require the tui subcommand)."""
        from llama_cli.server_runner import main

        with pytest.raises(SystemExit) as exc_info:
            main(["summary-balanced"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "summary-balanced" in captured.err
        assert "tui" in captured.err.lower()


# =============================================================================
# cli_main
# =============================================================================


class TestCliMain:
    """Tests for cli_main function."""

    def test_cli_main_raises_system_exit(self) -> None:
        """cli_main should raise SystemExit with the return code from main()."""
        with patch("llama_cli.server_runner.main", return_value=0):
            from llama_cli.server_runner import cli_main

            with pytest.raises(SystemExit) as exc_info:
                cli_main()

            assert exc_info.value.code == 0

    def test_cli_main_passes_return_code(self) -> None:
        """cli_main should pass through the return code from main()."""
        with patch("llama_cli.server_runner.main", return_value=1):
            from llama_cli.server_runner import cli_main

            with pytest.raises(SystemExit) as exc_info:
                cli_main()

            assert exc_info.value.code == 1

    def test_cli_main_passes_argv(self) -> None:
        """cli_main should pass sys.argv[1:] to main()."""
        with patch("llama_cli.server_runner.main", return_value=0) as mock_main:
            from llama_cli.server_runner import cli_main

            with pytest.raises(SystemExit):
                cli_main()

            mock_main.assert_called_once()
            # Verify it was called with sys.argv[1:]
            call_args = mock_main.call_args
            assert call_args[0][0] == sys.argv[1:]


# =============================================================================
# Orchestration regression tests — dynamic registry integration
# =============================================================================


class TestBuildTuiModeConfigs:
    """Tests for _build_tui_mode_configs fixed mode behavior."""

    def test_build_tui_mode_configs_returns_all_modes(self) -> None:
        """_build_tui_mode_configs should return configs for all fixed modes."""
        from llama_cli.server_runner import _build_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="both", port=None, port2=None)

        result = _build_tui_mode_configs(cfg, parsed)

        # Should have entries for summary-balanced, summary-fast, qwen35, both
        assert "summary-balanced" in result
        assert "both" in result

    def test_build_tui_mode_configs_with_port_overrides(self) -> None:
        """_build_tui_mode_configs should apply port overrides from parsed args."""
        from llama_cli.server_runner import _build_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="both", port=9090, port2=9091)

        result = _build_tui_mode_configs(cfg, parsed)

        # Find the 'both' entry
        assert "both" in result
        server_bins, server_names, configs, gpu_indices = result["both"]
        assert len(configs) == 2
        assert configs[0].port == 9090
        assert configs[1].port == 9091

    def test_build_tui_mode_configs_uses_custom_builtin_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TUI launch configs should use persisted custom overrides for built-ins."""
        from llama_cli.server_runner import _build_tui_mode_configs
        from llama_manager.config import Config, SlotProfileSpec
        from llama_manager.slot_profile_store import save_custom_slot_profile

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        save_custom_slot_profile(
            SlotProfileSpec(
                profile_id="summary-balanced",
                model="/models/edited-summary.gguf",
                alias="summary-balanced",
                device="SYCL0",
                port=17777,
                ctx_size=12345,
                ubatch_size=256,
                threads=6,
            )
        )

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-balanced", port=None, port2=None)

        result = _build_tui_mode_configs(cfg, parsed)
        configs = result["summary-balanced"][2]

        assert len(configs) == 1
        assert configs[0].model == "/models/edited-summary.gguf"
        assert configs[0].port == 17777
        assert configs[0].ctx_size == 12345
        assert configs[0].threads == 6


class TestResolveTuiModeConfigs:
    """Tests for _resolve_tui_mode_configs port override behavior."""

    def test_resolve_mode_configs_no_overrides(self) -> None:
        """_resolve_tui_mode_configs should return default configs when no overrides."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-balanced", port=None, port2=None)

        configs = _resolve_tui_mode_configs("summary-balanced", cfg, parsed)

        assert len(configs) == 1
        assert configs[0].alias == "summary-balanced"
        assert configs[0].port == cfg.deployment.summary_balanced_port

    def test_resolve_mode_configs_single_port_override(self) -> None:
        """_resolve_tui_mode_configs should apply single port override."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-balanced", port=9999, port2=None)

        configs = _resolve_tui_mode_configs("summary-balanced", cfg, parsed)

        assert len(configs) == 1
        assert configs[0].port == 9999

    def test_resolve_mode_configs_both_ports_for_multi_profile_mode(self) -> None:
        """_resolve_tui_mode_configs should apply both ports for multi-profile modes."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="both", port=8080, port2=8081)

        configs = _resolve_tui_mode_configs("both", cfg, parsed)

        assert len(configs) == 2
        assert configs[0].port == 8080
        assert configs[1].port == 8081

    def test_resolve_mode_configs_port2_only_preserves_first_default(self) -> None:
        """_resolve_tui_mode_configs should preserve first port default when only port2 given."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config

        cfg = Config()
        parsed = argparse.Namespace(mode="both", port=None, port2=8081)

        configs = _resolve_tui_mode_configs("both", cfg, parsed)

        assert len(configs) == 2
        assert configs[0].port == cfg.deployment.summary_balanced_port  # First port stays default
        assert configs[1].port == 8081

    def test_resolve_mode_configs_uses_tui_registry_when_registry_not_supplied(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback registry should include custom overrides for TUI resolution."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config, SlotProfileSpec
        from llama_manager.slot_profile_store import save_custom_slot_profile

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        save_custom_slot_profile(
            SlotProfileSpec(
                profile_id="summary-fast",
                model="/models/edited-fast.gguf",
                alias="summary-fast",
                device="SYCL0",
                port=18888,
                ctx_size=8192,
                ubatch_size=128,
                threads=3,
            )
        )

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-fast", port=None, port2=None)

        configs = _resolve_tui_mode_configs("summary-fast", cfg, parsed)

        assert len(configs) == 1
        assert configs[0].model == "/models/edited-fast.gguf"
        assert configs[0].port == 18888
        assert configs[0].threads == 3


class _FakeTextualDashboardApp:
    """Fake Textual app for TUI tests without terminal rendering."""

    def __init__(self, controller: DashboardController) -> None:
        self.controller = controller

    def run(self) -> None:
        return None


def _risky_cfg() -> ServerConfig:
    return ServerConfig(
        model="/home/kmk/models/test-model.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=80,
        ctx_size=2048,
        ubatch_size=512,
        threads=4,
    )


def test_parse_args_tui_supports_acknowledge_risky_flag() -> None:
    parsed = parse_args(["tui", "summary-balanced", "--acknowledge-risky"])
    assert parsed.mode == "tui"
    assert parsed.tui_mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


def test_parse_tui_args_supports_acknowledge_risky_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-runner", "summary-balanced", "--acknowledge-risky"],
    )
    parsed = parse_tui_args()
    assert parsed.mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


def test_ack_token_validation_is_attempt_scoped() -> None:
    manager = ServerManager()
    attempt_id = manager.begin_launch_attempt("attempt-1")
    valid_token = manager.issue_ack_token(attempt_id)

    assert manager.validate_ack_token(attempt_id, valid_token) is True
    assert manager.validate_ack_token(attempt_id, "ack:other") is False


def test_cleanup_clears_attempt_ack_cache() -> None:
    manager = ServerManager()
    attempt_id = manager.begin_launch_attempt("attempt-1")
    manager.acknowledge_risk("summary-balanced", "privileged_port", attempt_id)
    assert manager.is_risk_acknowledged("summary-balanced", "privileged_port", attempt_id)

    manager.cleanup_servers()

    assert manager.is_risk_acknowledged("summary-balanced", "privileged_port", attempt_id) is False


def test_tui_risk_prompt_tracks_required_and_acknowledged_states() -> None:
    app = DashboardController([_risky_cfg()], [0])

    app._build_risk_panel_required()
    assert app.active_risk_kind == "hardware"
    assert app.risks_acknowledged is False

    app._build_risk_panel_acknowledged()
    assert app.active_risk_kind == "hardware"
    assert app.risks_acknowledged is True


def test_tui_run_keeps_acknowledged_risk_prompt_active() -> None:
    app = DashboardController([_risky_cfg()], [0])
    app.running = False

    with (
        patch("llama_cli.tui.controller.DashboardApp", _FakeTextualDashboardApp),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(status="success", launched=["summary-balanced"]),
        ),
        patch.object(
            app.server_manager,
            "start_servers",
            side_effect=lambda configs, log_handlers=None: [MagicMock() for _ in configs],
        ),
        patch.object(app.server_manager, "cleanup_servers"),
    ):
        app.run(acknowledged=True)

    assert app.active_risk_kind == "hardware"
    assert app.risks_acknowledged is True


def test_dry_run_prompts_for_risky_operation_with_exact_prompt() -> None:
    from llama_manager.dry_run import DryRunResult

    mock_result = DryRunResult(
        mode="summary-balanced",
        slot_payloads=[],
        has_error=False,
        errors=[],
        warnings=["privileged_port"],
        artifact_payload=None,
    )

    with (
        patch("llama_cli.commands.dry_run.run_dry_run", return_value=mock_result),
        patch("builtins.input", return_value="n") as mock_input,
        pytest.raises(SystemExit) as exc,
    ):
        dry_run("summary-balanced", primary_port="8080")

    assert exc.value.code == 1
    mock_input.assert_called_once_with("Confirm risky operation [y/N]: ")


def test_dry_run_invalid_mode_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        dry_run("invalid-mode")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "invalid mode" in captured.err
    assert "Valid modes:" in captured.err


def test_dry_run_exits_when_backend_validation_fails() -> None:
    from llama_manager.dry_run import DryRunResult

    mock_result = DryRunResult(
        mode="summary-fast",
        slot_payloads=[],
        has_error=True,
        errors=["backend blocked"],
        warnings=[],
        artifact_payload=None,
    )

    with (
        patch("llama_cli.commands.dry_run.run_dry_run", return_value=mock_result),
        pytest.raises(SystemExit) as exc,
    ):
        dry_run("summary-fast")

    assert exc.value.code == 1


def test_tui_run_exits_when_launch_is_blocked() -> None:
    safe_cfg = ServerConfig(
        model="/home/kmk/models/test-model.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=2048,
        ubatch_size=512,
        threads=4,
    )
    app = DashboardController([safe_cfg], [0])
    app.running = False

    blocked_error = ErrorDetail(
        error_code=ErrorCode.PORT_CONFLICT,
        failed_check="lockfile_creation",
        why_blocked="blocked",
        how_to_fix="fix",
    )

    with (
        patch("llama_cli.tui.controller.DashboardApp", _FakeTextualDashboardApp),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(
                status="blocked",
                launched=[],
                errors=MultiValidationError(errors=[blocked_error]),
            ),
        ),
        patch.object(app.server_manager, "start_servers"),
        patch("builtins.input", side_effect=RuntimeError("unexpected prompt")),
        pytest.raises(SystemExit) as exc,
    ):
        app.run(acknowledged=False)

    assert exc.value.code == 1
    messages = [message for _ts, message in app._status_messages]
    assert "launch blocked - no slots could be launched" in messages


def test_tui_run_buffers_degraded_warnings() -> None:
    app = DashboardController([_risky_cfg()], [0])
    app.running = False

    with (
        patch("llama_cli.tui.controller.DashboardApp", _FakeTextualDashboardApp),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(
                status="degraded",
                launched=["summary-balanced"],
                warnings=["slot blocked"],
            ),
        ),
        patch.object(
            app.server_manager,
            "start_servers",
            side_effect=lambda configs, log_handlers=None: [MagicMock() for _ in configs],
        ),
        patch.object(app.server_manager, "cleanup_servers"),
    ):
        app.run(acknowledged=True)

    messages = [message for _ts, message in app._status_messages]
    assert "launch degraded - some slots blocked" in messages
    assert "slot blocked" in messages


def test_server_runner_main_dispatches_dry_run_mode() -> None:
    parsed = Namespace(
        mode="dry-run",
        dry_run_mode="both",
        ports=[8080, 8081],
        acknowledge_risky=True,
    )
    with (
        patch("llama_cli.server_runner.parse_args", return_value=parsed),
        patch("llama_cli.server_runner._run_dry_run_mode", return_value=0) as mock_run,
    ):
        code = server_runner.main(["dry-run", "both", "8080", "8081"])

    assert code == 0
    mock_run.assert_called_once_with(parsed, True)


def test_server_runner_main_dry_run_without_target_mode_returns_one() -> None:
    parsed = Namespace(mode="dry-run", dry_run_mode=None, ports=[], acknowledge_risky=False)
    code = server_runner._run_dry_run_mode(parsed, acknowledged=False)
    assert code == 1
