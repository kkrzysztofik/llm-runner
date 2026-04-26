"""Tests for server runner module (server_runner.py).

Covers:
- _resolve_port: port resolution with defaults
- _normalize_main_args: argument normalization
- _build_target_configs: config building for different modes
- _run_dry_run_mode: dry-run handling
- verify_risks: risk verification
- _print_backend_error_and_exit: backend error output
- _print_validation_error: validation error output
- check_prereqs: prerequisite checks
- run_summary_balanced / run_summary_fast / run_qwen35: mode runners
- run_both: dual-server launch
- _acknowledge_risk_if_required: risk acknowledgment
- _run_mode: mode dispatch
- main: CLI entry point dispatch
- cli_main: console script entry
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.server_runner import (
    _build_target_configs,
    _normalize_main_args,
    _resolve_port,
    _run_dry_run_mode,
    verify_risks,
)
from llama_manager import Config, ErrorCode, ErrorDetail, ServerConfig, ServerManager

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
        import sys

        result = _normalize_main_args(None)
        assert result == sys.argv[1:]

    def test_normalize_empty_args(self) -> None:
        """_normalize_main_args with empty list should return empty list."""
        result = _normalize_main_args([])
        assert result == []

    def test_normalize_valid_mode(self) -> None:
        """_normalize_main_args should pass through valid modes unchanged."""
        for mode in [
            "summary-balanced",
            "summary-fast",
            "qwen35",
            "both",
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

    def test_normalize_skip_first_when_invalid_mode(self) -> None:
        """_normalize_main_args should skip first arg when it's not a valid mode."""
        result = _normalize_main_args(["not-a-mode", "summary-balanced"])
        assert result == ["summary-balanced"]

    def test_normalize_non_flag_non_mode_passes_through(self) -> None:
        """_normalize_main_args should pass through non-flag args that aren't modes."""
        # "8080" is not a valid mode and doesn't start with "-",
        # so it should be skipped
        result = _normalize_main_args(["8080"])
        assert result == []


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

    @patch("llama_cli.dry_run.dry_run")
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

    @patch("llama_cli.dry_run.dry_run")
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
# verify_risks
# =============================================================================


class TestVerifyRisks:
    """Tests for verify_risks function."""

    def test_verify_risks_no_risks(self) -> None:
        """verify_risks should do nothing when no risks detected."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"
        manager.is_risk_acknowledged.return_value = True

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = ["warning_bypass"]

        # Should not raise
        verify_risks(manager, [cfg], acknowledged=True)

    def test_verify_risks_with_risks_acknowledged(self) -> None:
        """verify_risks should acknowledge risks when flag is set."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = []

        with patch("llama_cli.server_runner.detect_risky_operations", return_value=["risky_op"]):
            verify_risks(manager, [cfg], acknowledged=True)

        # Should have added the risk label
        assert "warning_bypass" in cfg.risky_acknowledged

    def test_verify_risks_without_acknowledgment(self) -> None:
        """verify_risks should print warning when risks not acknowledged."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = []

        with (
            patch("llama_cli.server_runner.detect_risky_operations", return_value=["risky_op"]),
            patch("llama_cli.server_runner.input", return_value="y"),
        ):
            verify_risks(manager, [cfg], acknowledged=False)

        # Should have called acknowledge_risk
        manager.acknowledge_risk.assert_called_once()


# =============================================================================
# _print_backend_error_and_exit
# =============================================================================


class TestPrintBackendErrorAndExit:
    """Tests for _print_backend_error_and_exit helper."""

    def test_print_backend_error_exits_code_1(self) -> None:
        """_print_backend_error_and_exit should raise SystemExit(1)."""
        with patch("llama_cli.server_runner.print"):
            from llama_cli.server_runner import _print_backend_error_and_exit

            with pytest.raises(SystemExit) as exc_info:
                _print_backend_error_and_exit()
            assert exc_info.value.code == 1

    def test_print_backend_error_writes_stderr(self) -> None:
        """_print_backend_error_and_exit should write error details to stderr."""
        with patch("llama_cli.server_runner.print") as mock_print:
            from llama_cli.server_runner import _print_backend_error_and_exit

            with pytest.raises(SystemExit):
                _print_backend_error_and_exit()

            # Should print 4 lines to stderr (error + 3 indented details)
            assert mock_print.call_count == 4
            # First call: error line
            first_call = mock_print.call_args_list[0]
            assert first_call[1]["file"] == sys.stderr

    def test_print_backend_error_all_details(self) -> None:
        """_print_backend_error_and_exit should print all error detail lines."""
        with patch("llama_cli.server_runner.print") as mock_print:
            from llama_cli.server_runner import _print_backend_error_and_exit

            with pytest.raises(SystemExit):
                _print_backend_error_and_exit()

            all_calls = [str(call) for call in mock_print.call_args_list]
            combined = " ".join(all_calls)
            assert "failed_check: acknowledgement_required" in combined
            assert "why_blocked: risky operation detected and not acknowledged" in combined
            assert "how_to_fix: use --acknowledge-risky flag or confirm with 'y'" in combined


# =============================================================================
# _print_validation_error
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
            from llama_cli.server_runner import _print_validation_error

            _print_validation_error(error)
        assert exc_info.value.code == 1

    def test_print_validation_error_writes_stderr(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_print_validation_error should write all error fields to stderr."""
        error = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_exists",
            why_blocked="model file missing",
            how_to_fix="download model to specified path",
        )
        with pytest.raises(SystemExit):
            from llama_cli.server_runner import _print_validation_error

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
        self, capsys: "pytest.CaptureFixture[str]"
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
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_print_validation_error should include failed_check in output."""
        error = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_exists",
            why_blocked="model file missing",
            how_to_fix="download model",
        )
        with pytest.raises(SystemExit):
            from llama_cli.server_runner import _print_validation_error

            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "failed_check: model_exists" in captured.err

    def test_print_validation_error_why_blocked_line(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_print_validation_error should include why_blocked in output."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked="port out of range",
            how_to_fix="use valid port",
        )
        with pytest.raises(SystemExit):
            from llama_cli.server_runner import _print_validation_error

            _print_validation_error(error)

        captured = capsys.readouterr()
        assert "why_blocked: port out of range" in captured.err

    def test_print_validation_error_how_to_fix_line(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_print_validation_error should include how_to_fix in output."""
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
        assert "how_to_fix: use port between 1 and 65535" in captured.err


# =============================================================================
# check_prereqs
# =============================================================================


class TestCheckPrereqs:
    """Tests for check_prereqs function."""

    def test_check_prereqs_calls_require_executable_with_intel_bin(self) -> None:
        """check_prereqs should call require_executable with Config().llama_server_bin_intel."""
        with patch("llama_cli.server_runner.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_server_bin_intel = "/fake/path/llama-server"
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.server_runner.require_executable") as mock_require:
                from llama_cli.server_runner import check_prereqs

                check_prereqs()

                mock_require.assert_called_once_with(
                    "/fake/path/llama-server",
                    "Intel llama-server",
                )


# =============================================================================
# run_summary_balanced
# =============================================================================


class TestRunSummaryBalanced:
    """Tests for run_summary_balanced function."""

    def test_run_summary_balanced_success(self) -> None:
        """run_summary_balanced should launch server on valid config."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.validate_server_config", return_value=None),
        ):
            from llama_cli.server_runner import run_summary_balanced

            result = run_summary_balanced(8080, manager)

            assert result == 0
            manager.run_server_foreground.assert_called_once()
            call_args = manager.run_server_foreground.call_args
            assert call_args[0][0] == "summary-balanced"

    def test_run_summary_balanced_port_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_balanced should exit on invalid port."""
        manager = MagicMock(spec=ServerManager)

        with patch(
            "llama_cli.server_runner.validate_port",
            return_value=ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_range",
                why_blocked="port out of range",
                how_to_fix="use valid port",
            ),
        ):
            from llama_cli.server_runner import run_summary_balanced

            with pytest.raises(SystemExit) as exc_info:
                run_summary_balanced(0, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "PORT_INVALID" in captured.err

    def test_run_summary_balanced_model_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_balanced should exit when model not found."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch(
                "llama_cli.server_runner.require_model",
                return_value=ErrorDetail(
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    failed_check="model_exists",
                    why_blocked="model file missing",
                    how_to_fix="download model",
                ),
            ),
        ):
            from llama_cli.server_runner import run_summary_balanced

            with pytest.raises(SystemExit) as exc_info:
                run_summary_balanced(8080, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "FILE_NOT_FOUND" in captured.err

    def test_run_summary_balanced_backend_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_balanced should exit on backend validation error."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch(
                "llama_cli.server_runner.validate_server_config",
                return_value=ErrorDetail(
                    error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
                    failed_check="backend_eligibility",
                    why_blocked="backend not eligible",
                    how_to_fix="check backend config",
                ),
            ),
        ):
            from llama_cli.server_runner import run_summary_balanced

            with pytest.raises(SystemExit) as exc_info:
                run_summary_balanced(8080, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "BACKEND_NOT_ELIGIBLE" in captured.err


# =============================================================================
# run_summary_fast
# =============================================================================


class TestRunSummaryFast:
    """Tests for run_summary_fast function."""

    def test_run_summary_fast_success(self) -> None:
        """run_summary_fast should launch server on valid config."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.validate_server_config", return_value=None),
        ):
            from llama_cli.server_runner import run_summary_fast

            result = run_summary_fast(8082, manager)

            assert result == 0
            manager.run_server_foreground.assert_called_once()
            call_args = manager.run_server_foreground.call_args
            assert call_args[0][0] == "summary-fast"

    def test_run_summary_fast_port_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_fast should exit on invalid port."""
        manager = MagicMock(spec=ServerManager)

        with patch(
            "llama_cli.server_runner.validate_port",
            return_value=ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_range",
                why_blocked="port out of range",
                how_to_fix="use valid port",
            ),
        ):
            from llama_cli.server_runner import run_summary_fast

            with pytest.raises(SystemExit) as exc_info:
                run_summary_fast(0, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "PORT_INVALID" in captured.err

    def test_run_summary_fast_model_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_fast should exit when model not found."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch(
                "llama_cli.server_runner.require_model",
                return_value=ErrorDetail(
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    failed_check="model_exists",
                    why_blocked="model file missing",
                    how_to_fix="download model",
                ),
            ),
        ):
            from llama_cli.server_runner import run_summary_fast

            with pytest.raises(SystemExit) as exc_info:
                run_summary_fast(8082, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "FILE_NOT_FOUND" in captured.err

    def test_run_summary_fast_backend_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_summary_fast should exit on backend validation error."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch(
                "llama_cli.server_runner.validate_server_config",
                return_value=ErrorDetail(
                    error_code=ErrorCode.CONFIG_ERROR,
                    failed_check="config_validation",
                    why_blocked="invalid config",
                    how_to_fix="fix config",
                ),
            ),
        ):
            from llama_cli.server_runner import run_summary_fast

            with pytest.raises(SystemExit) as exc_info:
                run_summary_fast(8082, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "CONFIG_ERROR" in captured.err


# =============================================================================
# run_qwen35
# =============================================================================


class TestRunQwen35:
    """Tests for run_qwen35 function."""

    def test_run_qwen35_success(self) -> None:
        """run_qwen35 should launch NVIDIA CUDA server on valid config."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.require_executable", return_value=None),
            patch("llama_cli.server_runner.validate_server_config", return_value=None),
        ):
            from llama_cli.server_runner import run_qwen35

            result = run_qwen35(8081, manager)

            assert result == 0
            manager.run_server_foreground.assert_called_once()
            call_args = manager.run_server_foreground.call_args
            assert call_args[0][0] == "qwen35-coding"

    def test_run_qwen35_port_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_qwen35 should exit on invalid port."""
        manager = MagicMock(spec=ServerManager)

        with patch(
            "llama_cli.server_runner.validate_port",
            return_value=ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_range",
                why_blocked="port out of range",
                how_to_fix="use valid port",
            ),
        ):
            from llama_cli.server_runner import run_qwen35

            with pytest.raises(SystemExit) as exc_info:
                run_qwen35(0, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "PORT_INVALID" in captured.err

    def test_run_qwen35_model_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_qwen35 should exit when model not found."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch(
                "llama_cli.server_runner.require_model",
                return_value=ErrorDetail(
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    failed_check="model_exists",
                    why_blocked="model file missing",
                    how_to_fix="download model",
                ),
            ),
        ):
            from llama_cli.server_runner import run_qwen35

            with pytest.raises(SystemExit) as exc_info:
                run_qwen35(8081, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "FILE_NOT_FOUND" in captured.err

    def test_run_qwen35_executable_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_qwen35 should exit when NVIDIA executable not found."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch(
                "llama_cli.server_runner.require_executable",
                return_value=ErrorDetail(
                    error_code=ErrorCode.TOOLCHAIN_MISSING,
                    failed_check="executable_exists",
                    why_blocked="NVIDIA llama-server not found",
                    how_to_fix="build CUDA backend",
                ),
            ),
        ):
            from llama_cli.server_runner import run_qwen35

            with pytest.raises(SystemExit) as exc_info:
                run_qwen35(8081, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "TOOLCHAIN_MISSING" in captured.err

    def test_run_qwen35_backend_error(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """run_qwen35 should exit on backend validation error."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.require_executable", return_value=None),
            patch(
                "llama_cli.server_runner.validate_server_config",
                return_value=ErrorDetail(
                    error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
                    failed_check="backend_eligibility",
                    why_blocked="CUDA not available",
                    how_to_fix="check CUDA setup",
                ),
            ),
        ):
            from llama_cli.server_runner import run_qwen35

            with pytest.raises(SystemExit) as exc_info:
                run_qwen35(8081, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "BACKEND_NOT_ELIGIBLE" in captured.err


# =============================================================================
# run_both
# =============================================================================


class TestRunBoth:
    """Tests for run_both function."""

    def test_run_both_success(self) -> None:
        """run_both should launch both servers and wait."""
        manager = MagicMock(spec=ServerManager)
        manager.wait_for_any.return_value = 0

        with (
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.validate_ports", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.require_executable", return_value=None),
            patch("llama_cli.server_runner.validate_slots", return_value=None),
        ):
            from llama_cli.server_runner import run_both

            result = run_both(8080, 8081, manager)

            assert result == 0
            manager.start_servers.assert_called_once()
            manager.wait_for_any.assert_called_once()
            manager.cleanup_servers.assert_called_once()

    def test_run_both_validation_error_prints_all_errors(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """run_both should print all validation errors and exit code 1."""
        manager = MagicMock(spec=ServerManager)

        with (
            patch(
                "llama_cli.server_runner.validate_slots",
                return_value=MagicMock(
                    errors=[
                        ErrorDetail(
                            error_code=ErrorCode.PORT_INVALID,
                            failed_check="port_range",
                            why_blocked="port out of range",
                            how_to_fix="use valid port",
                        ),
                        ErrorDetail(
                            error_code=ErrorCode.FILE_NOT_FOUND,
                            failed_check="model_exists",
                            why_blocked="model missing",
                            how_to_fix="download model",
                        ),
                    ],
                ),
            ),
            patch("llama_cli.server_runner.validate_port", return_value=None),
            patch("llama_cli.server_runner.validate_ports", return_value=None),
            patch("llama_cli.server_runner.require_model", return_value=None),
            patch("llama_cli.server_runner.require_executable", return_value=None),
        ):
            from llama_cli.server_runner import run_both

            with pytest.raises(SystemExit) as exc_info:
                run_both(8080, 8081, manager)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "PORT_INVALID" in captured.err
            assert "FILE_NOT_FOUND" in captured.err

    def test_run_both_port_validation_fails(self) -> None:
        """run_both should exit when port validation fails."""
        manager = MagicMock(spec=ServerManager)

        with patch(
            "llama_cli.server_runner.validate_port",
            return_value=ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_range",
                why_blocked="port out of range",
                how_to_fix="use valid port",
            ),
        ):
            from llama_cli.server_runner import run_both

            with pytest.raises(SystemExit) as exc_info:
                run_both(0, 8081, manager)

            assert exc_info.value.code == 1


# =============================================================================
# _acknowledge_risk_if_required
# =============================================================================


class TestAcknowledgeRiskIfRequired:
    """Tests for _acknowledge_risk_if_required helper."""

    def test_acknowledge_risk_already_acknowledged(self) -> None:
        """_acknowledge_risk_if_required should return early if risk already acknowledged."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = True

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        from llama_cli.server_runner import _acknowledge_risk_if_required

        _acknowledge_risk_if_required(
            manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=False
        )

        # Should not call acknowledge_risk since already acknowledged
        manager.acknowledge_risk.assert_not_called()

    def test_acknowledge_risk_eof_exits_via_backend_error(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_acknowledge_risk_if_required should call backend error on EOF."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        with patch("llama_cli.server_runner.input", side_effect=EOFError):
            from llama_cli.server_runner import _acknowledge_risk_if_required

            with pytest.raises(SystemExit) as exc_info:
                _acknowledge_risk_if_required(
                    manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=False
                )

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "acknowledgement_required" in captured.err

    def test_acknowledge_risk_non_y_exits_via_backend_error(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_acknowledge_risk_if_required should exit when user response is not 'y'."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        with patch("llama_cli.server_runner.input", return_value="n"):
            from llama_cli.server_runner import _acknowledge_risk_if_required

            with pytest.raises(SystemExit) as exc_info:
                _acknowledge_risk_if_required(
                    manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=False
                )

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "acknowledgement_required" in captured.err

    def test_acknowledge_risk_y_calls_acknowledge_risk(self) -> None:
        """_acknowledge_risk_if_required should call manager.acknowledge_risk when user says 'y'."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        with patch("llama_cli.server_runner.input", return_value="y"):
            from llama_cli.server_runner import _acknowledge_risk_if_required

            _acknowledge_risk_if_required(
                manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=False
            )

            ack_token_value: str = "token-1"  # noqa: S105,S106
            manager.acknowledge_risk.assert_called_once_with(
                cfg.alias,
                "risky_op",
                launch_attempt_id="launch-1",
                ack_token=ack_token_value,
            )

    def test_acknowledge_risk_accepted_with_acknowledged_flag(self) -> None:
        """_acknowledge_risk_if_required should call acknowledge_risk when acknowledged=True."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        from llama_cli.server_runner import _acknowledge_risk_if_required

        _acknowledge_risk_if_required(
            manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=True
        )

        manager.acknowledge_risk.assert_called_once()

    def test_acknowledge_risk_prints_warning_when_not_acknowledged(
        self, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """_acknowledge_risk_if_required should print warning when risk detected and not acknowledged."""
        manager = MagicMock(spec=ServerManager)
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

        with patch("llama_cli.server_runner.input", return_value="y"):
            from llama_cli.server_runner import _acknowledge_risk_if_required

            _acknowledge_risk_if_required(
                manager, cfg, "risky_op", "launch-1", "token-1", acknowledged=False
            )

            captured = capsys.readouterr()
            assert "warning: risky operation detected in test: risky_op" in captured.out


# =============================================================================
# _run_mode
# =============================================================================


class TestRunMode:
    """Tests for _run_mode function."""

    def test_run_mode_summary_balanced(self) -> None:
        """_run_mode should dispatch to run_summary_balanced for summary-balanced."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0
        cfg = Config()

        with patch("llama_cli.server_runner.run_summary_balanced", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("summary-balanced", [8080], manager, cfg)

            assert result == 0
            mock_runner.assert_called_once_with(8080, manager)

    def test_run_mode_summary_fast(self) -> None:
        """_run_mode should dispatch to run_summary_fast for summary-fast."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0
        cfg = Config()

        with patch("llama_cli.server_runner.run_summary_fast", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("summary-fast", [8082], manager, cfg)

            assert result == 0
            mock_runner.assert_called_once_with(8082, manager)

    def test_run_mode_qwen35(self) -> None:
        """_run_mode should dispatch to run_qwen35 for qwen35."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0
        cfg = Config()

        with patch("llama_cli.server_runner.run_qwen35", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("qwen35", [8081], manager, cfg)

            assert result == 0
            mock_runner.assert_called_once_with(8081, manager)

    def test_run_mode_both(self) -> None:
        """_run_mode should dispatch to run_both for both."""
        manager = MagicMock(spec=ServerManager)
        manager.wait_for_any.return_value = 0
        cfg = Config()

        with patch("llama_cli.server_runner.run_both", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("both", [8080, 8081], manager, cfg)

            assert result == 0
            mock_runner.assert_called_once_with(8080, 8081, manager)

    def test_run_mode_unknown_returns_1(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """_run_mode should return 1 for unknown mode after printing usage."""
        manager = MagicMock(spec=ServerManager)
        cfg = Config()

        from llama_cli.server_runner import _run_mode

        result = _run_mode("unknown-mode", [], manager, cfg)

        assert result == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_run_mode_summary_balanced_uses_default_port(self) -> None:
        """_run_mode should use Config default port when ports list is empty."""
        manager = MagicMock(spec=ServerManager)
        manager.run_server_foreground.return_value = 0
        cfg = Config()

        with patch("llama_cli.server_runner.run_summary_balanced", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("summary-balanced", [], manager, cfg)

            assert result == 0
            # Should use cfg.summary_balanced_port (8080) as default
            mock_runner.assert_called_once_with(8080, manager)

    def test_run_mode_both_uses_default_ports(self) -> None:
        """_run_mode should use Config default ports when ports list is empty."""
        manager = MagicMock(spec=ServerManager)
        manager.wait_for_any.return_value = 0

        with patch("llama_cli.server_runner.run_both", return_value=0) as mock_runner:
            from llama_cli.server_runner import _run_mode

            result = _run_mode("both", [], manager, Config())

            assert result == 0
            mock_runner.assert_called_once_with(8080, 8081, manager)


# =============================================================================
# main — dispatch paths
# =============================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    def test_main_no_mode_prints_usage(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """main should print usage and return 1 when no mode provided."""
        with (
            patch("llama_cli.server_runner._normalize_main_args", return_value=[]),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            mock_parse.return_value = argparse.Namespace(mode=None)

            from llama_cli.server_runner import main

            result = main()

            assert result == 1
            captured = capsys.readouterr()
            assert "Usage:" in captured.out

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

            with patch("llama_cli.build_cli.main", return_value=0) as mock_build:
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

            with patch("llama_cli.build_cli.main", return_value=0) as mock_build:
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

            with patch("llama_cli.build_cli.main", return_value=0) as mock_build:
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

            with patch("llama_cli.doctor_cli.main", return_value=0) as mock_doctor:
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

            with patch("llama_cli.profile_cli.main", return_value=0) as mock_profile:
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

            with patch("llama_cli.smoke_cli.run_smoke", return_value=0) as mock_smoke:
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

            with patch("llama_cli.smoke_cli.run_smoke", return_value=0) as mock_smoke:
                from llama_cli.server_runner import main

                result = main()

                assert result == 0
                smoke_args = mock_smoke.call_args[0][0]
                assert smoke_args == ["both"]

    def test_main_value_error_returns_1(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """main should catch ValueError from _run_mode and return 1."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["invalid", "summary-balanced"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="invalid",
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

            with (
                patch("llama_cli.server_runner.ServerManager"),
                patch("llama_cli.server_runner.check_prereqs"),
                patch("llama_cli.server_runner._build_target_configs", return_value=[]),
                patch("llama_cli.server_runner.verify_risks"),
                patch(
                    "llama_cli.server_runner._run_mode",
                    side_effect=ValueError("bad mode"),
                ),
            ):
                from llama_cli.server_runner import main

                result = main()

                assert result == 1
                captured = capsys.readouterr()
                assert "invalid arguments" in captured.err

    def test_main_index_error_returns_1(self, capsys: "pytest.CaptureFixture[str]") -> None:
        """main should catch IndexError from _run_mode and return 1."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["invalid", "summary-balanced"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="invalid",
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

            with (
                patch("llama_cli.server_runner.ServerManager"),
                patch("llama_cli.server_runner.check_prereqs"),
                patch("llama_cli.server_runner._build_target_configs", return_value=[]),
                patch("llama_cli.server_runner.verify_risks"),
                patch(
                    "llama_cli.server_runner._run_mode",
                    side_effect=IndexError("index out of range"),
                ),
            ):
                from llama_cli.server_runner import main

                result = main()

                assert result == 1
                captured = capsys.readouterr()
                assert "index error" in captured.err

    def test_main_signal_handlers_registered(self) -> None:
        """main should register signal handlers for SIGINT and SIGTERM."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["summary-balanced", "8080"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="summary-balanced",
                ports=[8080],
                acknowledge_risky=True,
                backend="sycl",
                dry_run=False,
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

            with patch("llama_cli.server_runner.ServerManager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr_cls.return_value = mock_mgr

                with (
                    patch("llama_cli.server_runner.check_prereqs"),
                    patch("llama_cli.server_runner._build_target_configs", return_value=[]),
                    patch("llama_cli.server_runner.verify_risks"),
                    patch("llama_cli.server_runner._run_mode", return_value=0),
                    patch("llama_cli.server_runner.Colors"),
                    patch("llama_cli.server_runner.atexit"),
                    patch("llama_cli.server_runner.os"),
                    patch("llama_cli.server_runner.signal") as mock_signal,
                ):
                    from llama_cli.server_runner import main

                    main()

                    mock_mgr_cls.assert_called_once()
                    mock_signal.signal.assert_any_call(mock_signal.SIGINT, mock_mgr.on_interrupt)
                    mock_signal.signal.assert_any_call(mock_signal.SIGTERM, mock_mgr.on_terminate)

    def test_main_atexit_cleanup_registered(self) -> None:
        """main should register manager.cleanup_servers with atexit."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["summary-balanced", "8080"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="summary-balanced",
                ports=[8080],
                acknowledge_risky=True,
                backend="sycl",
                dry_run=False,
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

            with patch("llama_cli.server_runner.ServerManager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr_cls.return_value = mock_mgr

                with (
                    patch("llama_cli.server_runner.check_prereqs"),
                    patch("llama_cli.server_runner._build_target_configs", return_value=[]),
                    patch("llama_cli.server_runner.verify_risks"),
                    patch("llama_cli.server_runner._run_mode", return_value=0),
                    patch("llama_cli.server_runner.Colors"),
                    patch("llama_cli.server_runner.atexit") as mock_atexit,
                    patch("llama_cli.server_runner.os"),
                    patch("llama_cli.server_runner.signal"),
                ):
                    from llama_cli.server_runner import main

                    main()

                    mock_atexit.register.assert_called_once_with(mock_mgr.cleanup_servers)

    def test_main_zes_env_var_set(self) -> None:
        """main should set ZES_ENABLE_SYSMAN=1 environment variable."""
        with (
            patch(
                "llama_cli.server_runner._normalize_main_args",
                return_value=["summary-balanced", "8080"],
            ),
            patch("llama_cli.server_runner.parse_args") as mock_parse,
        ):
            parsed = argparse.Namespace(
                mode="summary-balanced",
                ports=[8080],
                acknowledge_risky=True,
                backend="sycl",
                dry_run=False,
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

            with (
                patch("llama_cli.server_runner.ServerManager"),
                patch("llama_cli.server_runner.check_prereqs"),
                patch("llama_cli.server_runner._build_target_configs", return_value=[]),
                patch("llama_cli.server_runner.verify_risks"),
                patch("llama_cli.server_runner._run_mode", return_value=0),
                patch("llama_cli.server_runner.Colors"),
                patch("llama_cli.server_runner.atexit"),
                patch("llama_cli.server_runner.signal"),
                patch("llama_cli.server_runner.os") as mock_os,
            ):
                mock_os.environ = {}
                from llama_cli.server_runner import main

                main()

                assert mock_os.environ["ZES_ENABLE_SYSMAN"] == "1"


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
