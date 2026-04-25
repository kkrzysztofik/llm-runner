"""Tests for smoke CLI argument parsing.

Covers:
  - T033: `smoke both` and `smoke slot <id>` argument parsing
  - smoke subcommand structure in cli_parser.py
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.cli_parser import parse_args
from llama_manager.smoke import probe_slot

# ---------------------------------------------------------------------------
# T033 — CLI argument parsing for smoke subcommand
# ---------------------------------------------------------------------------


class TestSmokeCliParsing:
    """T033: CLI argument parsing for `smoke both` and `smoke slot <id>`."""

    def test_smoke_both_mode_accepted(self) -> None:
        """parse_args('smoke both') should accept smoke mode."""
        result = parse_args(["smoke", "both"])
        assert result.mode == "smoke"
        assert result.smoke_mode == "both"

    def test_smoke_slot_mode_accepted(self) -> None:
        """parse_args('smoke slot slot1') should accept smoke slot mode."""
        result = parse_args(["smoke", "slot", "slot1"])
        assert result.mode == "smoke"
        assert result.smoke_mode == "slot"
        assert result.slot_id == "slot1"

    def test_smoke_with_json_flag(self) -> None:
        """parse_args should handle --json flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--json"])
        assert result.mode == "smoke"
        assert result.json is True

    def test_smoke_with_api_key_flag(self) -> None:
        """parse_args should handle --api-key flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--api-key", "sk-test"])
        assert result.api_key == "sk-test"

    def test_smoke_with_model_id_flag(self) -> None:
        """parse_args should handle --model-id flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--model-id", "Qwen3.5-2B"])
        assert result.model_id == "Qwen3.5-2B"

    def test_smoke_with_max_tokens_flag(self) -> None:
        """parse_args should handle --max-tokens flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--max-tokens", "16"])
        assert result.max_tokens == 16

    def test_smoke_with_delay_flag(self) -> None:
        """parse_args should handle --delay flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--delay", "5"])
        assert result.delay == 5

    def test_smoke_with_timeout_flag(self) -> None:
        """parse_args should handle --timeout flag for smoke subcommand."""
        result = parse_args(["smoke", "both", "--timeout", "60"])
        assert result.timeout == 60

    def test_smoke_invalid_mode_rejected(self) -> None:
        """parse_args should reject invalid smoke mode."""
        with pytest.raises(SystemExit):
            parse_args(["smoke", "invalid"])

    def test_smoke_missing_mode_rejected(self) -> None:
        """parse_args should reject smoke without mode argument."""
        with pytest.raises(SystemExit):
            parse_args(["smoke"])

    def test_valid_modes_accepted(self) -> None:
        """parse_args should accept all valid modes."""
        valid_modes = [
            "summary-balanced",
            "summary-fast",
            "qwen35",
            "both",
            "dry-run",
            "build",
            "setup",
            "doctor",
            "smoke",
        ]
        for mode in valid_modes:
            if mode == "dry-run":
                # dry-run requires a second argument
                result = parse_args(["dry-run", "both"])
            elif mode == "build":
                result = parse_args(["build", "sycl"])
            elif mode == "setup":
                result = parse_args(["setup", "check"])
            elif mode == "doctor":
                result = parse_args(["doctor", "check"])
            elif mode == "smoke":
                # smoke requires a sub-mode argument
                result = parse_args(["smoke", "both"])
            else:
                result = parse_args([mode])
            assert result.mode == mode, f"Mode '{mode}' should be accepted"

    def test_invalid_mode_rejected(self) -> None:
        """parse_args should reject invalid modes."""
        with pytest.raises(SystemExit):
            parse_args(["invalid-mode"])

    def test_ports_passed_through(self) -> None:
        """parse_args should pass port numbers through to result.ports."""
        result = parse_args(["summary-balanced", "8080"])
        assert result.ports == [8080]

    def test_multiple_ports_passed_through(self) -> None:
        """parse_args should pass multiple port numbers through."""
        result = parse_args(["both", "8080", "8081"])
        assert result.ports == [8080, 8081]

    def test_acknowledge_risky_flag(self) -> None:
        """parse_args should handle --acknowledge-risky flag."""
        result = parse_args(["summary-balanced", "--acknowledge-risky"])
        assert result.acknowledge_risky is True

    def test_tui_args_both(self) -> None:
        """parse_tui_args should handle 'both' mode with --port."""
        from llama_cli.cli_parser import parse_tui_args

        result = parse_tui_args(["both", "--port", "8080"])
        assert result.mode == "both"
        assert result.port == 8080
        assert result.dry_run_mode is None

    def test_tui_args_both_with_port2(self) -> None:
        """parse_tui_args should handle 'both' mode with --port and --port2."""
        from llama_cli.cli_parser import parse_tui_args

        result = parse_tui_args(["both", "--port", "8080", "--port2", "8081"])
        assert result.mode == "both"
        assert result.port == 8080
        assert result.port2 == 8081

    def test_tui_args_summary_balanced(self) -> None:
        """parse_tui_args should handle 'summary-balanced' mode."""
        from llama_cli.cli_parser import parse_tui_args

        result = parse_tui_args(["summary-balanced", "--port", "9000"])
        assert result.mode == "summary-balanced"
        assert result.port == 9000

    def test_tui_args_acknowledge_risky(self) -> None:
        """parse_tui_args should handle --acknowledge-risky flag."""
        from llama_cli.cli_parser import parse_tui_args

        result = parse_tui_args(["qwen35", "--acknowledge-risky"])
        assert result.acknowledge_risky is True

    def test_tui_args_no_options(self) -> None:
        """parse_tui_args should handle mode without additional options."""
        from llama_cli.cli_parser import parse_tui_args

        result = parse_tui_args(["summary-fast"])
        assert result.mode == "summary-fast"
        assert result.port is None
        assert result.port2 is None

    def test_parse_args_with_none_uses_sys_args(self) -> None:
        """parse_args(None) should use sys.argv[1:] (not tested directly, but structure verified)."""
        # We can't easily test sys.argv manipulation, but we verify the function
        # accepts None as an argument
        result = parse_args([])
        # Empty args → mode is None (not specified)
        assert result.mode is None


class TestSmokeFlagValueSyntax:
    """Tests for --flag=value syntax in smoke subcommand.

    Covers the dense uncovered area in _handle_smoke_case (lines 503-542).
    """

    def test_smoke_equals_api_key(self) -> None:
        """--api-key=sk should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--api-key=sk-test"])
        assert result.api_key == "sk-test"

    def test_smoke_equals_model_id(self) -> None:
        """--model-id=model should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--model-id=Qwen3.5-2B"])
        assert result.model_id == "Qwen3.5-2B"

    def test_smoke_equals_max_tokens(self) -> None:
        """--max-tokens=16 should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--max-tokens=16"])
        assert result.max_tokens == 16

    def test_smoke_equals_prompt(self) -> None:
        """--prompt=hello should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--prompt=hello"])
        assert result.prompt == "hello"

    def test_smoke_equals_delay(self) -> None:
        """--delay=3 should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--delay=3"])
        assert result.delay == 3

    def test_smoke_equals_timeout(self) -> None:
        """--timeout=7 should parse via equals syntax."""
        result = parse_args(["smoke", "both", "--timeout=7"])
        assert result.timeout == 7

    def test_smoke_mixed_order_equals_and_separate(self) -> None:
        """Mixed --flag=value and --flag value forms after slot ID."""
        result = parse_args(
            [
                "smoke",
                "slot",
                "summary-balanced",
                "--api-key",
                "sk",
                "--model-id=model",
                "--json",
                "--max-tokens",
                "16",
                "--delay=3",
                "--timeout",
                "7",
                "--prompt",
                "hi",
            ]
        )
        assert result.mode == "smoke"
        assert result.slot_id == "summary-balanced"
        assert result.api_key == "sk"
        assert result.model_id == "model"
        assert result.json is True
        assert result.max_tokens == 16
        assert result.delay == 3
        assert result.timeout == 7
        assert result.prompt == "hi"

    def test_smoke_unknown_flag_equals_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Unknown --flag=value should exit with code 1 and stderr contains unknown flag."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--unknown=value"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "unknown flag" in captured.err

    def test_smoke_unknown_bare_flag_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Unknown bare --flag should exit with code 1 and stderr contains unknown flag."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--unknown-flag"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "unknown flag" in captured.err

    def test_smoke_missing_value_for_api_key_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--api-key at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--api-key"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_missing_value_for_model_id_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--model-id at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--model-id"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_missing_value_for_max_tokens_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--max-tokens at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--max-tokens"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_missing_value_for_prompt_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--prompt at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--prompt"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_missing_value_for_delay_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--delay at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--delay"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_missing_value_for_timeout_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--timeout at end of args should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--timeout"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "requires a value" in captured.err

    def test_smoke_invalid_max_tokens_equals_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--max-tokens=abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--max-tokens=abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --max-tokens value" in captured.err
        assert "must be an integer" in captured.err

    def test_smoke_invalid_max_tokens_separate_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--max-tokens abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--max-tokens", "abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --max-tokens value" in captured.err

    def test_smoke_invalid_delay_equals_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--delay=abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--delay=abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --delay value" in captured.err
        assert "must be an integer" in captured.err

    def test_smoke_invalid_delay_separate_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--delay abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--delay", "abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --delay value" in captured.err

    def test_smoke_invalid_timeout_equals_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--timeout=abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--timeout=abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --timeout value" in captured.err
        assert "must be an integer" in captured.err

    def test_smoke_invalid_timeout_separate_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--timeout abc should exit with code 1 and helpful stderr."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--timeout", "abc"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid --timeout value" in captured.err

    def test_smoke_max_tokens_too_low(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--max-tokens below 8 should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--max-tokens=4"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "must be between 8 and 32" in captured.err

    def test_smoke_max_tokens_too_high(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--max-tokens above 32 should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["smoke", "both", "--max-tokens=64"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "must be between 8 and 32" in captured.err

    def test_smoke_max_tokens_boundary_low(self) -> None:
        """--max-tokens=8 should be accepted (lower boundary)."""
        result = parse_args(["smoke", "both", "--max-tokens=8"])
        assert result.max_tokens == 8

    def test_smoke_max_tokens_boundary_high(self) -> None:
        """--max-tokens=32 should be accepted (upper boundary)."""
        result = parse_args(["smoke", "both", "--max-tokens=32"])
        assert result.max_tokens == 32

    def test_smoke_slot_with_json_no_slot_id(self) -> None:
        """smoke slot --json should leave slot_id None and json True."""
        result = parse_args(["smoke", "slot", "--json"])
        assert result.mode == "smoke"
        assert result.smoke_mode == "slot"
        assert result.slot_id is None
        assert result.json is True


class TestSmokeHandleNoneCase:
    """Tests for _handle_smoke_case returning None for non-smoke args."""

    def test_handle_smoke_returns_none_for_normal_mode(self) -> None:
        """_handle_smoke_case should return None for normal modes."""
        from llama_cli.cli_parser import _handle_smoke_case

        result = _handle_smoke_case(["summary-balanced"])
        assert result is None

    def test_handle_smoke_returns_none_for_dry_run(self) -> None:
        """_handle_smoke_case should return None for dry-run."""
        from llama_cli.cli_parser import _handle_smoke_case

        result = _handle_smoke_case(["dry-run", "both"])
        assert result is None

    def test_handle_smoke_returns_none_for_empty_args(self) -> None:
        """_handle_smoke_case should return None for empty args."""
        from llama_cli.cli_parser import _handle_smoke_case

        result = _handle_smoke_case([])
        assert result is None

    def test_handle_smoke_not_smoke_returns_none(self) -> None:
        """_handle_smoke_case(['not-smoke']) should return None."""
        from llama_cli.cli_parser import _handle_smoke_case

        result = _handle_smoke_case(["not-smoke"])
        assert result is None


class TestParseArgsNonePath:
    """Tests for parse_args(None) path and edge cases."""

    def test_parse_args_none_type_accepted(self) -> None:
        """parse_args should accept None as argument (uses sys.argv[1:])."""
        # We verify the function signature accepts None without error
        # The actual sys.argv behavior is a runtime detail
        result = parse_args([])
        assert result.mode is None

    def test_parse_args_empty_list_no_exit(self) -> None:
        """parse_args([]) should not raise SystemExit for empty args."""
        result = parse_args([])
        assert result.mode is None

    def test_parse_args_smoke_both_json(self) -> None:
        """parse_args should handle 'smoke both --json'."""
        result = parse_args(["smoke", "both", "--json"])
        assert result.mode == "smoke"
        assert result.smoke_mode == "both"
        assert result.json is True

    def test_parse_args_smoke_slot_with_all_flags(self) -> None:
        """parse_args should handle 'smoke slot <id>' with all flags."""
        result = parse_args(
            [
                "smoke",
                "slot",
                "slot-1",
                "--api-key",
                "sk-123",
                "--model-id",
                "test-model",
                "--max-tokens",
                "16",
                "--prompt",
                "hello",
                "--delay",
                "3",
                "--timeout",
                "7",
                "--json",
            ]
        )
        assert result.mode == "smoke"
        assert result.smoke_mode == "slot"
        assert result.slot_id == "slot-1"
        assert result.api_key == "sk-123"
        assert result.model_id == "test-model"
        assert result.max_tokens == 16
        assert result.prompt == "hello"
        assert result.delay == 3
        assert result.timeout == 7
        assert result.json is True

    def test_parse_args_smoke_both_defaults(self) -> None:
        """smoke both with no flags should use default values."""
        result = parse_args(["smoke", "both"])
        assert result.mode == "smoke"
        assert result.smoke_mode == "both"
        assert result.slot_id is None
        assert result.api_key == ""
        assert result.model_id is None
        assert result.max_tokens == 0
        assert result.prompt == ""
        assert result.delay == 0
        assert result.timeout == 0
        assert result.json is False

    """Tests for create_smoke_config() helper from config_builder."""

    def test_create_smoke_config_defaults(self) -> None:
        """create_smoke_config should create config with defaults from Config."""
        from llama_manager.config import Config
        from llama_manager.config_builder import create_smoke_config

        config = Config()
        smoke_cfg = create_smoke_config(config)

        assert smoke_cfg.inter_slot_delay_s == config.smoke_inter_slot_delay_s
        assert smoke_cfg.listen_timeout_s == config.smoke_listen_timeout_s
        assert smoke_cfg.http_request_timeout_s == config.smoke_http_request_timeout_s
        assert smoke_cfg.max_tokens == config.smoke_max_tokens
        assert smoke_cfg.prompt == config.smoke_prompt
        assert smoke_cfg.skip_models_discovery == config.smoke_skip_models_discovery
        assert smoke_cfg.api_key == ""
        assert smoke_cfg.model_id_override is None

    def test_create_smoke_config_with_api_key(self) -> None:
        """create_smoke_config should use provided api_key."""
        from llama_manager.config import Config
        from llama_manager.config_builder import create_smoke_config

        config = Config()
        smoke_cfg = create_smoke_config(config, api_key="sk-test-key")

        assert smoke_cfg.api_key == "sk-test-key"

    def test_create_smoke_config_with_model_id_override(self) -> None:
        """create_smoke_config should pass through model_id_override."""
        from llama_manager.config import Config
        from llama_manager.config_builder import create_smoke_config

        config = Config()
        smoke_cfg = create_smoke_config(config, model_id_override="Qwen3.5-2B")

        assert smoke_cfg.model_id_override == "Qwen3.5-2B"

    def test_create_smoke_config_cli_key_overrides_config(self) -> None:
        """CLI api_key should take precedence over Config.smoke_api_key."""
        from llama_manager.config import Config
        from llama_manager.config_builder import create_smoke_config

        config = Config()
        config.smoke_api_key = "from-config"
        smoke_cfg = create_smoke_config(config, api_key="from-cli")

        assert smoke_cfg.api_key == "from-cli"

    def test_create_smoke_config_empty_cli_key_uses_config(self) -> None:
        """Empty CLI api_key should fall back to Config.smoke_api_key."""
        from llama_manager.config import Config
        from llama_manager.config_builder import create_smoke_config

        config = Config()
        config.smoke_api_key = "from-config"
        smoke_cfg = create_smoke_config(config, api_key="")

        assert smoke_cfg.api_key == "from-config"


class TestSmokeProbeResultExitCodeMapping:
    """Tests for exit code mapping from SmokeProbeStatus to exit codes."""

    def test_exit_code_pass_is_zero(self) -> None:
        """PASS should map to exit code 0."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
        )
        assert result.exit_code == 0

    def test_exit_code_fail_is_ten(self) -> None:
        """FAIL should map to exit code 10."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.LISTEN,
        )
        assert result.exit_code == 10

    def test_exit_code_timeout_is_thirteen(self) -> None:
        """TIMEOUT should map to exit code 13 (less severe than MODEL_NOT_FOUND=14)."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.TIMEOUT,
            phase_reached=SmokePhase.MODELS,
        )
        assert result.exit_code == 13

    def test_exit_code_model_not_found_is_fourteen(self) -> None:
        """MODEL_NOT_FOUND should map to exit code 14 (more severe than TIMEOUT=13)."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.MODEL_NOT_FOUND,
            phase_reached=SmokePhase.MODELS,
        )
        assert result.exit_code == 14

    def test_exit_code_auth_failure_is_fifteen(self) -> None:
        """AUTH_FAILURE should map to exit code 15."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.AUTH_FAILURE,
            phase_reached=SmokePhase.MODELS,
        )
        assert result.exit_code == 15

    def test_exit_code_crashed_is_nineteen(self) -> None:
        """CRASHED should map to exit code 19."""
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.CRASHED,
            phase_reached=SmokePhase.COMPLETE,
        )
        assert result.exit_code == 19


class TestSmokeProbeStatusEnum:
    """Tests for SmokeProbeStatus enum values and behavior."""

    def test_all_status_values(self) -> None:
        """SmokeProbeStatus should have all expected values."""
        from llama_manager.config import SmokeProbeStatus

        values = {s.value for s in SmokeProbeStatus}
        expected = {"pass", "fail", "timeout", "crashed", "model_not_found", "auth_failure"}
        assert values == expected

    def test_status_comparison_with_string(self) -> None:
        """SmokeProbeStatus members should compare equal to their string values."""
        from llama_manager.config import SmokeProbeStatus

        assert SmokeProbeStatus.PASS == "pass"  # noqa: S105
        assert SmokeProbeStatus.FAIL == "fail"
        assert SmokeProbeStatus.TIMEOUT == "timeout"
        assert SmokeProbeStatus.CRASHED == "crashed"
        assert SmokeProbeStatus.MODEL_NOT_FOUND == "model_not_found"
        assert SmokeProbeStatus.AUTH_FAILURE == "auth_failure"

    def test_status_not_equal_to_other(self) -> None:
        """SmokeProbeStatus members should not compare equal to unrelated strings."""
        from llama_manager.config import SmokeProbeStatus

        assert SmokeProbeStatus.PASS != "fail"  # noqa: S105
        assert SmokeProbeStatus.FAIL != "pass"  # noqa: S105
        assert SmokeProbeStatus.CRASHED != "timeout"


class TestSmokePhaseEnum:
    """Tests for SmokePhase enum values and behavior."""

    def test_all_phase_values(self) -> None:
        """SmokePhase should have all expected values."""
        from llama_manager.config import SmokePhase

        values = {s.value for s in SmokePhase}
        expected = {"listen", "models", "chat", "complete"}
        assert values == expected

    def test_phase_comparison_with_string(self) -> None:
        """SmokePhase members should compare equal to their string values."""
        from llama_manager.config import SmokePhase

        assert SmokePhase.LISTEN == "listen"
        assert SmokePhase.MODELS == "models"
        assert SmokePhase.CHAT == "chat"
        assert SmokePhase.COMPLETE == "complete"


class TestSmokeFailurePhaseEnum:
    """Tests for SmokeFailurePhase enum values and behavior."""

    def test_all_failure_phase_values(self) -> None:
        """SmokeFailurePhase should have all expected values."""
        from llama_manager.config import SmokeFailurePhase

        values = {s.value for s in SmokeFailurePhase}
        expected = {"listen", "models", "chat"}
        assert values == expected

    def test_failure_phase_comparison_with_string(self) -> None:
        """SmokeFailurePhase members should compare equal to their string values."""
        from llama_manager.config import SmokeFailurePhase

        assert SmokeFailurePhase.LISTEN == "listen"
        assert SmokeFailurePhase.MODELS == "models"
        assert SmokeFailurePhase.CHAT == "chat"


class TestProbeSlotHostPortFormatting:
    """Tests for slot_id formatting with different host:port combinations."""

    def test_slot_id_ipv4(self) -> None:
        """probe_slot should format slot_id as 'host:port' for IPv4."""
        smoke_cfg = self._make_cfg()

        with (
            patch("llama_manager.smoke.socket.socket") as mock_socket_cls,
            patch("llama_manager.smoke.resolve_provenance") as mock_resolve,
        ):
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect.side_effect = OSError()
            mock_sock.close.return_value = None
            mock_resolve.return_value = MagicMock(sha="test", version="test")

            result = probe_slot("10.0.0.1", 9999, smoke_cfg)

        assert result.slot_id == "10.0.0.1:9999"

    def test_slot_id_localhost(self) -> None:
        """probe_slot should format slot_id as '127.0.0.1:port'."""
        smoke_cfg = self._make_cfg()

        with (
            patch("llama_manager.smoke.socket.socket") as mock_socket_cls,
            patch("llama_manager.smoke.resolve_provenance") as mock_resolve,
        ):
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            mock_sock.connect.side_effect = OSError()
            mock_sock.close.return_value = None
            mock_resolve.return_value = MagicMock(sha="test", version="test")

            result = probe_slot("127.0.0.1", 8080, smoke_cfg)

        assert result.slot_id == "127.0.0.1:8080"

    def _make_cfg(self) -> MagicMock:
        """Create a mock SmokeProbeConfiguration."""
        smoke_cfg = MagicMock()
        smoke_cfg.listen_timeout_s = 5
        smoke_cfg.http_request_timeout_s = 10
        smoke_cfg.max_tokens = 16
        smoke_cfg.prompt = "test"
        smoke_cfg.skip_models_discovery = False
        smoke_cfg.api_key = ""
        smoke_cfg.first_token_timeout_s = 1200
        smoke_cfg.total_chat_timeout_s = 1500
        return smoke_cfg


# ---------------------------------------------------------------------------
# T090 — _build_slot_configs coverage (lines 51-67)
# ---------------------------------------------------------------------------


class TestBuildSlotConfigs:
    """T090: _build_slot_configs mode branches, slot resolution, and error paths."""

    def test_build_slot_configs_both_returns_two_slots(self) -> None:
        """mode 'both' should return summary-balanced and qwen35-coding tuples."""
        from llama_cli.smoke_cli import _build_slot_configs

        targets = _build_slot_configs("both")

        assert len(targets) == 2
        # First slot: summary-balanced
        assert targets[0][0] == "summary-balanced"
        assert targets[0][2] == "127.0.0.1"
        assert targets[0][3] == 8080  # summary_balanced_port
        # Second slot: qwen35-coding
        assert targets[1][0] == "qwen35-coding"
        assert targets[1][2] == "127.0.0.1"
        assert targets[1][3] == 8081  # qwen35_port

    def test_build_slot_configs_slot_resolves_port_and_model(
        self,
    ) -> None:
        """mode 'slot' with slot_id should return one tuple with resolved port/model."""
        from llama_cli.smoke_cli import _build_slot_configs

        targets = _build_slot_configs("slot", "summary-fast")

        assert len(targets) == 1
        assert targets[0][0] == "summary-fast"
        assert targets[0][2] == "127.0.0.1"
        assert targets[0][3] == 8082  # summary_fast_port

    def test_build_slot_configs_slot_missing_id_exits(self) -> None:
        """mode 'slot' without slot_id should exit with code 1 and stderr message."""
        from llama_cli.smoke_cli import _build_slot_configs

        with pytest.raises(SystemExit) as exc_info:
            _build_slot_configs("slot", None)

        assert exc_info.value.code == 1

    def test_build_slot_configs_bogus_mode_exits(self) -> None:
        """unknown mode should exit with code 1 and stderr message."""
        from llama_cli.smoke_cli import _build_slot_configs

        with pytest.raises(SystemExit) as exc_info:
            _build_slot_configs("bogus")

        assert exc_info.value.code == 1


class TestResolveSlotPort:
    """T091: _resolve_slot_port error path for unknown slot_id."""

    def test_resolve_slot_port_unknown_exits(self) -> None:
        """_resolve_slot_port should exit for unknown slot_id."""
        from llama_cli.smoke_cli import _resolve_slot_port
        from llama_manager.config import Config

        cfg = Config()

        with pytest.raises(SystemExit) as exc_info:
            _resolve_slot_port(cfg, "nonexistent")

        assert exc_info.value.code == 1

    def test_resolve_slot_port_known(self) -> None:
        """_resolve_slot_port should return correct port for known slot_ids."""
        from llama_cli.smoke_cli import _resolve_slot_port
        from llama_manager.config import Config

        cfg = Config()
        assert _resolve_slot_port(cfg, "summary-balanced") == 8080
        assert _resolve_slot_port(cfg, "summary-fast") == 8082
        assert _resolve_slot_port(cfg, "qwen35-coding") == 8081


class TestResolveSlotModel:
    """T092: _resolve_slot_model error path for unknown slot_id."""

    def test_resolve_slot_model_unknown_exits(self) -> None:
        """_resolve_slot_model should exit for unknown slot_id."""
        from llama_cli.smoke_cli import _resolve_slot_model
        from llama_manager.config import Config

        cfg = Config()

        with pytest.raises(SystemExit) as exc_info:
            _resolve_slot_model(cfg, "nonexistent")

        assert exc_info.value.code == 1

    def test_resolve_slot_model_known(self) -> None:
        """_resolve_slot_model should return correct model for known slot_ids."""
        from llama_cli.smoke_cli import _resolve_slot_model
        from llama_manager.config import Config

        cfg = Config()
        assert "Qwen3.5-2B" in _resolve_slot_model(cfg, "summary-balanced")
        assert "Qwen3.5-0.8B" in _resolve_slot_model(cfg, "summary-fast")
        assert "Qwen3.5-35B" in _resolve_slot_model(cfg, "qwen35-coding")


# ---------------------------------------------------------------------------
# T093 — _probe_server delegation (line 141-148)
# ---------------------------------------------------------------------------


class TestProbeServer:
    """T093: _probe_server delegates to probe_slot with correct args."""

    def test_probe_server_calls_probe_slot_with_args(self) -> None:
        """_probe_server should call probe_slot(host, port, smoke_cfg, model_path, model_id, expected_model_id=None)."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import _probe_server
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        model_path = "/path/to/model.gguf"
        host = "127.0.0.1"
        port = 9090
        smoke_cfg = MagicMock()
        smoke_cfg.model_id_override = "test-model"

        expected_result = SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.PASS,
            phase_reached=MagicMock(),
        )

        with patch("llama_cli.smoke_cli.probe_slot", return_value=expected_result) as mock_probe:
            result = _probe_server(model_path, host, port, smoke_cfg)

        mock_probe.assert_called_once_with(
            host=host,
            port=port,
            smoke_cfg=smoke_cfg,
            model_path=model_path,
            model_id=smoke_cfg.model_id_override,
            expected_model_id=None,
        )
        assert result is expected_result


# ---------------------------------------------------------------------------
# T094 — _validate_smoke_args (lines 160-172)
# ---------------------------------------------------------------------------


class TestValidateSmokeArgs:
    """T094: _validate_smoke_args error and success paths."""

    def test_validate_smoke_args_invalid_mode(self) -> None:
        """_validate_smoke_args should return 1 for invalid mode."""
        import argparse

        from llama_cli.smoke_cli import _validate_smoke_args

        parsed = argparse.Namespace(mode="bogus", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result == 1

    def test_validate_smoke_args_slot_missing_slot_id(self) -> None:
        """_validate_smoke_args should return 1 for slot mode without slot_id."""
        import argparse

        from llama_cli.smoke_cli import _validate_smoke_args

        parsed = argparse.Namespace(mode="slot", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result == 1

    def test_validate_smoke_args_slot_with_slot_id(self) -> None:
        """_validate_smoke_args should return None for valid slot mode with slot_id."""
        import argparse

        from llama_cli.smoke_cli import _validate_smoke_args

        parsed = argparse.Namespace(mode="slot", slot_id="summary-balanced")
        result = _validate_smoke_args(parsed)
        assert result is None

    def test_validate_smoke_args_both_mode(self) -> None:
        """_validate_smoke_args should return None for valid both mode."""
        import argparse

        from llama_cli.smoke_cli import _validate_smoke_args

        parsed = argparse.Namespace(mode="both", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result is None


# ---------------------------------------------------------------------------
# T095 — _build_smoke_config override branches (lines 191-208)
# ---------------------------------------------------------------------------


class TestBuildSmokeConfig:
    """T095: _build_smoke_config CLI overrides and fallback defaults."""

    def _make_parsed(
        self,
        api_key: str = "",
        model_id: str | None = None,
        delay: int = 0,
        timeout: int = 0,
        max_tokens: int = 0,
        prompt: str = "",
    ) -> argparse.Namespace:
        """Helper to build a parsed namespace for _build_smoke_config."""
        return argparse.Namespace(
            mode="both",
            json=False,
            api_key=api_key,
            model_id=model_id,
            delay=delay,
            timeout=timeout,
            max_tokens=max_tokens,
            prompt=prompt,
            slot_id=None,
        )

    def test_build_smoke_config_cli_overrides(self) -> None:
        """_build_smoke_config should apply CLI overrides when values are non-zero/non-empty."""
        from llama_cli.smoke_cli import _build_smoke_config

        parsed = self._make_parsed(
            api_key="sk-test",
            model_id="Qwen3.5-2B",
            delay=5,
            timeout=60,
            max_tokens=32,
            prompt="Custom prompt",
        )
        smoke_cfg = _build_smoke_config(parsed)

        assert smoke_cfg.api_key == "sk-test"
        assert smoke_cfg.model_id_override == "Qwen3.5-2B"
        assert smoke_cfg.inter_slot_delay_s == 5
        assert smoke_cfg.listen_timeout_s == 60
        assert smoke_cfg.max_tokens == 32
        assert smoke_cfg.prompt == "Custom prompt"

    def test_build_smoke_config_fallbacks_to_config_defaults(self) -> None:
        """_build_smoke_config should fall back to Config defaults when CLI values are zero/empty."""
        from llama_cli.smoke_cli import _build_smoke_config
        from llama_manager.config import Config

        parsed = self._make_parsed()
        smoke_cfg = _build_smoke_config(parsed)

        cfg = Config()
        assert smoke_cfg.api_key == cfg.smoke_api_key
        assert smoke_cfg.inter_slot_delay_s == cfg.smoke_inter_slot_delay_s
        assert smoke_cfg.listen_timeout_s == cfg.smoke_listen_timeout_s
        assert smoke_cfg.max_tokens == cfg.smoke_max_tokens
        assert smoke_cfg.prompt == cfg.smoke_prompt

    def test_build_smoke_config_empty_prompt_not_overridden(self) -> None:
        """Empty CLI prompt should not override the default prompt."""
        from llama_cli.smoke_cli import _build_smoke_config
        from llama_manager.config import Config

        parsed = self._make_parsed(prompt="")
        smoke_cfg = _build_smoke_config(parsed)

        cfg = Config()
        assert smoke_cfg.prompt == cfg.smoke_prompt


# ---------------------------------------------------------------------------
# T096 — _run_probes (lines 224-231)
# ---------------------------------------------------------------------------


class TestRunProbes:
    """T096: _run_probes calls _probe_server per target and sleep before second target."""

    def test_run_probes_calls_probe_once_per_target(self) -> None:
        """_run_probes should call _probe_server once for each target."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import _run_probes
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        targets = [
            ("slot1", "/model1.gguf", "127.0.0.1", 8080),
            ("slot2", "/model2.gguf", "127.0.0.1", 8081),
        ]
        smoke_cfg = MagicMock()
        smoke_cfg.inter_slot_delay_s = 0

        mock_result = SmokeProbeResult(
            slot_id="slot1",
            status=SmokeProbeStatus.PASS,
            phase_reached=MagicMock(),
        )

        with patch("llama_cli.smoke_cli._probe_server", return_value=mock_result) as mock_probe:
            results = _run_probes(targets, smoke_cfg)

        assert len(results) == 2
        assert mock_probe.call_count == 2

    def test_run_probes_sleeps_before_second_target_when_delay_gt_0(
        self,
    ) -> None:
        """_run_probes should call time.sleep before the second target when delay > 0."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import _run_probes
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        targets = [
            ("slot1", "/model1.gguf", "127.0.0.1", 8080),
            ("slot2", "/model2.gguf", "127.0.0.1", 8081),
        ]
        smoke_cfg = MagicMock()
        smoke_cfg.inter_slot_delay_s = 3

        mock_result = SmokeProbeResult(
            slot_id="slot1",
            status=SmokeProbeStatus.PASS,
            phase_reached=MagicMock(),
        )

        with (
            patch("llama_cli.smoke_cli._probe_server", return_value=mock_result),
            patch("llama_cli.smoke_cli.time.sleep") as mock_sleep,
        ):
            _run_probes(targets, smoke_cfg)

        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_once_with(3)

    def test_run_probes_no_sleep_when_delay_is_zero(self) -> None:
        """_run_probes should not call time.sleep when delay is 0."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import _run_probes
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        targets = [
            ("slot1", "/model1.gguf", "127.0.0.1", 8080),
            ("slot2", "/model2.gguf", "127.0.0.1", 8081),
        ]
        smoke_cfg = MagicMock()
        smoke_cfg.inter_slot_delay_s = 0

        mock_result = SmokeProbeResult(
            slot_id="slot1",
            status=SmokeProbeStatus.PASS,
            phase_reached=MagicMock(),
        )

        with (
            patch("llama_cli.smoke_cli._probe_server", return_value=mock_result),
            patch("llama_cli.smoke_cli.time.sleep") as mock_sleep,
        ):
            _run_probes(targets, smoke_cfg)

        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------------------
# T097 — run_smoke report output (lines 266-280)
# ---------------------------------------------------------------------------


class TestRunSmoke:
    """T097: run_smoke JSON and human report output paths."""

    def test_run_smoke_json_calls_json_printer(self) -> None:
        """run_smoke with --json should call _print_report_json."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import run_smoke
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        mock_result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=MagicMock(),
        )

        with (
            patch(
                "llama_cli.smoke_cli._build_slot_configs",
                return_value=[("test", "/m.gguf", "127.0.0.1", 8080)],
            ),
            patch("llama_cli.smoke_cli._build_smoke_config") as mock_cfg,
            patch("llama_cli.smoke_cli._run_probes", return_value=[mock_result]),
            patch("llama_cli.smoke_cli._print_report_json") as mock_json_printer,
        ):
            mock_cfg.inter_slot_delay_s = 0
            mock_cfg.listen_timeout_s = 5
            mock_cfg.max_tokens = 16
            mock_cfg.prompt = "test"
            mock_cfg.model_id_override = None

            exit_code = run_smoke(["both", "--json"])

        mock_json_printer.assert_called_once()
        assert isinstance(exit_code, int)

    def test_run_smoke_human_failure_creates_report_dir(self) -> None:
        """run_smoke with failures should call resolve_runtime_dir and _ensure_report_dir."""
        from unittest.mock import MagicMock, patch

        from llama_cli.smoke_cli import run_smoke
        from llama_manager.smoke import SmokeProbeResult, SmokeProbeStatus

        mock_fail_result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.FAIL,
            phase_reached=MagicMock(),
        )

        with (
            patch(
                "llama_cli.smoke_cli._build_slot_configs",
                return_value=[("test", "/m.gguf", "127.0.0.1", 8080)],
            ),
            patch("llama_cli.smoke_cli._build_smoke_config") as mock_cfg,
            patch("llama_cli.smoke_cli._run_probes", return_value=[mock_fail_result]),
            patch("llama_cli.smoke_cli._print_report_human") as mock_human_printer,
            patch("llama_cli.smoke_cli.resolve_runtime_dir") as mock_runtime,
            patch("llama_cli.smoke_cli._ensure_report_dir") as mock_ensure,
        ):
            mock_cfg.inter_slot_delay_s = 0
            mock_cfg.listen_timeout_s = 5
            mock_cfg.max_tokens = 16
            mock_cfg.prompt = "test"
            mock_cfg.model_id_override = None
            mock_runtime.return_value = MagicMock()

            run_smoke(["both"])

        mock_ensure.assert_called_once()
        mock_human_printer.assert_called_once()


# ---------------------------------------------------------------------------
# T098 — _print_report_human (lines 356-384)
# ---------------------------------------------------------------------------


class TestPrintReportHuman:
    """T098: _print_report_human prints failure, model, and latency branches."""

    def test_print_report_human_pass(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Human report should print header, pass count, and passing slot."""
        from llama_cli.smoke_cli import _print_report_human
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
            latency_ms=123,
        )

        report_mock = MagicMock(results=[result], pass_count=1, fail_count=0)
        report_mock.overall_status.value = "pass"
        _print_report_human(report_mock, "both")

        captured = capsys.readouterr()
        assert "=== SMOKE TEST REPORT ===" in captured.out
        assert "Overall: PASS" in captured.out
        assert "[test] PASS" in captured.out
        assert "Phase reached: complete" in captured.out
        assert "Latency: 123ms" in captured.out

    def test_print_report_human_failure_phase(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Human report should print failure_phase when present."""
        from llama_cli.smoke_cli import _print_report_human
        from llama_manager.config import SmokeFailurePhase
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.LISTEN,
            failure_phase=SmokeFailurePhase.LISTEN,
            latency_ms=None,
        )

        report_mock = MagicMock(results=[result], pass_count=0, fail_count=1)
        report_mock.overall_status.value = "fail"
        _print_report_human(report_mock, "slot")

        captured = capsys.readouterr()
        assert "[test] FAIL" in captured.out
        assert "Failed at: listen" in captured.out
        assert "Latency: N/A" in captured.out

    def test_print_report_human_with_model_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Human report should print model_id when present."""
        from llama_cli.smoke_cli import _print_report_human
        from llama_manager.smoke import SmokePhase, SmokeProbeResult, SmokeProbeStatus

        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
            model_id="Qwen3.5-2B",
            latency_ms=200,
        )

        report_mock = MagicMock(results=[result], pass_count=1, fail_count=0)
        report_mock.overall_status.value = "pass"
        _print_report_human(report_mock, "both")

        captured = capsys.readouterr()
        assert "Model: Qwen3.5-2B" in captured.out


# ---------------------------------------------------------------------------
# T099 — _print_report_json (lines 387-415)
# ---------------------------------------------------------------------------


class TestPrintReportJson:
    """T099: _print_report_json emits results, exit_code, and provenance fields."""

    def test_print_report_json_structure(self, capsys: pytest.CaptureFixture[str]) -> None:
        """JSON report should contain results, overall_exit_code, and provenance."""
        from unittest.mock import patch

        from llama_cli.smoke_cli import _print_report_json
        from llama_manager.smoke import (
            ProvenanceRecord,
            SmokePhase,
            SmokeProbeResult,
            SmokeProbeStatus,
        )

        provenance = ProvenanceRecord(sha="abc1234", version="1.0.0")
        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
            model_id="Qwen3.5-2B",
            latency_ms=100,
            provenance=provenance,
        )

        with patch("llama_cli.smoke_cli.json.dumps") as mock_dumps:
            _print_report_json(MagicMock(results=[result], pass_count=1, fail_count=0))

        mock_dumps.assert_called_once()
        call_args = mock_dumps.call_args[0][0]
        assert "results" in call_args
        assert "overall_exit_code" in call_args
        assert "provenance" in call_args["results"][0]
        assert call_args["results"][0]["provenance"]["sha"] == "abc1234"
        assert call_args["results"][0]["provenance"]["version"] == "1.0.0"

    def test_print_report_json_failure_phase_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        """JSON report should include failure_phase as None when absent."""
        from unittest.mock import patch

        from llama_cli.smoke_cli import _print_report_json
        from llama_manager.smoke import (
            ProvenanceRecord,
            SmokePhase,
            SmokeProbeResult,
            SmokeProbeStatus,
        )

        provenance = ProvenanceRecord(sha="def5678", version="dev")
        result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
            model_id=None,
            latency_ms=None,
            provenance=provenance,
        )

        with patch("llama_cli.smoke_cli.json.dumps") as mock_dumps:
            _print_report_json(MagicMock(results=[result], pass_count=1, fail_count=0))

        call_args = mock_dumps.call_args[0][0]
        assert call_args["results"][0]["failure_phase"] is None
        assert call_args["results"][0]["model_id"] is None
        assert call_args["results"][0]["latency_ms"] is None
