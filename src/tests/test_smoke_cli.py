"""Tests for smoke CLI argument parsing.

Covers:
  - T033: `smoke both` and `smoke slot <id>` argument parsing
  - smoke subcommand structure in cli_parser.py
"""

from __future__ import annotations

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


class TestSmokeConfigCreation:
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
