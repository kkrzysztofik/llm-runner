"""Additional smoke CLI tests to cover uncovered branches.

Covers:
  - Unknown slot in _build_slot_configs (lines 73-78)
  - _validate_smoke_args returning None path (line 258)
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from llama_cli.commands.smoke import (
    _build_slot_configs,
    _validate_smoke_args,
)
from llama_manager.config import SmokePhase, SmokeProbeStatus
from llama_manager.probe import SmokeProbeResult


class TestBuildSlotConfigsUnknownSlot:
    """Tests for _build_slot_configs with unknown slot_id."""

    def test_build_slot_configs_unknown_slot_exits(self) -> None:
        """_build_slot_configs with unknown slot_id should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _build_slot_configs("slot", "unknown-slot")

        assert exc_info.value.code == 1

    def test_build_slot_configs_unknown_slot_stderr_message(self, capsys) -> None:
        """_build_slot_configs unknown slot should list valid slots in stderr."""
        with pytest.raises(SystemExit) as exc_info:
            _build_slot_configs("slot", "nonexistent")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "unknown slot" in captured.err.lower()
        assert "summary-balanced" in captured.err or "Valid slots" in captured.err


class TestValidateSmokeArgsNonePath:
    """Tests for _validate_smoke_args returning None (valid path)."""

    def test_validate_both_mode_returns_none(self) -> None:
        """_validate_smoke_args should return None for valid 'both' mode."""
        parsed = argparse.Namespace(mode="both", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result is None

    def test_validate_slot_mode_with_id_returns_none(self) -> None:
        """_validate_smoke_args should return None for valid 'slot' mode with slot_id."""
        parsed = argparse.Namespace(mode="slot", slot_id="summary-balanced")
        result = _validate_smoke_args(parsed)
        assert result is None

    def test_validate_slot_mode_without_id_returns_one(self) -> None:
        """_validate_smoke_args should return 1 for 'slot' mode without slot_id."""
        parsed = argparse.Namespace(mode="slot", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result == 1

    def test_validate_invalid_mode_returns_one(self) -> None:
        """_validate_smoke_args should return 1 for invalid mode."""
        parsed = argparse.Namespace(mode="invalid", slot_id=None)
        result = _validate_smoke_args(parsed)
        assert result == 1


class TestRunSmokeValidationPath:
    """Tests for run_smoke validation failure path."""

    def test_run_smoke_slot_without_id_returns_one(self) -> None:
        """run_smoke should return 1 for slot mode without slot_id."""
        from llama_cli.commands.smoke import run_smoke

        exit_code = run_smoke(["slot"])
        assert exit_code == 1

    def test_run_smoke_json_path_with_validation(self) -> None:
        """run_smoke --json with valid args should go through JSON path."""
        from llama_cli.commands.smoke import run_smoke

        mock_result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
        )

        with (
            patch(
                "llama_cli.commands.smoke._build_slot_configs",
                return_value=[("test", "/m.gguf", "127.0.0.1", 8080)],
            ),
            patch("llama_cli.commands.smoke._build_smoke_config") as mock_cfg,
            patch("llama_cli.commands.smoke._run_probes", return_value=[mock_result]),
            patch("llama_cli.commands.smoke._print_report_json") as mock_json_printer,
        ):
            mock_cfg.return_value.inter_slot_delay_s = 0
            mock_cfg.return_value.listen_timeout_s = 5
            mock_cfg.return_value.max_tokens = 16
            mock_cfg.return_value.prompt = "test"
            mock_cfg.return_value.model_id_override = None

            exit_code = run_smoke(["both", "--json"])

        mock_json_printer.assert_called_once()
        assert isinstance(exit_code, int)

    def test_run_smoke_human_path(self) -> None:
        """run_smoke without --json should go through human path."""
        from llama_cli.commands.smoke import run_smoke

        mock_result = SmokeProbeResult(
            slot_id="test",
            status=SmokeProbeStatus.PASS,
            phase_reached=SmokePhase.COMPLETE,
        )

        with (
            patch(
                "llama_cli.commands.smoke._build_slot_configs",
                return_value=[("test", "/m.gguf", "127.0.0.1", 8080)],
            ),
            patch("llama_cli.commands.smoke._build_smoke_config") as mock_cfg,
            patch("llama_cli.commands.smoke._run_probes", return_value=[mock_result]),
            patch("llama_cli.commands.smoke._print_report_human") as mock_human_printer,
        ):
            mock_cfg.return_value.inter_slot_delay_s = 0
            mock_cfg.return_value.listen_timeout_s = 5
            mock_cfg.return_value.max_tokens = 16
            mock_cfg.return_value.prompt = "test"
            mock_cfg.return_value.model_id_override = None

            exit_code = run_smoke(["both"])

        mock_human_printer.assert_called_once()
        assert isinstance(exit_code, int)
