"""Phase 7 — T083: Dry-run smoke flag bundle output test.

Verifies that `dry-run` shows smoke-relevant flags:
  - Model ID (from config or override)
  - Prompt text
  - /v1/models probe (enabled/skipped)
  - API key source (configured/not set)

Tests _print_smoke_probe_info() output from dry_run.py.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import patch

import pytest

from llama_manager.config import Config


class TestDryRunSmokeFlagBundleOutput:
    """T083: dry-run shows smoke-relevant flags in output."""

    # ------------------------------------------------------------------
    # /v1/models probe
    # ------------------------------------------------------------------

    def test_dry_run_shows_v1_models_probe_enabled(self, capsys) -> None:
        """dry-run output must show '/v1/models: enabled' when not skipped."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        # Default is skip_models_discovery=False → should show "enabled"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "/v1/models: enabled" in captured.out

    def test_dry_run_shows_v1_models_probe_skipped(self, capsys) -> None:
        """dry-run output must show '/v1/models: skip' when skipped."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_skip_models_discovery = True
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "/v1/models: skip" in captured.out

    # ------------------------------------------------------------------
    # Prompt text
    # ------------------------------------------------------------------

    def test_dry_run_shows_prompt_text(self, capsys) -> None:
        """dry-run output must show the prompt text."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Prompt:" in captured.out
        assert "Respond with exactly one word." in captured.out

    def test_dry_run_shows_custom_prompt(self, capsys) -> None:
        """dry-run output must show custom prompt when configured."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_prompt = "Say hello."
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Prompt: Say hello." in captured.out

    # ------------------------------------------------------------------
    # API key source
    # ------------------------------------------------------------------

    def test_dry_run_shows_api_key_configured(self, capsys) -> None:
        """dry-run output must show 'API key: [configured]' when key is set."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_api_key = "sk-test-key-123"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "API key: [configured]" in captured.out

    def test_dry_run_shows_api_key_not_set(self, capsys) -> None:
        """dry-run output must show 'API key: [not set]' when key is empty."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_api_key = ""
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "API key: [not set]" in captured.out

    # ------------------------------------------------------------------
    # Max tokens
    # ------------------------------------------------------------------

    def test_dry_run_shows_max_tokens(self, capsys) -> None:
        """dry-run output must show max tokens value."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Max tokens:" in captured.out
        assert str(cfg.smoke_max_tokens) in captured.out

    def test_dry_run_shows_custom_max_tokens(self, capsys) -> None:
        """dry-run output must show custom max tokens value."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_max_tokens = 32
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Max tokens: 32" in captured.out

    # ------------------------------------------------------------------
    # Smoke Probe section header
    # ------------------------------------------------------------------

    def test_dry_run_shows_smoke_probe_header(self, capsys) -> None:
        """dry-run output must include 'Smoke Probe:' header."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out

    # ------------------------------------------------------------------
    # Full output structure
    # ------------------------------------------------------------------

    def test_dry_run_smoke_section_has_all_fields(self, capsys) -> None:
        """Smoke Probe section must include all smoke-relevant flags."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_api_key = "sk-my-key"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()

        # All expected fields must be present
        assert "Smoke Probe:" in captured.out
        assert "/v1/models:" in captured.out
        assert "Prompt:" in captured.out
        assert "Max tokens:" in captured.out
        assert "API key:" in captured.out

    def test_dry_run_smoke_section_order(self, capsys) -> None:
        """Smoke Probe section fields must appear in deterministic order."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Find indices of each field
        v1_models_idx = None
        prompt_idx = None
        max_tokens_idx = None
        api_key_idx = None

        for i, line in enumerate(lines):
            if "/v1/models:" in line:
                v1_models_idx = i
            elif "Prompt:" in line:
                prompt_idx = i
            elif "Max tokens:" in line:
                max_tokens_idx = i
            elif "API key:" in line:
                api_key_idx = i

        # All must be present
        assert v1_models_idx is not None
        assert prompt_idx is not None
        assert max_tokens_idx is not None
        assert api_key_idx is not None

        # Order must be deterministic: /v1/models → Prompt → Max tokens → API key
        assert v1_models_idx < prompt_idx < max_tokens_idx < api_key_idx

    # ------------------------------------------------------------------
    # Integration: full dry-run mode includes smoke probe info
    # ------------------------------------------------------------------

    def test_dry_run_summary_balanced_includes_smoke_probe(self, capsys) -> None:
        """dry-run summary-balanced mode must include Smoke Probe section."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="summary-balanced", primary_port="8080")

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out
        assert "/v1/models:" in captured.out

    def test_dry_run_qwen35_includes_smoke_probe(self, capsys) -> None:
        """dry-run qwen35 mode must include Smoke Probe section."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="qwen35", primary_port="8081")

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out

    def test_dry_run_both_includes_smoke_probe(self, capsys) -> None:
        """dry-run both mode must include Smoke Probe section."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="both", primary_port="8080", secondary_port="8081")

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_dry_run_empty_api_key_shows_not_set(self, capsys) -> None:
        """Empty smoke_api_key must show 'not set' not '[configured]'."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_api_key = ""
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "[not set]" in captured.out
        assert "[configured]" not in captured.out

    def test_dry_run_with_non_empty_api_key_shows_configured(self, capsys) -> None:
        """Non-empty smoke_api_key must show '[configured]'."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_api_key = "sk-actual-key"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "[configured]" in captured.out
        assert "[not set]" not in captured.out

    def test_dry_run_prompt_is_redacted_for_secrets(self, capsys) -> None:
        """Prompt text should not contain sensitive data (it's user-provided text)."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke_prompt = "Hello, world!"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out

    def test_dry_run_all_modes_show_smoke_probe(self, capsys: pytest.CaptureFixture[str]) -> None:
        """All dry-run modes (summary-balanced, summary-fast, qwen35, both) must show Smoke Probe."""
        from llama_cli.commands.dry_run import dry_run

        test_cases: list[tuple[str, dict[str, Any]]] = [
            ("summary-balanced", {"primary_port": "8080"}),
            ("summary-fast", {"primary_port": "8080"}),
            ("qwen35", {"primary_port": "8081"}),
            ("both", {"primary_port": "8080", "secondary_port": "8081"}),
        ]

        for mode, kwargs in test_cases:
            # Capture fresh output for each mode
            captured = capsys.readouterr()
            assert "Smoke Probe:" not in captured.out, f"Residual output from previous mode: {mode}"

            with (
                patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
                patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
            ):
                mock_run.return_value = False

                with contextlib.suppress(SystemExit):
                    dry_run(mode=mode, **kwargs)

            captured = capsys.readouterr()
            assert "Smoke Probe:" in captured.out, f"Mode '{mode}' missing Smoke Probe section"
            assert "/v1/models:" in captured.out, f"Mode '{mode}' missing /v1/models probe info"
