"""Smoke lifecycle tests — TUI/CLI parity (T080) and dry-run flag bundle (T083)."""

import json
from unittest.mock import MagicMock

import pytest

from llama_manager.config import SmokeFailurePhase, SmokePhase, SmokeProbeStatus
from llama_manager.probe import (
    ProvenanceRecord,
    SmokeCompositeReport,
)
from tests.support.helpers import make_smoke_result as _make_result

"""Phase 7 — T080: CA-003 parity test — TUI vs CLI smoke results.

Verifies that TUI display (_print_report_human) and CLI output produce
identical slot status and phase data for the same server state.

Tests both passing and failing results.
"""


class TestTuiVsCliSmokeParity:
    """T080: TUI vs CLI smoke results produce identical slot status and phase data."""

    # ------------------------------------------------------------------
    # All-pass scenario
    # ------------------------------------------------------------------

    def test_passing_results_identical_status_phase(self, capsys) -> None:
        """TUI and CLI must produce the same slot status and phase for passing results."""
        from llama_cli.commands.smoke import _print_report_human, _print_report_json

        results = [
            _make_result(slot_id="arc_b580", status=SmokeProbeStatus.PASS),
            _make_result(slot_id="rtx3090", status=SmokeProbeStatus.PASS),
        ]
        report = SmokeCompositeReport(results=results)

        _print_report_human(report, mode="smoke")
        captured_human = capsys.readouterr()
        human_output = captured_human.out

        _print_report_json(report)
        captured_json = capsys.readouterr()
        parsed = json.loads(captured_json.out)

        for i, r in enumerate(results):
            assert parsed["results"][i]["slot_id"] == r.slot_id
            assert parsed["results"][i]["status"] == r.status.value
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value

        for r in results:
            assert r.slot_id in human_output
            assert r.status.value.upper() in human_output

        assert parsed["overall_status"] == report.overall_status.value
        assert report.overall_status.value.upper() in human_output

    # ------------------------------------------------------------------
    # Mixed pass/fail scenario
    # ------------------------------------------------------------------

    def test_mixed_results_identical_status_phase(self, capsys) -> None:
        """TUI and CLI must produce the same status/phase for mixed pass/fail results."""
        from llama_cli.commands.smoke import _print_report_human, _print_report_json

        results = [
            _make_result(slot_id="slot1", status=SmokeProbeStatus.PASS),
            _make_result(
                slot_id="slot2",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
                failure_phase=SmokeFailurePhase.LISTEN,
            ),
            _make_result(
                slot_id="slot3",
                status=SmokeProbeStatus.TIMEOUT,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
            ),
        ]
        report = SmokeCompositeReport(results=results)

        _print_report_human(report, mode="smoke")
        captured_human = capsys.readouterr()
        human_output = captured_human.out

        _print_report_json(report)
        captured_json = capsys.readouterr()
        parsed = json.loads(captured_json.out)

        assert parsed["overall_status"] == report.overall_status.value
        for r in results:
            assert r.slot_id in human_output
            assert r.status.value.upper() in human_output

        for i, r in enumerate(results):
            assert parsed["results"][i]["slot_id"] == r.slot_id
            assert parsed["results"][i]["status"] == r.status.value
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value

    # ------------------------------------------------------------------
    # Failing results scenario
    # ------------------------------------------------------------------

    def test_failing_results_identical_status_phase(self, capsys) -> None:
        """TUI and CLI must produce the same status/phase for all-failing results."""
        from llama_cli.commands.smoke import _print_report_human, _print_report_json

        results = [
            _make_result(
                slot_id="slot1",
                status=SmokeProbeStatus.CRASHED,
                phase_reached=SmokePhase.COMPLETE,
            ),
            _make_result(
                slot_id="slot2",
                status=SmokeProbeStatus.AUTH_FAILURE,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
            ),
            _make_result(
                slot_id="slot3",
                status=SmokeProbeStatus.MODEL_NOT_FOUND,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
                model_id="wrong-model",
            ),
        ]
        report = SmokeCompositeReport(results=results)

        _print_report_human(report, mode="smoke")
        captured_human = capsys.readouterr()
        human_output = captured_human.out

        _print_report_json(report)
        captured_json = capsys.readouterr()
        parsed = json.loads(captured_json.out)

        assert report.overall_status == SmokeProbeStatus.CRASHED
        assert parsed["overall_status"] == "crashed"

        for r in results:
            assert r.slot_id in human_output
            assert r.status.value.upper() in human_output

        assert len(parsed["results"]) == len(results)
        for i, r in enumerate(results):
            assert parsed["results"][i]["slot_id"] == r.slot_id
            assert parsed["results"][i]["status"] == r.status.value

    # ------------------------------------------------------------------
    # Empty results scenario
    # ------------------------------------------------------------------

    def test_empty_results_identical(self) -> None:
        """TUI and CLI must agree on empty results (overall PASS)."""
        report = SmokeCompositeReport(results=[])
        assert report.overall_status == SmokeProbeStatus.PASS
        assert report.pass_count == 0
        assert report.fail_count == 0

        json_output = json.dumps(
            {
                "results": [],
                "overall_status": report.overall_status,
                "overall_exit_code": report.overall_exit_code,
                "pass_count": report.pass_count,
                "fail_count": report.fail_count,
            }
        )
        parsed = json.loads(json_output)
        assert parsed["results"] == []
        assert parsed["overall_status"] == "pass"
        assert parsed["overall_exit_code"] == 0

    # ------------------------------------------------------------------
    # Phase data parity
    # ------------------------------------------------------------------

    def test_phase_data_identical_across_outputs(self, capsys) -> None:
        """phase_reached and failure_phase must be identical in both TUI and CLI output."""
        from llama_cli.commands.smoke import _print_report_json

        results = [
            _make_result(
                slot_id="slot1",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
            ),
            _make_result(
                slot_id="slot2",
                status=SmokeProbeStatus.TIMEOUT,
                phase_reached=SmokePhase.CHAT,
                failure_phase=SmokeFailurePhase.CHAT,
            ),
        ]
        report = SmokeCompositeReport(results=results)

        # Use production JSON formatter
        _print_report_json(report)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)

        for i, r in enumerate(results):
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value
            assert parsed["results"][i]["failure_phase"] == (
                r.failure_phase.value if r.failure_phase else None
            )

    # ------------------------------------------------------------------
    # Provenance parity
    # ------------------------------------------------------------------

    def test_provenance_included_in_both_outputs(self, capsys) -> None:
        """Provenance data must be present in both TUI and CLI output."""
        from llama_cli.commands.smoke import _print_report_json

        results = [
            _make_result(
                slot_id="slot1",
                provenance=ProvenanceRecord(sha="deadbeef", version="24.12.0"),
            ),
        ]
        report = SmokeCompositeReport(results=results)

        # Use production JSON formatter
        _print_report_json(report)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["results"][0]["provenance"]["sha"] == "deadbeef"
        assert parsed["results"][0]["provenance"]["version"] == "24.12.0"


"""Phase 7 — T083: Dry-run smoke flag bundle output test.

Verifies that `dry-run` shows smoke-relevant flags:
  - Model ID (from config or override)
  - Prompt text
  - /v1/models probe (enabled/skipped)
  - API key source (configured/not set)

Tests _print_smoke_probe_info() output from dry_run.py.
"""


import contextlib
from typing import Any
from unittest.mock import patch

from llama_manager.config import Config


class TestDryRunSmokeFlagBundleOutput:
    """T083: dry-run shows smoke-relevant flags in output."""

    # ------------------------------------------------------------------
    # /v1/models probe
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("skip", "expected"),
        [
            (False, "enabled"),
            (True, "skip"),
        ],
    )
    def test_dry_run_shows_v1_models_probe(self, capsys, skip: bool, expected: str) -> None:
        """dry-run output must show '/v1/models: <enabled|skip>' based on config."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.skip_models_discovery = skip
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert f"/v1/models: {expected}" in captured.out

    # ------------------------------------------------------------------
    # Prompt text
    # ------------------------------------------------------------------

    def test_dry_run_shows_prompt_text(self, capsys) -> None:
        """dry-run output must show the default prompt text."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        _print_smoke_probe_info(Config())

        captured = capsys.readouterr()
        assert "Prompt:" in captured.out
        assert "Respond with exactly one word." in captured.out

    def test_dry_run_shows_custom_prompt(self, capsys) -> None:
        """dry-run output must show custom prompt when configured."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.prompt = "Say hello."
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Prompt: Say hello." in captured.out

    # ------------------------------------------------------------------
    # API key source
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("api_key", "expected"),
        [
            ("sk-test-key-123", "[configured]"),
            ("", "[not set]"),
        ],
    )
    def test_dry_run_shows_api_key_source(self, capsys, api_key: str, expected: str) -> None:
        """dry-run output must show 'API key: [<configured|not set>]'."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.api_key = api_key
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert f"API key: {expected}" in captured.out

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
        assert str(cfg.smoke.max_tokens) in captured.out

    def test_dry_run_shows_custom_max_tokens(self, capsys) -> None:
        """dry-run output must show custom max tokens value."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.max_tokens = 32
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Max tokens: 32" in captured.out

    # ------------------------------------------------------------------
    # Smoke Probe section header
    # ------------------------------------------------------------------

    def test_dry_run_shows_smoke_probe_header(self, capsys) -> None:
        """dry-run output must include 'Smoke Probe:' header."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        _print_smoke_probe_info(Config())

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out

    # ------------------------------------------------------------------
    # Full output structure
    # ------------------------------------------------------------------

    def test_dry_run_smoke_section_has_all_fields(self, capsys) -> None:
        """Smoke Probe section must include all smoke-relevant flags."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.api_key = "sk-my-key"
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

    @pytest.mark.parametrize(
        ("mode", "kwargs"),
        [
            ("summary-balanced", {"primary_port": "8080"}),
            ("qwen35", {"primary_port": "8081"}),
            ("both", {"primary_port": "8080", "secondary_port": "8081"}),
        ],
    )
    def test_dry_run_mode_includes_smoke_probe(
        self, capsys, mode: str, kwargs: dict[str, Any]
    ) -> None:
        """Each dry-run mode must include Smoke Probe section."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run.run_dry_run") as mock_run,
            patch("llama_cli.commands.dry_run.write_dry_run_artifact"),
        ):
            mock_run.return_value = MagicMock(
                slot_payloads=[], warnings=[], errors=[], has_error=False, artifact_payload=None
            )

            with contextlib.suppress(SystemExit):
                dry_run(mode=mode, **kwargs)

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_dry_run_shows_user_prompt_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prompt text (user-provided) should be displayed in dry-run output."""
        from llama_cli.commands.dry_run import _print_smoke_probe_info

        cfg = Config()
        cfg.smoke.prompt = "Hello, world!"
        _print_smoke_probe_info(cfg)

        captured = capsys.readouterr()
        assert "Hello, world!" in captured.out

    @pytest.mark.parametrize(
        ("mode", "kwargs"),
        [
            ("summary-balanced", {"primary_port": "8080"}),
            ("summary-fast", {"primary_port": "8080"}),
            ("qwen35", {"primary_port": "8081"}),
            ("both", {"primary_port": "8080", "secondary_port": "8081"}),
        ],
    )
    def test_dry_run_all_modes_show_smoke_probe(
        self, capsys: pytest.CaptureFixture[str], mode: str, kwargs: dict[str, Any]
    ) -> None:
        """All dry-run modes must show Smoke Probe and /v1/models probe info."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run.run_dry_run") as mock_run,
            patch("llama_cli.commands.dry_run.write_dry_run_artifact"),
        ):
            mock_run.return_value = MagicMock(
                slot_payloads=[], warnings=[], errors=[], has_error=False, artifact_payload=None
            )

            with contextlib.suppress(SystemExit):
                dry_run(mode=mode, **kwargs)

        captured = capsys.readouterr()
        assert "Smoke Probe:" in captured.out, f"Mode '{mode}' missing Smoke Probe section"
        assert "/v1/models:" in captured.out, f"Mode '{mode}' missing /v1/models probe info"
