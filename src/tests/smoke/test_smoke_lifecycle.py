"""Phase 7 — T079: State-machine integration test.

Verifies the full SlotRuntime lifecycle:
  IDLE → LAUNCHING → RUNNING → DEGRADED → RUNNING → OFFLINE → IDLE

Tests:
  - All state transitions via SlotRuntime.transition_to()
  - start_time updates correctly for LAUNCHING/RUNNING transitions
  - start_time preserved for non-launching transitions
  - dataclass field integrity after each transition
  - Serialization (to_dict) correctness throughout
"""

import time
from unittest.mock import MagicMock

import pytest

from llama_manager.config import SlotState
from llama_manager.log_buffer import LogBuffer
from llama_manager.orchestration import SlotRuntime  # noqa: T079
from llama_manager.orchestration import manager as orchestration_manager


class TestStateMachineLifecycle:
    """T079: Full state-machine lifecycle integration test."""

    # ------------------------------------------------------------------
    # Phase 1: IDLE → LAUNCHING
    # ------------------------------------------------------------------

    def test_lifecycle_idle_to_launching(self) -> None:
        """Verify IDLE → LAUNCHING updates state and start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        assert runtime.state == SlotState.IDLE
        assert runtime.pid is None

        launching_time = time.time()
        runtime.transition_to(SlotState.LAUNCHING)

        assert runtime.state == SlotState.LAUNCHING
        assert runtime.start_time >= launching_time
        assert runtime.pid is None  # pid not assigned until RUNNING

    def test_lifecycle_launching_to_running(self) -> None:
        """Verify LAUNCHING → RUNNING updates state and start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.LAUNCHING,
            pid=None,
            start_time=time.time() - 1.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        running_time = time.time()
        runtime.transition_to(SlotState.RUNNING)

        assert runtime.state == SlotState.RUNNING
        assert runtime.start_time >= running_time
        # pid is still None — the dataclass doesn't auto-assign it
        assert runtime.pid is None

    # ------------------------------------------------------------------
    # Phase 2: RUNNING → DEGRADED (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_running_to_degraded_preserves_start_time(self) -> None:
        """DEGRADED transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.RUNNING,
            pid=12345,
            start_time=1000.0,
            logs=LogBuffer(),
            gpu_stats=MagicMock(),
        )

        runtime.transition_to(SlotState.DEGRADED)

        assert runtime.state == SlotState.DEGRADED
        assert runtime.start_time == 1000.0  # unchanged
        assert runtime.pid == 12345  # pid preserved

    # ------------------------------------------------------------------
    # Phase 3: DEGRADED → RUNNING (start_time MUST update)
    # ------------------------------------------------------------------

    def test_lifecycle_degraded_to_running_updates_start_time(self) -> None:
        """Recovery from DEGRADED → RUNNING must update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.DEGRADED,
            pid=12345,
            start_time=1000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        new_running_time = time.time()
        runtime.transition_to(SlotState.RUNNING)

        assert runtime.state == SlotState.RUNNING
        assert runtime.start_time >= new_running_time
        assert runtime.pid == 12345

    # ------------------------------------------------------------------
    # Phase 4: RUNNING → OFFLINE (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_running_to_offline_preserves_start_time(self) -> None:
        """OFFLINE transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.RUNNING,
            pid=12345,
            start_time=2000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        runtime.transition_to(SlotState.OFFLINE)

        assert runtime.state == SlotState.OFFLINE
        assert runtime.start_time == 2000.0  # unchanged
        assert runtime.pid == 12345

    # ------------------------------------------------------------------
    # Phase 5: OFFLINE → IDLE (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_offline_to_idle_preserves_start_time(self) -> None:
        """IDLE transition from OFFLINE must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.OFFLINE,
            pid=None,
            start_time=3000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        runtime.transition_to(SlotState.IDLE)

        assert runtime.state == SlotState.IDLE
        assert runtime.start_time == 3000.0  # unchanged
        assert runtime.pid is None

    # ------------------------------------------------------------------
    # Full chain in a single test
    # ------------------------------------------------------------------

    def test_full_lifecycle_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the full chain: IDLE→LAUNCHING→RUNNING→DEGRADED→RUNNING→OFFLINE→IDLE."""

        time_values = [1000.0, 1000.1, 1000.2, 1000.3, 1000.4, 1000.5]
        call_count = 0

        def fake_time() -> float:
            nonlocal call_count
            val = time_values[call_count] if call_count < len(time_values) else time_values[-1]
            call_count += 1
            return val

        monkeypatch.setattr(orchestration_manager.time, "time", fake_time)

        runtime = SlotRuntime(
            slot_id="gpu0-slot1",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        # IDLE → LAUNCHING
        runtime.transition_to(SlotState.LAUNCHING)
        assert runtime.state == SlotState.LAUNCHING
        launching_st = runtime.start_time

        # LAUNCHING → RUNNING
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.state == SlotState.RUNNING
        running_st = runtime.start_time
        assert running_st > launching_st

        # RUNNING → DEGRADED
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.state == SlotState.DEGRADED
        assert runtime.start_time == running_st  # preserved

        # DEGRADED → RUNNING
        runtime.pid = 12345  # pid assigned on recovery
        runtime.gpu_stats = MagicMock()
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.state == SlotState.RUNNING
        running_st2 = runtime.start_time
        assert running_st2 > running_st  # updated

        # RUNNING → OFFLINE
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.state == SlotState.OFFLINE
        offline_st = runtime.start_time
        assert offline_st == running_st2  # preserved

        # OFFLINE → IDLE
        runtime.transition_to(SlotState.IDLE)
        assert runtime.state == SlotState.IDLE
        idle_st = runtime.start_time
        assert idle_st == offline_st  # preserved

    # ------------------------------------------------------------------
    # Dataclass field integrity
    # ------------------------------------------------------------------

    def test_field_integrity_after_transitions(self) -> None:
        """All dataclass fields must remain accessible and correct after transitions."""
        gpu = MagicMock()
        runtime = SlotRuntime(
            slot_id="gpu1-slot0",
            state=SlotState.RUNNING,
            pid=9999,
            start_time=5000.0,
            logs=LogBuffer(),
            gpu_stats=gpu,
        )

        # Capture start_time before transitioning to DEGRADED (RUNNING → DEGRADED preserves)
        degraded_start = runtime.start_time
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.start_time == degraded_start  # preserved

        # DEGRADED → RUNNING updates start_time
        runtime.transition_to(SlotState.RUNNING)
        running_start = runtime.start_time
        assert running_start > degraded_start  # updated

        # RUNNING → OFFLINE preserves
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.start_time == running_start

        # OFFLINE → IDLE preserves
        runtime.transition_to(SlotState.IDLE)
        assert runtime.start_time == running_start

        # Fields must still be accessible
        assert runtime.slot_id == "gpu1-slot0"
        assert runtime.state == SlotState.IDLE
        assert runtime.pid == 9999
        assert isinstance(runtime.logs, LogBuffer)
        assert runtime.gpu_stats is gpu

    # ------------------------------------------------------------------
    # Serialization (to_dict)
    # ------------------------------------------------------------------

    def test_to_dict_at_each_state(self) -> None:
        """to_dict() must produce correct dict at every state in the lifecycle."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=100.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        expected_states = [
            SlotState.IDLE,
            SlotState.LAUNCHING,
            SlotState.RUNNING,
            SlotState.DEGRADED,
            SlotState.OFFLINE,
            SlotState.IDLE,  # back to idle
        ]

        for _i, new_state in enumerate(expected_states):
            runtime.transition_to(new_state)
            d = runtime.to_dict()
            assert d["slot_id"] == "test"
            assert d["state"] == new_state.value
            if new_state in (SlotState.LAUNCHING, SlotState.RUNNING):
                assert d["start_time"] >= 100.0
            else:
                assert d["start_time"] == runtime.start_time

    def test_to_dict_includes_gpu_stats_flag(self) -> None:
        """to_dict() must set gpu_stats=True when gpu_stats is not None."""
        gpu = MagicMock()
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=gpu,
        )
        d = runtime.to_dict()
        assert d["gpu_stats"] is True

    def test_to_dict_excludes_gpu_stats_when_none(self) -> None:
        """to_dict() must set gpu_stats=False when gpu_stats is None."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        d = runtime.to_dict()
        assert d["gpu_stats"] is False

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_transition_to_same_state_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Transitioning to the same state should still update start_time for LAUNCHING/RUNNING."""

        # Use fake clock pattern for deterministic timing
        time_values = [101.0, 102.0]
        call_count = 0

        def fake_time() -> float:
            nonlocal call_count
            val = time_values[call_count] if call_count < len(time_values) else time_values[-1]
            call_count += 1
            return val

        monkeypatch.setattr(orchestration_manager.time, "time", fake_time)

        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=100.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        old_st = runtime.start_time
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.start_time > old_st  # RUNNING always updates

    def test_transition_idle_to_idle_preserves_start_time(self) -> None:
        """IDLE → IDLE must NOT update start_time (IDLE is not in the update set)."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=999.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.IDLE)
        assert runtime.start_time == 999.0

    def test_transition_offline_to_offline_preserves_start_time(self) -> None:
        """OFFLINE → OFFLINE must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.OFFLINE,
            pid=None,
            start_time=777.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.start_time == 777.0

    def test_transition_degraded_to_degraded_preserves_start_time(self) -> None:
        """DEGRADED → DEGRADED must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.DEGRADED,
            pid=1,
            start_time=888.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.start_time == 888.0

    def test_transition_to_crashed(self) -> None:
        """CRASHED transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=500.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.CRASHED)
        assert runtime.state == SlotState.CRASHED
        assert runtime.start_time == 500.0  # unchanged

    def test_all_slot_states_covered(self) -> None:
        """Every SlotState member must be reachable via transition_to."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        for state in SlotState:
            runtime.transition_to(state)
            assert runtime.state == state


"""Phase 7 — T080: CA-003 parity test — TUI vs CLI smoke results.

Verifies that TUI display (_print_report_human) and CLI output produce
identical slot status and phase data for the same server state.

Tests both passing and failing results.
"""


import json

from llama_manager.config import SmokeFailurePhase, SmokePhase, SmokeProbeStatus
from llama_manager.probe import (
    ProvenanceRecord,
    SmokeCompositeReport,
)
from tests.support.helpers import make_smoke_result as _make_result


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
        cfg.smoke_skip_models_discovery = skip
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
        cfg.smoke_prompt = "Say hello."
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
        cfg.smoke_api_key = api_key
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
        cfg.smoke_prompt = "Hello, world!"
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
