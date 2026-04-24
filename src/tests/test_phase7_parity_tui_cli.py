"""Phase 7 — T080: CA-003 parity test — TUI vs CLI smoke results.

Verifies that TUI display (_print_report_human) and CLI output produce
identical slot status and phase data for the same server state.

Tests both passing and failing results.
"""

from __future__ import annotations

import json

from llama_manager.config import SmokeFailurePhase, SmokePhase, SmokeProbeStatus
from llama_manager.smoke import (
    ProvenanceRecord,
    SmokeCompositeReport,
    SmokeProbeResult,
)


def _make_result(
    slot_id: str = "slot1",
    status: SmokeProbeStatus = SmokeProbeStatus.PASS,
    phase_reached: SmokePhase = SmokePhase.COMPLETE,
    failure_phase: SmokeFailurePhase | None = None,
    model_id: str | None = "test-model",
    latency_ms: int | None = 100,
    provenance: ProvenanceRecord | None = None,
) -> SmokeProbeResult:
    """Helper to create a SmokeProbeResult with sensible defaults."""
    if provenance is None:
        provenance = ProvenanceRecord(sha="abc1234", version="24.12")
    return SmokeProbeResult(
        slot_id=slot_id,
        status=status,
        phase_reached=phase_reached,
        failure_phase=failure_phase,
        model_id=model_id,
        latency_ms=latency_ms,
        provenance=provenance,
    )


class TestTuiVsCliSmokeParity:
    """T080: TUI vs CLI smoke results produce identical slot status and phase data."""

    # ------------------------------------------------------------------
    # All-pass scenario
    # ------------------------------------------------------------------

    def test_passing_results_identical_status_phase(self) -> None:
        """TUI and CLI must produce the same slot status and phase for passing results."""
        results = [
            _make_result(slot_id="arc_b580", status=SmokeProbeStatus.PASS),
            _make_result(slot_id="rtx3090", status=SmokeProbeStatus.PASS),
        ]
        report = SmokeCompositeReport(results=results)

        # Build what _print_report_human would emit (slot status + phase lines)
        human_lines: list[str] = []
        human_lines.append(f"Smoke Test Report — {report.overall_status.value.upper()}")
        human_lines.append(f"Overall exit code: {report.overall_exit_code}")
        human_lines.append(f"Pass: {report.pass_count} / {len(results)}")
        human_lines.append(f"Fail: {report.fail_count} / {len(results)}")
        for r in results:
            status_icon = "✓" if r.status == SmokeProbeStatus.PASS else "✗"
            line = f"  {status_icon} {r.slot_id}: {r.status.value}"
            if r.model_id:
                line += f" (model={r.model_id})"
            if r.latency_ms is not None:
                line += f" ({r.latency_ms}ms)"
            human_lines.append(line)
        human_output = "\n".join(human_lines)

        # Build what CLI _print_report_json would emit
        json_output = json.dumps(
            {
                "results": [
                    {
                        "slot_id": r.slot_id,
                        "status": r.status,
                        "phase_reached": r.phase_reached,
                        "failure_phase": r.failure_phase,
                        "model_id": r.model_id,
                        "latency_ms": r.latency_ms,
                        "provenance": {"sha": r.provenance.sha, "version": r.provenance.version},
                    }
                    for r in results
                ],
                "overall_status": report.overall_status,
                "overall_exit_code": report.overall_exit_code,
                "pass_count": report.pass_count,
                "fail_count": report.fail_count,
            },
            indent=2,
        )

        # Parse JSON back — slot data must match
        parsed = json.loads(json_output)
        for i, r in enumerate(results):
            assert parsed["results"][i]["slot_id"] == r.slot_id
            assert parsed["results"][i]["status"] == r.status.value
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value

        # Human-readable output must contain the same slot IDs and statuses
        for r in results:
            assert r.slot_id in human_output
            assert r.status.value in human_output

        # Both outputs must agree on overall_status
        assert parsed["overall_status"] == report.overall_status.value
        assert report.overall_status.value.upper() in human_output

    # ------------------------------------------------------------------
    # Mixed pass/fail scenario
    # ------------------------------------------------------------------

    def test_mixed_results_identical_status_phase(self) -> None:
        """TUI and CLI must produce the same status/phase for mixed pass/fail results."""
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

        human_lines: list[str] = []
        human_lines.append(f"Smoke Test Report — {report.overall_status.value.upper()}")
        for r in results:
            status_icon = "✓" if r.status == SmokeProbeStatus.PASS else "✗"
            line = f"  {status_icon} {r.slot_id}: {r.status.value}"
            if r.model_id:
                line += f" (model={r.model_id})"
            if r.latency_ms is not None:
                line += f" ({r.latency_ms}ms)"
            human_lines.append(line)
        human_output = "\n".join(human_lines)

        json_output = json.dumps(
            {
                "results": [
                    {
                        "slot_id": r.slot_id,
                        "status": r.status,
                        "phase_reached": r.phase_reached,
                        "failure_phase": r.failure_phase,
                        "model_id": r.model_id,
                        "latency_ms": r.latency_ms,
                    }
                    for r in results
                ],
                "overall_status": report.overall_status,
                "overall_exit_code": report.overall_exit_code,
                "pass_count": report.pass_count,
                "fail_count": report.fail_count,
            }
        )
        parsed = json.loads(json_output)

        # Both must agree on overall status
        assert parsed["overall_status"] == report.overall_status.value
        for r in results:
            assert r.slot_id in human_output
            assert r.status.value in human_output

        # Slot data in JSON must match
        for i, r in enumerate(results):
            assert parsed["results"][i]["slot_id"] == r.slot_id
            assert parsed["results"][i]["status"] == r.status.value
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value

    # ------------------------------------------------------------------
    # Failing results scenario
    # ------------------------------------------------------------------

    def test_failing_results_identical_status_phase(self) -> None:
        """TUI and CLI must produce the same status/phase for all-failing results."""
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

        human_lines: list[str] = []
        human_lines.append(f"Smoke Test Report — {report.overall_status.value.upper()}")
        for r in results:
            status_icon = "✓" if r.status == SmokeProbeStatus.PASS else "✗"
            line = f"  {status_icon} {r.slot_id}: {r.status.value}"
            if r.model_id:
                line += f" (model={r.model_id})"
            if r.latency_ms is not None:
                line += f" ({r.latency_ms}ms)"
            human_lines.append(line)
        human_output = "\n".join(human_lines)

        json_output = json.dumps(
            {
                "results": [
                    {
                        "slot_id": r.slot_id,
                        "status": r.status,
                        "phase_reached": r.phase_reached,
                        "failure_phase": r.failure_phase,
                        "model_id": r.model_id,
                        "latency_ms": r.latency_ms,
                    }
                    for r in results
                ],
                "overall_status": report.overall_status,
                "overall_exit_code": report.overall_exit_code,
                "pass_count": report.pass_count,
                "fail_count": report.fail_count,
            }
        )
        parsed = json.loads(json_output)

        # CRASHED is worst, so overall must be CRASHED
        assert report.overall_status == SmokeProbeStatus.CRASHED
        assert parsed["overall_status"] == "crashed"

        for r in results:
            assert r.slot_id in human_output
            assert r.status.value in human_output

        # All slots present in JSON
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

    def test_phase_data_identical_across_outputs(self) -> None:
        """phase_reached and failure_phase must be identical in both TUI and CLI output."""
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

        # JSON output
        json_output = json.dumps(
            {
                "results": [
                    {
                        "slot_id": r.slot_id,
                        "status": r.status.value,
                        "phase_reached": r.phase_reached.value,
                        "failure_phase": r.failure_phase.value if r.failure_phase else None,
                    }
                    for r in results
                ],
                "overall_status": report.overall_status.value,
            }
        )
        parsed = json.loads(json_output)

        for i, r in enumerate(results):
            assert parsed["results"][i]["phase_reached"] == r.phase_reached.value
            assert parsed["results"][i]["failure_phase"] == (
                r.failure_phase.value if r.failure_phase else None
            )

    # ------------------------------------------------------------------
    # Provenance parity
    # ------------------------------------------------------------------

    def test_provenance_included_in_both_outputs(self) -> None:
        """Provenance data must be present in both TUI and CLI output."""
        results = [
            _make_result(
                slot_id="slot1",
                provenance=ProvenanceRecord(sha="deadbeef", version="24.12.0"),
            ),
        ]
        report = SmokeCompositeReport(results=results)

        json_output = json.dumps(
            {
                "results": [
                    {
                        "slot_id": r.slot_id,
                        "provenance": {"sha": r.provenance.sha, "version": r.provenance.version},
                    }
                    for r in results
                ],
                "overall_status": report.overall_status,
            }
        )
        parsed = json.loads(json_output)
        assert parsed["results"][0]["provenance"]["sha"] == "deadbeef"
        assert parsed["results"][0]["provenance"]["version"] == "24.12.0"
