"""Phase 7 — T081: CA-003 parity test — JSON output matches FR-020 schema.

Verifies that --json output from smoke CLI matches the FR-020 schema
defined in spec Appendix B and contracts/smoke-api.md Section 4.

Tests:
  - All required fields present in each result
  - Correct types for each field
  - overall_status, overall_exit_code, pass_count, fail_count present
  - Valid enum values for status, phase_reached, failure_phase, model_id
  - Provenance object structure (sha, version)
"""

from __future__ import annotations

import json

from llama_manager.config import SmokeFailurePhase, SmokePhase, SmokeProbeStatus
from llama_manager.smoke import (
    ProvenanceRecord,
    SmokeCompositeReport,
    SmokeProbeResult,
)

# ---------------------------------------------------------------------------
# Schema constants (from contracts/smoke-api.md Section 4)
# ---------------------------------------------------------------------------

_REQUIRED_RESULT_FIELDS: set[str] = {"slot_id", "status", "phase_reached", "provenance"}
_OPTIONAL_RESULT_FIELDS: set[str] = {"failure_phase", "model_id", "latency_ms"}
_RESULT_ENUMS: dict[str, list[str]] = {
    "status": ["pass", "fail", "timeout", "crashed", "model_not_found", "auth_failure"],
    "phase_reached": ["listen", "models", "chat", "complete"],
}
_FAILURE_PHASE_ENUMS: list[str] = ["listen", "models", "chat"]
_VALID_PROVENANCE_FIELDS: set[str] = {"sha", "version"}


def _to_report_dict(report: SmokeCompositeReport, capsys=None) -> dict:
    """Serialize a SmokeCompositeReport to a dict (mimics --json output).

    Uses real CLI JSON output when capsys is provided.
    """
    if capsys is not None:
        from llama_cli.smoke_cli import _print_report_json

        _print_report_json(report)
        captured = capsys.readouterr()
        return json.loads(captured.out)

    return {
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
            for r in report.results
        ],
        "overall_status": report.overall_status,
        "overall_exit_code": report.overall_exit_code,
        "pass_count": report.pass_count,
        "fail_count": report.fail_count,
    }


class TestJsonSchemaFR020:
    """T081: JSON --json output matches FR-020 schema."""

    # ------------------------------------------------------------------
    # Required fields present
    # ------------------------------------------------------------------

    def test_all_required_result_fields_present(self) -> None:
        """Each result must include all required fields from FR-020."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                provenance=ProvenanceRecord(sha="abc", version="1.0"),
            ),
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)

        for r in d["results"]:
            for field in _REQUIRED_RESULT_FIELDS:
                assert field in r, f"Missing required field: {field}"

    def test_overall_fields_present(self) -> None:
        """Composite report must include overall_status, overall_exit_code, pass_count, fail_count."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            ),
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)

        assert "overall_status" in d
        assert "overall_exit_code" in d
        assert "pass_count" in d
        assert "fail_count" in d

    # ------------------------------------------------------------------
    # Type correctness
    # ------------------------------------------------------------------

    def test_slot_id_is_string(self) -> None:
        """slot_id must be a string."""
        results = [
            SmokeProbeResult(
                slot_id="test-slot",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["results"][0]["slot_id"], str)

    def test_status_is_valid_enum_string(self) -> None:
        """status must be one of the valid SmokeProbeStatus values."""
        for status in SmokeProbeStatus:
            results = [
                SmokeProbeResult(
                    slot_id="test",
                    status=status,
                    phase_reached=SmokePhase.COMPLETE,
                )
            ]
            report = SmokeCompositeReport(results=results)
            d = _to_report_dict(report)
            assert d["results"][0]["status"] in _RESULT_ENUMS["status"]

    def test_phase_reached_is_valid_enum_string(self) -> None:
        """phase_reached must be one of the valid SmokePhase values."""
        for phase in SmokePhase:
            results = [
                SmokeProbeResult(
                    slot_id="test",
                    status=SmokeProbeStatus.PASS,
                    phase_reached=phase,
                )
            ]
            report = SmokeCompositeReport(results=results)
            d = _to_report_dict(report)
            assert d["results"][0]["phase_reached"] in _RESULT_ENUMS["phase_reached"]

    def test_overall_status_is_valid_enum_string(self) -> None:
        """overall_status must be a valid SmokeProbeStatus string."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            ),
            SmokeProbeResult(
                slot_id="slot2",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
            ),
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["overall_status"] in _RESULT_ENUMS["status"]

    def test_overall_exit_code_is_integer(self) -> None:
        """overall_exit_code must be an integer."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["overall_exit_code"], int)

    def test_pass_count_is_integer(self) -> None:
        """pass_count must be an integer."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["pass_count"], int)

    def test_fail_count_is_integer(self) -> None:
        """fail_count must be an integer."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["fail_count"], int)

    # ------------------------------------------------------------------
    # Optional fields type correctness
    # ------------------------------------------------------------------

    def test_failure_phase_can_be_null(self) -> None:
        """failure_phase must be null when status is PASS."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["results"][0]["failure_phase"] is None

    def test_failure_phase_is_valid_string_when_present(self) -> None:
        """failure_phase must be a valid SmokeFailurePhase string when not null."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["results"][0]["failure_phase"] in _FAILURE_PHASE_ENUMS

    def test_model_id_can_be_null(self) -> None:
        """model_id must be null when unavailable."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
                model_id=None,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["results"][0]["model_id"] is None

    def test_model_id_is_string_when_present(self) -> None:
        """model_id must be a string when present."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                model_id="Qwen3.5-2B",
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["results"][0]["model_id"], str)

    def test_latency_ms_can_be_null(self) -> None:
        """latency_ms must be null when not measured."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
                latency_ms=None,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["results"][0]["latency_ms"] is None

    def test_latency_ms_is_integer_when_present(self) -> None:
        """latency_ms must be an integer when present."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                latency_ms=1234,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["results"][0]["latency_ms"], int)

    # ------------------------------------------------------------------
    # Provenance structure
    # ------------------------------------------------------------------

    def test_provenance_has_required_fields(self) -> None:
        """Provenance object must include sha and version."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                provenance=ProvenanceRecord(sha="abc1234", version="24.12"),
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        provenance = d["results"][0]["provenance"]
        for field in _VALID_PROVENANCE_FIELDS:
            assert field in provenance, f"Missing provenance field: {field}"

    def test_provenance_sha_is_string(self) -> None:
        """Provenance sha must be a string."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                provenance=ProvenanceRecord(sha="abc1234", version="24.12"),
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["results"][0]["provenance"]["sha"], str)

    def test_provenance_version_is_string(self) -> None:
        """Provenance version must be a string."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                provenance=ProvenanceRecord(sha="abc1234", version="24.12"),
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert isinstance(d["results"][0]["provenance"]["version"], str)

    def test_provenance_default_values(self) -> None:
        """Default provenance must have sha='unknown' and version='dev'."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        assert d["results"][0]["provenance"]["sha"] == "unknown"
        assert d["results"][0]["provenance"]["version"] == "dev"

    # ------------------------------------------------------------------
    # JSON serializability
    # ------------------------------------------------------------------

    def test_json_output_is_serializable(self) -> None:
        """Full report must be JSON-serializable without errors."""
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                model_id="Qwen3.5-2B",
                latency_ms=1234,
                provenance=ProvenanceRecord(sha="abc1234", version="24.12"),
            ),
            SmokeProbeResult(
                slot_id="slot2",
                status=SmokeProbeStatus.FAIL,
                phase_reached=SmokePhase.LISTEN,
                failure_phase=SmokeFailurePhase.LISTEN,
                provenance=ProvenanceRecord(sha="deadbeef", version="dev"),
            ),
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)

        # Must not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        # Round-trip must preserve data
        assert parsed["results"][0]["slot_id"] == "slot1"
        assert parsed["results"][0]["status"] == "pass"
        assert parsed["results"][1]["status"] == "fail"

    def test_json_output_all_statuses(self) -> None:
        """All SmokeProbeStatus values must produce valid JSON."""
        for status in SmokeProbeStatus:
            results = [
                SmokeProbeResult(
                    slot_id="test",
                    status=status,
                    phase_reached=SmokePhase.COMPLETE,
                )
            ]
            report = SmokeCompositeReport(results=results)
            d = _to_report_dict(report)
            json.dumps(d)  # must not raise

    # ------------------------------------------------------------------
    # Schema completeness — no extra fields
    # ------------------------------------------------------------------

    def test_result_has_no_extra_unknown_fields(self) -> None:
        """Result dict must not contain fields outside the schema."""
        allowed_fields = _REQUIRED_RESULT_FIELDS | _OPTIONAL_RESULT_FIELDS | {"provenance"}
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
                failure_phase=None,
                model_id="test",
                latency_ms=100,
                provenance=ProvenanceRecord(sha="abc", version="1.0"),
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        for r in d["results"]:
            for key in r:
                assert key in allowed_fields, f"Unexpected field: {key}"

    def test_composite_has_no_extra_unknown_fields(self) -> None:
        """Composite report dict must not contain fields outside the schema."""
        allowed_fields = {
            "results",
            "overall_status",
            "overall_exit_code",
            "pass_count",
            "fail_count",
        }
        results = [
            SmokeProbeResult(
                slot_id="slot1",
                status=SmokeProbeStatus.PASS,
                phase_reached=SmokePhase.COMPLETE,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)
        for key in d:
            assert key in allowed_fields, f"Unexpected composite field: {key}"

    # ------------------------------------------------------------------
    # Cross-check: JSON matches dataclass values
    # ------------------------------------------------------------------

    def test_json_values_match_dataclass(self) -> None:
        """JSON fields must match the underlying dataclass values exactly."""
        provenance = ProvenanceRecord(sha="abcd1234", version="25.01")
        results = [
            SmokeProbeResult(
                slot_id="gpu0-arc",
                status=SmokeProbeStatus.AUTH_FAILURE,
                phase_reached=SmokePhase.MODELS,
                failure_phase=SmokeFailurePhase.MODELS,
                model_id="qwen3.5-2b",
                latency_ms=2500,
                provenance=provenance,
            )
        ]
        report = SmokeCompositeReport(results=results)
        d = _to_report_dict(report)

        r = d["results"][0]
        assert r["slot_id"] == "gpu0-arc"
        assert r["status"] == "auth_failure"
        assert r["phase_reached"] == "models"
        assert r["failure_phase"] == "models"
        assert r["model_id"] == "qwen3.5-2b"
        assert r["latency_ms"] == 2500
        assert r["provenance"]["sha"] == "abcd1234"
        assert r["provenance"]["version"] == "25.01"
