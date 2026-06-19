"""Tests for slot runtime stats: parser, persistence, and HTTP collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from llama_manager.slot_stats import (
    SlotStatsSnapshot,
    collect_slot_stats,
    load_slot_stats,
    parse_metrics_payload,
    parse_slots_payload,
    save_slot_stats,
    slot_stats_file_path,
)

# =============================================================================
# Task 1: Parser tests
# =============================================================================


def test_parse_slots_payload_sums_decoded_tokens() -> None:
    """parse_slots_payload should sum n_decoded across slots."""
    payload = [
        {"next_token": {"n_decoded": 7}, "n_prompt_tokens_processed": 11},
        {"next_token": {"n_decoded": 3}, "n_prompt_tokens_processed": 5},
    ]

    result = parse_slots_payload("summary", 8080, payload, now=10.0)

    assert result == SlotStatsSnapshot(
        alias="summary",
        port=8080,
        updated_at=10.0,
        tokens_in=16,
        tokens_out=10,
    )


def test_parse_slots_payload_reads_rate_fields_defensively() -> None:
    """parse_slots_payload should read tps and prompt_tps from nested fields."""
    payload = [
        {
            "next_token": {"n_decoded": 12, "tps": 4.25},
            "prompt": {"n_tokens_processed": 20, "tokens_per_second": 123.4},
        }
    ]

    result = parse_slots_payload("code", 8081, payload, now=20.0)

    assert result.tps == 4.25
    assert result.prompt_tps == 123.4
    assert result.tokens_in == 20
    assert result.tokens_out == 12


def test_parse_slots_payload_ignores_invalid_shape() -> None:
    """parse_slots_payload should return zeros for non-list payload."""
    result = parse_slots_payload("code", 8081, {"bad": "shape"}, now=20.0)

    assert result.tokens_in == 0
    assert result.tokens_out == 0
    assert result.tps is None
    assert result.prompt_tps is None


def test_parse_slots_payload_empty_list() -> None:
    """parse_slots_payload should return zeros for empty list."""
    result = parse_slots_payload("empty", 9999, [], now=5.0)

    assert result.tokens_in == 0
    assert result.tokens_out == 0
    assert result.tps is None
    assert result.updated_at == 5.0


def test_parse_slots_payload_fallback_token_fields() -> None:
    """parse_slots_payload should try multiple token field names."""
    payload = [
        {"n_prompt_tokens": 42, "next_token": {"n_decoded": 8}},
    ]
    result = parse_slots_payload("fallback", 8080, payload, now=1.0)
    assert result.tokens_in == 42
    assert result.tokens_out == 8


def test_parse_slots_payload_tps_candidates() -> None:
    """parse_slots_payload should try multiple TPS field names."""
    payload = [
        {"tokens_per_second": 15.5},
    ]
    result = parse_slots_payload("tps_test", 8080, payload, now=1.0)
    assert result.tps == 15.5


def test_parse_slots_payload_prompt_tps_nested() -> None:
    """parse_slots_payload should read prompt_tps from nested prompt.tokens_per_second."""
    payload = [
        {
            "next_token": {"n_decoded": 5},
            "prompt": {"tokens_per_second": 50.0},
        }
    ]
    result = parse_slots_payload("prompt_tps", 8080, payload, now=1.0)
    assert result.prompt_tps == 50.0
    assert result.tokens_out == 5


def test_parse_slots_payload_multiple_slots_sums() -> None:
    """parse_slots_payload should sum across multiple slots."""
    payload = [
        {"next_token": {"n_decoded": 10}, "n_prompt_tokens_processed": 5},
        {"next_token": {"n_decoded": 20}, "n_prompt_tokens_processed": 15},
        {"next_token": {"n_decoded": 30}, "n_prompt_tokens_processed": 25},
    ]
    result = parse_slots_payload("multi", 8080, payload, now=1.0)
    assert result.tokens_out == 60
    assert result.tokens_in == 45


def test_parse_slots_payload_ignores_non_dict_in_list() -> None:
    """parse_slots_payload should skip non-dict entries in the list."""
    payload = [
        {"next_token": {"n_decoded": 5}},
        "invalid",
        42,
        None,
        {"next_token": {"n_decoded": 3}},
    ]
    result = parse_slots_payload("skip", 8080, payload, now=1.0)
    assert result.tokens_out == 8


def test_parse_slots_payload_real_idle_shape_has_no_cumulative_output() -> None:
    """Idle llama.cpp /slots payload should not invent output tokens or rates."""
    payload = [
        {
            "id": 0,
            "is_processing": False,
            "n_prompt_tokens": 1565,
            "n_prompt_tokens_processed": 61,
            "next_token": [],
        },
        {
            "id": 1,
            "is_processing": False,
            "n_prompt_tokens": 1024,
            "n_prompt_tokens_processed": 0,
            "next_token": [],
        },
    ]

    result = parse_slots_payload("idle", 8080, payload, now=1.0)

    assert result.tps is None
    assert result.prompt_tps is None
    assert result.tokens_in == 1085
    assert result.tokens_out == 0


def test_parse_metrics_payload_reads_llama_cpp_counters_and_rates() -> None:
    """Prometheus /metrics payload has the cumulative counters needed by the TUI."""
    payload = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12345
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 456.7
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 89
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 12.34
"""

    result = parse_metrics_payload("summary", 8080, payload, now=10.0)

    assert result == SlotStatsSnapshot(
        alias="summary",
        port=8080,
        updated_at=10.0,
        tps=12.34,
        prompt_tps=456.7,
        tokens_in=12345,
        tokens_out=89,
        source="metrics",
    )


def test_parse_metrics_payload_ignores_labels_and_comments() -> None:
    """Metrics parser should handle labels and comments without storing raw payloads."""
    payload = """
# comment
llamacpp:prompt_tokens_total{model="qwen"} 100
llamacpp:prompt_tokens_total{model="other"} 23
llamacpp:tokens_predicted_total{model="qwen"} 7
llamacpp:predicted_tokens_seconds{model="qwen"} 3.5
llamacpp:prompt_tokens_seconds{model="qwen"} 10
"""

    result = parse_metrics_payload("code", 8081, payload, now=11.0)

    assert result is not None
    assert result.tokens_in == 123
    assert result.tokens_out == 7
    assert result.tps == 3.5
    assert result.prompt_tps == 10.0


def test_parse_metrics_payload_returns_none_without_expected_metrics() -> None:
    """Unexpected Prometheus payload should fall back to /slots collection."""
    assert parse_metrics_payload("code", 8081, "process_cpu_seconds_total 3", now=11.0) is None


def test_slot_stats_snapshot_to_display() -> None:
    """SlotStatsSnapshot.to_display should format values correctly."""
    stats = SlotStatsSnapshot(
        alias="test",
        port=8080,
        updated_at=1.0,
        tps=4.25,
        prompt_tps=123.4,
        tokens_in=100,
        tokens_out=50,
    )
    display = stats.to_display()
    assert display == {
        "tps": "4.2",
        "pp": "123.4",
        "tokens_in": "100",
        "tokens_out": "50",
    }


def test_slot_stats_snapshot_to_display_none_rates() -> None:
    """SlotStatsSnapshot.to_display should show '--' for None rates."""
    stats = SlotStatsSnapshot(
        alias="test",
        port=8080,
        updated_at=1.0,
    )
    display = stats.to_display()
    assert display == {
        "tps": "--",
        "pp": "--",
        "tokens_in": "0",
        "tokens_out": "0",
    }


def test_slot_stats_snapshot_negative_tokens_clamped() -> None:
    """SlotStatsSnapshot.to_display should clamp negative tokens to 0."""
    stats = SlotStatsSnapshot(
        alias="test",
        port=8080,
        updated_at=1.0,
        tokens_in=-5,
        tokens_out=-10,
    )
    display = stats.to_display()
    assert display["tokens_in"] == "0"
    assert display["tokens_out"] == "0"


# =============================================================================
# Task 2: Persistence tests
# =============================================================================


def test_slot_stats_file_path_uses_runtime_dir(tmp_path: Path) -> None:
    """slot_stats_file_path should return runtime_dir / 'slot-stats.json'."""
    assert slot_stats_file_path(tmp_path) == tmp_path / "slot-stats.json"


def test_save_and_load_slot_stats_round_trip(tmp_path: Path) -> None:
    """save_slot_stats / load_slot_stats should round-trip correctly."""
    stats = {
        "summary": SlotStatsSnapshot(
            alias="summary",
            port=8080,
            updated_at=10.0,
            tps=4.2,
            prompt_tps=111.0,
            tokens_in=20,
            tokens_out=9,
        )
    }

    save_slot_stats(stats, runtime_dir=tmp_path)

    assert load_slot_stats(runtime_dir=tmp_path) == stats


def test_load_slot_stats_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """load_slot_stats should return {} for missing file."""
    assert load_slot_stats(runtime_dir=tmp_path) == {}


def test_load_slot_stats_returns_empty_for_invalid_json(tmp_path: Path) -> None:
    """load_slot_stats should return {} for invalid JSON."""
    slot_stats_file_path(tmp_path).write_text("{not-json", encoding="utf-8")
    assert load_slot_stats(runtime_dir=tmp_path) == {}


def test_save_slot_stats_creates_parent_directory(tmp_path: Path) -> None:
    """save_slot_stats should create parent directory if needed."""
    sub = tmp_path / "sub" / "dir"
    stats = {
        "test": SlotStatsSnapshot(
            alias="test",
            port=8080,
            updated_at=1.0,
            tokens_out=5,
        )
    }
    # Patch slot_stats_file_path to use our subdirectory
    with patch(
        "llama_manager.slot_stats.slot_stats_file_path", return_value=sub / "slot-stats.json"
    ):
        save_slot_stats(stats, runtime_dir=tmp_path)

    assert (sub / "slot-stats.json").exists()


def test_save_load_slot_stats_multiple_aliases(tmp_path: Path) -> None:
    """save_slot_stats / load_slot_stats should handle multiple aliases."""
    stats = {
        "summary": SlotStatsSnapshot(
            alias="summary",
            port=8080,
            updated_at=10.0,
            tokens_out=10,
        ),
        "code": SlotStatsSnapshot(
            alias="code",
            port=8081,
            updated_at=20.0,
            tokens_out=20,
        ),
    }
    save_slot_stats(stats, runtime_dir=tmp_path)
    assert load_slot_stats(runtime_dir=tmp_path) == stats


def test_load_slot_stats_skips_invalid_entry(tmp_path: Path) -> None:
    """load_slot_stats should skip invalid slot entries."""
    # Write JSON with one valid and one invalid entry
    path = slot_stats_file_path(tmp_path)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "slots": {
                    "valid": {
                        "alias": "valid",
                        "port": 8080,
                        "updated_at": 1.0,
                        "tokens_in": 5,
                        "tokens_out": 3,
                    },
                    "invalid": "not-a-dict",
                },
            }
        ),
        encoding="utf-8",
    )
    result = load_slot_stats(runtime_dir=tmp_path)
    assert "valid" in result
    assert "invalid" not in result


def test_save_slot_stats_json_shape(tmp_path: Path) -> None:
    """save_slot_stats should write the expected JSON shape."""
    stats = {
        "test": SlotStatsSnapshot(
            alias="test",
            port=8080,
            updated_at=10.0,
            tps=4.2,
            prompt_tps=111.0,
            tokens_in=20,
            tokens_out=9,
        )
    }
    save_slot_stats(stats, runtime_dir=tmp_path)
    path = slot_stats_file_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "test" in data["slots"]
    assert data["slots"]["test"]["tps"] == 4.2
    assert data["slots"]["test"]["tokens_in"] == 20


# =============================================================================
# Task 3: HTTP collector tests
# =============================================================================


class _Response:
    """Mock urllib response."""

    def __init__(self, payload: Any, *, json_payload: bool = True) -> None:
        if json_payload:
            self._payload = json.dumps(payload).encode("utf-8")
        else:
            self._payload = str(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_collect_slot_stats_prefers_metrics_endpoint(monkeypatch: Any) -> None:
    """collect_slot_stats should fetch /metrics first for cumulative counters."""
    calls: list[str] = []

    def fake_urlopen(request, timeout: float):
        calls.append(request.full_url)
        return _Response(
            """
llamacpp:prompt_tokens_total 20
llamacpp:tokens_predicted_total 3
llamacpp:prompt_tokens_seconds 40
llamacpp:predicted_tokens_seconds 6
""",
            json_payload=False,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0)

    assert calls == ["http://127.0.0.1:8080/metrics"]
    assert result is not None
    assert result.alias == "summary"
    assert result.source == "metrics"
    assert result.tokens_in == 20
    assert result.tokens_out == 3


def test_collect_slot_stats_falls_back_to_slots_when_metrics_unavailable(monkeypatch: Any) -> None:
    """collect_slot_stats should retain /slots fallback for old or manually launched servers."""
    calls: list[str] = []

    def fake_urlopen(request, timeout: float):
        calls.append(request.full_url)
        if request.full_url.endswith("/metrics"):
            raise OSError("metrics disabled")
        return _Response([{"next_token": {"n_decoded": 3}, "n_prompt_tokens_processed": 9}])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0)

    assert calls == ["http://127.0.0.1:8080/metrics", "http://127.0.0.1:8080/slots"]
    assert result is not None
    assert result.source == "slots"
    assert result.tokens_in == 9
    assert result.tokens_out == 3


def test_collect_slot_stats_returns_none_on_http_failure(monkeypatch: Any) -> None:
    """collect_slot_stats should return None on OSError."""

    def fake_urlopen(request, timeout: float):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0) is None


def test_collect_slot_stats_returns_none_on_invalid_json(monkeypatch: Any) -> None:
    """collect_slot_stats should return None on invalid JSON response."""

    class _BadResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"not json"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _BadResponse())

    assert collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0) is None


def test_collect_slot_stats_returns_none_on_non_list_payload(monkeypatch: Any) -> None:
    """collect_slot_stats should return None for non-list JSON response."""

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response({"bad": "shape"}))

    assert collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0) is None


def test_collect_slot_stats_timeout_returns_none(monkeypatch: Any) -> None:
    """collect_slot_stats should return None on TimeoutError."""

    def fake_urlopen(request, timeout: float):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0) is None


def test_collect_slot_stats_parses_full_payload(monkeypatch: Any) -> None:
    """collect_slot_stats should parse a realistic /slots response."""
    payload = [
        {
            "id": 0,
            "next_token": {"n_decoded": 15, "tps": 5.5},
            "prompt": {"n_tokens_processed": 30, "tokens_per_second": 100.0},
        },
        {
            "id": 1,
            "next_token": {"n_decoded": 8},
            "n_prompt_tokens_processed": 12,
        },
    ]

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))

    result = collect_slot_stats("code", "127.0.0.1", 8081, now=50.0)

    assert result is not None
    assert result.tokens_out == 23
    assert result.tokens_in == 42
    assert result.tps == 5.5
    assert result.prompt_tps == 100.0
