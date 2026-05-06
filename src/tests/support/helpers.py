from __future__ import annotations

"""Reusable assertion helpers for tests."""


from typing import Any


def assert_dicts_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    message: str = "",
) -> None:
    """Assert two dictionaries are equal, with useful key diagnostics."""
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        error_parts = []
        if missing:
            error_parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra keys: {sorted(extra)}")
        raise AssertionError(
            f"Dict key mismatch. {', '.join(error_parts)}" + (f": {message}" if message else "")
        )

    for key in sorted(actual_keys):
        if actual[key] != expected[key]:
            error_msg = (
                f"Value mismatch for key '{key}': expected {expected[key]!r}, got {actual[key]!r}"
            )
            if message:
                error_msg += f" ({message})"
            raise AssertionError(error_msg)


def assert_sorted_identically(
    actual: list[Any],
    expected: list[Any],
    key_name: str | None = None,
    message: str = "",
) -> None:
    """Assert two sorted lists are identical."""
    if len(actual) != len(expected):
        raise AssertionError(
            f"List length mismatch: expected {len(expected)}, got {len(actual)}"
            + (f": {message}" if message else "")
        )

    for i, (act_item, exp_item) in enumerate(zip(actual, expected, strict=True)):
        if act_item != exp_item:
            error_msg = f"Item at index {i} mismatch: expected {exp_item!r}, got {act_item!r}"
            if key_name:
                error_msg += f" (key={key_name})"
            if message:
                error_msg += f" ({message})"
            raise AssertionError(error_msg)


def normalize_output_for_diff(output: str) -> str:
    """Normalize output strings for consistent diff comparison."""
    stripped_lines = [line.rstrip() for line in output.splitlines()]
    while stripped_lines and not stripped_lines[-1]:
        stripped_lines.pop()
    return "\n".join(stripped_lines)


def assert_json_has_keys(data: dict[str, Any], required_keys: set[str]) -> None:
    """Assert JSON-like dict contains all required keys."""
    missing = required_keys - set(data)
    if missing:
        raise AssertionError(f"Missing required JSON keys: {sorted(missing)}")


"""CLI test helpers."""


from argparse import Namespace
from pathlib import Path


def build_command_args(tmp_path: Path, **overrides: Any) -> Namespace:
    """Create a minimal argparse Namespace for build command tests."""
    defaults: dict[str, Any] = {
        "backend": "cuda",
        "source_dir": tmp_path / "source",
        "build_dir": tmp_path / "build",
        "output_dir": tmp_path / "output",
        "git_remote": "https://github.com/ggerganov/llama.cpp",
        "git_branch": "main",
        "git_commit": None,
        "jobs": 1,
        "json": False,
        "dry_run": False,
        "no_shallow_clone": False,
        "no_update_sources": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def namespace(**overrides: Any) -> Namespace:
    """Create a generic argparse Namespace."""
    return Namespace(**overrides)


"""Reusable object factories for tests."""


from llama_manager.build_pipeline import BuildBackend, BuildConfig
from llama_manager.config import (
    ModelSlot,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    ServerConfig,
    SmokePhase,
    SmokeProbeStatus,
)
from llama_manager.probe import ProvenanceRecord, SmokeProbeResult
from llama_manager.reports import FailureReport


def make_server_config(**overrides: object) -> ServerConfig:
    """Create a minimal ServerConfig for tests."""
    defaults: dict[str, object] = {
        "model": "/models/test.gguf",
        "alias": "test-slot",
        "device": "SYCL0",
        "port": 8080,
        "ctx_size": 4096,
        "ubatch_size": 512,
        "threads": 4,
        "server_bin": "dummy-llama-server",
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)  # type: ignore[arg-type]


def make_build_config(tmp_path: Path, **overrides: object) -> BuildConfig:
    """Create a minimal BuildConfig rooted under tmp_path."""
    defaults: dict[str, object] = {
        "backend": BuildBackend.CUDA,
        "source_dir": tmp_path / "llama.cpp",
        "build_dir": tmp_path / "build",
        "output_dir": tmp_path / "output",
        "git_remote_url": "https://github.com/ggerganov/llama.cpp",
        "git_branch": "main",
    }
    defaults.update(overrides)
    return BuildConfig(**defaults)  # type: ignore[arg-type]


def make_model_slot(**overrides: object) -> ModelSlot:
    """Create a minimal ModelSlot for tests."""
    defaults: dict[str, object] = {
        "slot_id": "slot1",
        "model_path": "/models/model.gguf",
        "port": 8080,
    }
    defaults.update(overrides)
    return ModelSlot(**defaults)  # type: ignore[arg-type]


def make_smoke_result(**overrides: object) -> SmokeProbeResult:
    """Create a minimal SmokeProbeResult for tests."""
    defaults: dict[str, object] = {
        "slot_id": "slot1",
        "status": SmokeProbeStatus.PASS,
        "phase_reached": SmokePhase.COMPLETE,
        "failure_phase": None,
        "latency_ms": 12,
        "provenance": ProvenanceRecord(sha="abc1234", version="test"),
    }
    defaults.update(overrides)
    return SmokeProbeResult(**defaults)  # type: ignore[arg-type]


def make_failure_report(tmp_path: Path, **overrides: object) -> FailureReport:
    """Create a minimal FailureReport for tests."""
    defaults: dict[str, object] = {
        "report_dir": tmp_path / "reports" / "2026-04-15T12-30-00",
        "build_artifact_json": "{}",
        "build_output": "",
        "error_details": [],
    }
    defaults.update(overrides)
    return FailureReport(**defaults)  # type: ignore[arg-type]


def make_profile_record(**overrides: object) -> ProfileRecord:
    """Create a minimal ProfileRecord for tests."""
    defaults: dict[str, object] = {
        "gpu_identifier": "test-gpu",
        "backend": "cuda",
        "flavor": ProfileFlavor.BALANCED,
        "driver_version": "1.0",
        "driver_version_hash": "0" * 16,
        "server_binary_version": "test",
        "profiled_at": "2026-04-15T12:00:00Z",
        "metrics": ProfileMetrics(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=1024,
        ),
    }
    defaults.update(overrides)
    return ProfileRecord(**defaults)  # type: ignore[arg-type]


"""Runtime filesystem helpers for tests."""


from collections.abc import Callable

from llama_manager.orchestration import create_lock


def make_runtime_dir(tmp_path: Path) -> Path:
    """Create and return an isolated runtime directory."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    return runtime_dir


def make_slot_lock(
    runtime_dir: Path,
    slot_id: str = "slot1",
    *,
    pid: int = 12345,
    port: int = 8080,
) -> Path:
    """Create a lockfile for a slot."""
    return create_lock(runtime_dir, slot_id, pid=pid, port=port)


def capture_artifact_path(
    output_path: Path,
    sink: list[dict],
) -> Callable[[Path, str, dict], Path]:
    """Build a write_artifact side effect that records payloads."""

    def _capture_artifact(_runtime_dir: Path, _slot_id: str, data: dict) -> Path:
        sink.append(data)
        return output_path

    return _capture_artifact


def fixture_path(name: str) -> Path:
    """Return a path inside the shared tests/fixtures directory."""
    return Path(__file__).resolve().parents[1] / "fixtures" / name


def valid_artifact_data(**overrides: object) -> dict[str, object]:
    """Create valid FR-007 dry-run artifact data.

    Uses per-slot dicts for resolved_command and validation_results.
    """
    data: dict[str, object] = {
        "timestamp": "2026-04-12T00:00:00Z",
        "slot_scope": ["slot1"],
        "resolved_command": {"slot1": {"cmd": ["echo", "test"]}},
        "validation_results": {"slot1": {"passed": True, "checks": []}},
        "warnings": [],
        "environment_redacted": {},
    }
    data.update(overrides)
    return data
