"""Reusable object factories for tests."""

from pathlib import Path

from llama_manager.build_pipeline import BuildBackend, BuildConfig
from llama_manager.config import (
    ModelSlot,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    ServerConfig,
    SmokeFailurePhase,
    SmokePhase,
    SmokeProbeStatus,
)
from llama_manager.reports import FailureReport
from llama_manager.smoke import ProvenanceRecord, SmokeProbeResult


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
        "phase": SmokePhase.CHAT_COMPLETION,
        "failure_phase": SmokeFailurePhase.NONE,
        "latency_ms": 12.5,
        "message": "ok",
        "provenance": ProvenanceRecord(version="test", git_sha="abc1234"),
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
