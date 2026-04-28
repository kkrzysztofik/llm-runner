"""Runtime filesystem helpers for tests."""

from collections.abc import Callable
from pathlib import Path

from llama_manager.process_manager import create_lock


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
    """Create valid FR-007 dry-run artifact data."""
    data: dict[str, object] = {
        "timestamp": "2026-04-12T00:00:00Z",
        "slot_scope": ["slot1"],
        "resolved_command": {"cmd": ["echo", "test"]},
        "validation_results": {"passed": True, "checks": []},
        "warnings": [],
        "environment_redacted": {},
    }
    data.update(overrides)
    return data
