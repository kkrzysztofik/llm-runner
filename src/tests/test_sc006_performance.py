import tempfile
import time
from math import ceil
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.dry_run import dry_run
from llama_manager.server import validate_port, validate_ports


def get_p95(data: list[float]) -> float:
    """Calculate p95 percentile."""
    sorted_data = sorted(data)
    if not sorted_data:
        raise ValueError("data must not be empty")
    idx = ceil(len(sorted_data) * 0.95) - 1
    idx = max(0, min(idx, len(sorted_data) - 1))
    return sorted_data[idx]


@patch("llama_cli.dry_run.write_artifact", return_value=tempfile.gettempdir() + "/fake_artifact")
@patch("llama_cli.dry_run.resolve_runtime_dir", return_value=tempfile.gettempdir())
@patch("llama_cli.dry_run.validate_server_config", return_value=None)
@patch("sys.stdout", new_callable=MagicMock)
@patch("sys.stderr", new_callable=MagicMock)
def test_performance_dry_run_resolution(
    mock_stderr,
    mock_stdout,
    mock_validate,
    mock_runtime,
    mock_artifact,
):
    """T041: Benchmark dry-run resolution time."""
    iterations = 100
    times: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        dry_run("summary-balanced")
        end = time.perf_counter()
        times.append(end - start)

    p95 = get_p95(times)
    # Requirement: single-slot dry-run <= 250ms
    assert p95 <= 0.250, f"p95 dry-run resolution too slow: {p95:.4f}s"


def test_performance_validation_paths():
    """T041: Benchmark lock/port validation paths."""
    iterations = 100

    # Per-slot lock/port validation
    port_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        validate_port(8080, "test_port")
        end = time.perf_counter()
        port_times.append(end - start)

    p95_port = get_p95(port_times)
    # Requirement: per-slot lock/port <= 150ms
    assert p95_port <= 0.150, f"p95 port validation too slow: {p95_port:.4f}s"

    # Port conflict validation
    conflict_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with pytest.raises(SystemExit):
            validate_ports(8080, 8080, "p1", "p2")
        end = time.perf_counter()
        conflict_times.append(end - start)

    p95_conflict = get_p95(conflict_times)
    assert p95_conflict <= 0.150, f"p95 port conflict validation too slow: {p95_conflict:.4f}s"


@patch("llama_cli.dry_run.write_artifact", return_value=tempfile.gettempdir() + "/fake_artifact")
@patch("llama_cli.dry_run.resolve_runtime_dir", return_value=tempfile.gettempdir())
@patch("llama_cli.dry_run.validate_server_config", return_value=None)
@patch("sys.stdout", new_callable=MagicMock)
@patch("sys.stderr", new_callable=MagicMock)
def test_performance_dry_run_two_slots(
    mock_stderr,
    mock_stdout,
    mock_validate,
    mock_runtime,
    mock_artifact,
):
    """T041: Benchmark two-slot dry-run resolution time."""
    iterations = 100
    times: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        # 'both' mode uses two slots
        dry_run("both")
        end = time.perf_counter()
        times.append(end - start)

    p95 = get_p95(times)
    # Requirement: two-slot <= 400ms
    assert p95 <= 0.400, f"p95 two-slot dry-run resolution too slow: {p95:.4f}s"
