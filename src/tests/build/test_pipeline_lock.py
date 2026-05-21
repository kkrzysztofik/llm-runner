"""Tests for build_pipeline/lock.py lock management."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from llama_manager.build_pipeline.lock import (
    acquire_lock,
    get_lock_error_message,
    is_lock_stale,
    release_lock,
)

# ── acquire_lock ───────────────────────────────────────────────────────────


class TestAcquireLock:
    def test_dry_run_returns_true(self) -> None:
        assert acquire_lock(Path("/tmp/test.lock"), "sycl", dry_run=True) is True

    def test_success(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        with patch("llama_manager.build_pipeline.lock.atomic_exclusive_create_json") as mock_create:
            mock_create.return_value = None
            result = acquire_lock(lock_path, "sycl")
        assert result is True
        mock_create.assert_called_once()

    def test_existing_active_fails(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        with (
            patch("llama_manager.build_pipeline.lock.atomic_exclusive_create_json") as mock_create,
            patch("llama_manager.build_pipeline.lock.is_lock_stale") as mock_stale,
        ):
            mock_create.side_effect = FileExistsError()
            mock_stale.return_value = False
            result = acquire_lock(lock_path, "sycl")
        assert result is False

    def test_stale_lock_acquired(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        with (
            patch("llama_manager.build_pipeline.lock.atomic_exclusive_create_json") as mock_create,
            patch("llama_manager.build_pipeline.lock.is_lock_stale") as mock_stale,
            patch("llama_manager.build_pipeline.lock.atomic_write_json") as mock_write,
        ):
            mock_create.side_effect = FileExistsError()
            mock_stale.return_value = True
            mock_write.return_value = None
            result = acquire_lock(lock_path, "sycl")
        assert result is True
        mock_write.assert_called_once()

    def test_stale_lock_replacement_fails(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        with (
            patch("llama_manager.build_pipeline.lock.atomic_exclusive_create_json") as mock_create,
            patch("llama_manager.build_pipeline.lock.is_lock_stale") as mock_stale,
            patch("llama_manager.build_pipeline.lock.atomic_write_json") as mock_write,
        ):
            mock_create.side_effect = FileExistsError()
            mock_stale.return_value = True
            mock_write.side_effect = OSError("permission denied")
            result = acquire_lock(lock_path, "sycl")
        assert result is False


# ── is_lock_stale ──────────────────────────────────────────────────────────


class TestIsLockStale:
    def test_missing_file(self, tmp_path: Path):
        lock_path = tmp_path / "nonexistent.lock"
        assert is_lock_stale(lock_path) is True

    def test_invalid_json(self, tmp_path: Path):
        lock_path = tmp_path / "invalid.lock"
        lock_path.write_text("not json {{{")
        assert is_lock_stale(lock_path) is True

    def test_missing_pid(self, tmp_path: Path):
        lock_path = tmp_path / "no_pid.lock"
        lock_path.write_text(json.dumps({"started_at": str(time.time())}))
        assert is_lock_stale(lock_path) is True

    def test_missing_started_at(self, tmp_path: Path):
        lock_path = tmp_path / "no_started.lock"
        lock_path.write_text(json.dumps({"pid": 1234}))
        assert is_lock_stale(lock_path) is True

    def test_pid_exists_not_stale(self, tmp_path: Path):
        lock_path = tmp_path / "valid.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 1,
                    "started_at": str(time.time()),
                    "backend": "sycl",
                }
            )
        )
        with patch("llama_manager.build_pipeline.lock.psutil.pid_exists") as mock_pid:
            mock_pid.return_value = True
            with patch("llama_manager.build_pipeline.lock.BuildLock") as mock_lock:
                mock_lock.return_value.is_stale.return_value = False
                assert is_lock_stale(lock_path) is False

    def test_pid_missing_stale(self, tmp_path: Path):
        lock_path = tmp_path / "dead_pid.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "started_at": str(time.time()),
                    "backend": "sycl",
                }
            )
        )
        with patch("llama_manager.build_pipeline.lock.psutil.pid_exists") as mock_pid:
            mock_pid.return_value = False
            assert is_lock_stale(lock_path) is True

    def test_stale_timeout(self, tmp_path: Path):
        lock_path = tmp_path / "old.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 1,
                    "started_at": str(time.time() - 3600),
                    "backend": "sycl",
                }
            )
        )
        with patch("llama_manager.build_pipeline.lock.psutil.pid_exists") as mock_pid:
            mock_pid.return_value = True
            with patch("llama_manager.build_pipeline.lock.BuildLock") as mock_lock:
                mock_lock.return_value.is_stale.return_value = True
                assert is_lock_stale(lock_path) is True

    def test_invalid_data_type(self, tmp_path: Path):
        lock_path = tmp_path / "list.lock"
        lock_path.write_text(json.dumps(["not", "a", "dict"]))
        assert is_lock_stale(lock_path) is True


# ── release_lock ───────────────────────────────────────────────────────────


class TestReleaseLock:
    def test_success(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        lock_path.write_text("{}")
        release_lock(lock_path)
        assert not lock_path.exists()

    def test_no_lock_file(self) -> None:
        release_lock(None)

    def test_nonexistent_path(self, tmp_path: Path):
        lock_path = tmp_path / "nonexistent.lock"
        release_lock(lock_path)


# ── get_lock_error_message ─────────────────────────────────────────────────


class TestGetLockErrorMessage:
    def test_active_lock(self, tmp_path: Path):
        lock_path = tmp_path / "build.lock"
        lock_path.write_text(json.dumps({"pid": 1234, "backend": "sycl"}))
        result = get_lock_error_message(lock_path)
        assert "1234" in result
        assert "sycl" in result

    def test_invalid_json(self, tmp_path: Path):
        lock_path = tmp_path / "invalid.lock"
        lock_path.write_text("not json")
        result = get_lock_error_message(lock_path)
        assert "could not be read" in result

    def test_missing_file(self, tmp_path: Path):
        lock_path = tmp_path / "nonexistent.lock"
        result = get_lock_error_message(lock_path)
        assert "could not be read" in result

    def test_missing_pid(self, tmp_path: Path):
        lock_path = tmp_path / "no_pid.lock"
        lock_path.write_text(json.dumps({"backend": "sycl"}))
        result = get_lock_error_message(lock_path)
        assert "unknown" in result
