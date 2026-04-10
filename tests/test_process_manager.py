"""Tests for llama_manager.process_manager.

Focused tests for:
- resolve_runtime_dir fallback behavior (T001)
- Runtime directory usability (T002)
"""

from pathlib import Path

import pytest

from llama_manager.process_manager import resolve_runtime_dir


class TestResolveRuntimeDir:
    """Tests for resolve_runtime_dir fallback and usability."""

    def test_env_var_takes_precedence(self, tmp_path: Path) -> None:
        """LLM_RUNNER_RUNTIME_DIR should take precedence over XDG_RUNTIME_DIR."""
        env_runtime = tmp_path / "env_runtime"
        xdg_runtime = tmp_path / "xdg_runtime"
        env_runtime.mkdir()
        xdg_runtime.mkdir()

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(env_runtime))
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_runtime))
            result = resolve_runtime_dir()
            assert result == env_runtime

    def test_falls_back_to_xdg_runtime_dir(self, tmp_path: Path) -> None:
        """Should fall back to XDG_RUNTIME_DIR/llm-runner when env var not set."""
        xdg_base = tmp_path / "xdg_runtime"
        xdg_llm_runner = xdg_base / "llm-runner"

        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_base))
            result = resolve_runtime_dir()
            # Should create the llm-runner subdirectory
            assert result == xdg_llm_runner
            assert result.exists()
            assert result.is_dir()

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """resolve_runtime_dir should create the directory if it doesn't exist."""
        target = tmp_path / "new_runtime"

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result = resolve_runtime_dir()
            assert result == target
            assert result.exists()
            assert result.is_dir()

    def test_no_env_vars_raises_error(self) -> None:
        """Should raise RuntimeError when neither env var is set."""
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.delenv("XDG_RUNTIME_DIR", raising=False)
            with pytest.raises(RuntimeError) as exc_info:
                resolve_runtime_dir()
            assert "No usable runtime directory" in str(exc_info.value)

    def test_xdg_creates_subdirectory(self, tmp_path: Path) -> None:
        """XDG_RUNTIME_DIR fallback should create llm-runner subdirectory."""
        xdg_base = tmp_path / "xdg_base"

        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_base))
            result = resolve_runtime_dir()
            # Should be XDG_RUNTIME_DIR/llm-runner
            assert result.name == "llm-runner"
            assert result.parent == xdg_base

    def test_env_var_path_with_spaces(self, tmp_path: Path) -> None:
        """Should handle runtime directory paths with spaces."""
        target = tmp_path / "runtime with spaces"

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result = resolve_runtime_dir()
            assert result == target
            assert result.exists()

    def test_writable_directory_required(self, tmp_path: Path) -> None:
        """Should skip non-writable directories and try next fallback."""
        env_dir = tmp_path / "env_dir"
        xdg_dir = tmp_path / "xdg_dir" / "llm-runner"
        env_dir.mkdir()
        xdg_dir.parent.mkdir(parents=True)

        # Make env_dir read-only (simulate non-writable)
        env_dir.chmod(0o555)

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(env_dir))
                mp.setenv("XDG_RUNTIME_DIR", str(xdg_dir.parent))
                result = resolve_runtime_dir()
                # Should fall back to XDG_RUNTIME_DIR
                assert result == xdg_dir
                assert result.exists()
        finally:
            # Restore permissions
            env_dir.chmod(0o755)

    def test_multiple_calls_return_same_path(self, tmp_path: Path) -> None:
        """Multiple calls with same env vars should return same Path object."""
        target = tmp_path / "runtime"
        target.mkdir()

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result1 = resolve_runtime_dir()
            result2 = resolve_runtime_dir()
            assert result1 == result2
            assert result1 == target
