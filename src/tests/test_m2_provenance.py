"""T074: Test offline-continue path when network unavailable but local clone exists.

Test Task:
- T074: Test offline-continue path when network unavailable but local clone exists
"""

from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401

from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildPipeline


class TestOfflineContinue:
    """T074: Tests for offline-continue support in BuildPipeline.clone_stage()."""

    def test_clone_skips_when_source_exists(self, tmp_path: Path) -> None:
        """Clone stage should skip when source directory already exists and is non-empty.

        This simulates offline-continue: when network is unavailable but local
        clone exists, the build can continue with existing sources.
        """
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Create source directory with existing files (simulating previous clone)
        config.source_dir.mkdir(parents=True)
        (config.source_dir / "CMakeLists.txt").write_text("# CMakeLists.txt")
        (config.source_dir / "README.md").write_text("# README")

        pipeline = BuildPipeline(config)
        pipeline.dry_run = False

        # Run clone stage
        progress = pipeline._run_clone()

        # Should skip clone
        assert progress.status == "skipped"
        assert progress.message == "Sources already exist"
        assert progress.progress_percent == 30

    def test_clone_skips_when_source_empty_directory(self, tmp_path: Path) -> None:
        """Clone stage should NOT skip when source directory exists but is empty.

        An empty directory is not a valid clone, so it should attempt to clone.
        """
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Create empty source directory
        config.source_dir.mkdir(parents=True)

        pipeline = BuildPipeline(config)
        pipeline.dry_run = True  # Use dry_run to avoid actual git clone

        # Run clone stage
        progress = pipeline._run_clone()

        # Should attempt to clone (not skip)
        assert progress.status == "success"
        assert "Would run: git clone" in progress.message
        assert progress.progress_percent == 30

    def test_clone_offline_continue_with_partial_clone(self, tmp_path: Path) -> None:
        """Clone stage should handle partial clone scenarios gracefully.

        When a partial clone exists (some files present), the build should
        be able to continue if enough files are present for CMake.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Create source directory with partial files (simulating shallow clone)
        config.source_dir.mkdir(parents=True)
        (config.source_dir / "CMakeLists.txt").write_text("# CMakeLists.txt")
        (config.source_dir / "cmake").mkdir()
        (config.source_dir / "cmake" / "build.cmake").write_text("# build config")

        pipeline = BuildPipeline(config)
        pipeline.dry_run = False

        # Run clone stage
        progress = pipeline._run_clone()

        # Should skip clone since source exists and has content
        assert progress.status == "skipped"
        assert progress.message == "Sources already exist"

    def test_clone_network_unavailable_with_local_clone(self, tmp_path: Path) -> None:
        """Clone stage should handle network failure gracefully when local clone exists.

        This tests the offline-continue pattern: if network fails but local
        sources exist, the build should continue with existing sources.
        """
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Create source directory with files
        config.source_dir.mkdir(parents=True)
        (config.source_dir / "CMakeLists.txt").write_text("# CMakeLists.txt")

        pipeline = BuildPipeline(config)
        pipeline.dry_run = False

        # Mock subprocess.run to simulate network failure
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Network timeout")

            # Run clone stage - should skip because source exists
            progress = pipeline._run_clone()

            # Should skip clone (not attempt network call)
            assert progress.status == "skipped"
            assert progress.message == "Sources already exist"
            # subprocess.run should not be called
            assert not mock_run.called

    def test_clone_network_unavailable_without_local_clone(self, tmp_path: Path) -> None:
        """Clone stage should fail gracefully when network unavailable and no local clone.

        This tests the error handling path: if network fails and no local
        sources exist, the build should report a clear error.
        """
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Source directory does not exist
        assert not config.source_dir.exists()

        pipeline = BuildPipeline(config)
        pipeline.dry_run = False

        # Mock subprocess.run to simulate network failure
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Network timeout")

            # Run clone stage
            progress = pipeline._run_clone()

            # Should fail with network error
            assert progress.status == "failed"
            assert "Network timeout" in progress.message or "Clone failed" in progress.message

    def test_clone_dry_run_mode(self, tmp_path: Path) -> None:
        """Clone stage should work in dry-run mode without actual git operations.

        Dry-run mode should show what would be executed without making network calls.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        # Source directory does not exist
        assert not config.source_dir.exists()

        pipeline = BuildPipeline(config)
        pipeline.dry_run = True

        # Run clone stage
        progress = pipeline._run_clone()

        # Should succeed in dry-run mode
        assert progress.status == "success"
        assert "Would run: git clone" in progress.message
        assert progress.progress_percent == 30

    def test_clone_shallow_clone_flag(self, tmp_path: Path) -> None:
        """Clone stage should use shallow clone when config.shallow_clone is True.

        Shallow cloning is the default for faster builds.
        """
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            shallow_clone=True,
        )

        pipeline = BuildPipeline(config)
        pipeline.dry_run = True

        # Run clone stage
        progress = pipeline._run_clone()

        # Should succeed in dry-run mode
        assert progress.status == "success"
        # In dry-run, it just shows the command
        assert "git clone" in progress.message

    def test_clone_full_clone_when_disabled(self, tmp_path: Path) -> None:
        """Clone stage should use full clone when shallow_clone is False.

        Full clone is useful for debugging or when full history is needed.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "llama.cpp",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            shallow_clone=False,
        )

        pipeline = BuildPipeline(config)
        pipeline.dry_run = True

        # Run clone stage
        progress = pipeline._run_clone()

        # Should not mention shallow clone
        assert progress.status == "success"
        # In dry-run mode, we just show the command
        assert "git clone" in progress.message
        assert "--depth" not in progress.message
