"""Tests for build CLI interface (build_cli.py).

Covers:
- parse_build_args: argument parsing for all options
- run_build_command: success/failure paths, JSON output
- main: entry point, KeyboardInterrupt handling
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_cli.build_cli import (
    main,
    parse_build_args,
    run_build_command,
)
from llama_manager.build_pipeline import (
    BuildArtifact,
    BuildPipeline,
    BuildResult,
)

# =============================================================================
# parse_build_args
# =============================================================================


class TestParseBuildArgs:
    """Tests for parse_build_args function."""

    def test_parse_backend_sycl(self) -> None:
        """parse_build_args should parse 'sycl' backend."""
        args = parse_build_args(["sycl"])
        assert args.backend == "sycl"
        assert args.dry_run is False
        assert args.json is False
        assert args.jobs is None
        assert args.retry_attempts == 2
        assert args.retry_delay == 5

    def test_parse_backend_cuda(self) -> None:
        """parse_build_args should parse 'cuda' backend."""
        args = parse_build_args(["cuda"])
        assert args.backend == "cuda"

    def test_parse_backend_both(self) -> None:
        """parse_build_args should parse 'both' backend."""
        args = parse_build_args(["both"])
        assert args.backend == "both"

    def test_parse_all_options(self) -> None:
        """parse_build_args should parse all options."""
        args = parse_build_args(
            [
                "sycl",
                "--source-dir",
                "/custom/source",
                "--build-dir",
                "/custom/build",
                "--output-dir",
                "/custom/output",
                "--jobs",
                "8",
                "--retry-attempts",
                "5",
                "--retry-delay",
                "10",
                "--dry-run",
                "--json",
            ]
        )
        assert args.backend == "sycl"
        assert args.source_dir == Path("/custom/source")
        assert args.build_dir == Path("/custom/build")
        assert args.output_dir == Path("/custom/output")
        assert args.jobs == 8
        assert args.retry_attempts == 5
        assert args.retry_delay == 10
        assert args.dry_run is True
        assert args.json is True

    def test_parse_git_options(self) -> None:
        """parse_build_args should parse git remote and branch."""
        args = parse_build_args(
            ["sycl", "--git-remote", "https://custom/repo.git", "--git-branch", "dev"]
        )
        assert args.git_remote == "https://custom/repo.git"
        assert args.git_branch == "dev"

    def test_parse_no_shallow_clone(self) -> None:
        """parse_build_args should parse --no-shallow-clone flag."""
        args = parse_build_args(["sycl", "--no-shallow-clone"])
        assert args.no_shallow_clone is True

    def test_parse_default_git_options(self) -> None:
        """parse_build_args should use default git remote and branch."""
        args = parse_build_args(["sycl"])
        assert args.git_remote == "https://github.com/ggerganov/llama.cpp.git"
        assert args.git_branch == "master"

    def test_parse_short_jobs_flag(self) -> None:
        """parse_build_args should parse -j flag."""
        args = parse_build_args(["sycl", "-j", "4"])
        assert args.jobs == 4


# =============================================================================
# run_build_command
# =============================================================================


class TestRunBuildCommand:
    """Tests for run_build_command function."""

    def _make_mock_pipeline(self, success: bool = True) -> MagicMock:
        """Create a mock BuildPipeline that returns the given success status."""
        mock_pipeline = MagicMock(spec=BuildPipeline)
        mock_pipeline.dry_run = False
        mock_result = BuildResult(
            success=success,
            artifact=BuildArtifact(
                artifact_type="llama-server",
                backend="sycl",
                created_at=0.0,
                git_remote_url="https://github.com/ggerganov/llama.cpp",
                git_commit_sha="abc123",
                git_branch="main",
                build_command=["cmake", "--build"],
                build_duration_seconds=10.0,
                exit_code=0,
                binary_path=Path("/tmp/llama-server"),
                binary_size_bytes=104857600,
                build_log_path=None,
                failure_report_path=None,
            )
            if success
            else None,
            error_message="" if success else "Build failed",
        )
        mock_pipeline.run.return_value = mock_result
        return mock_pipeline

    def test_run_build_sycl_success(self, tmp_path: Path, capsys) -> None:
        """run_build_command should succeed for SYCL backend."""
        args = parse_build_args(["sycl"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Build completed successfully" in captured.err

    def test_run_build_cuda_success(self, tmp_path: Path, capsys) -> None:
        """run_build_command should succeed for CUDA backend."""
        args = parse_build_args(["cuda"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 0

    def test_run_build_both_success(self, tmp_path: Path, capsys) -> None:
        """run_build_command should build both backends sequentially."""
        args = parse_build_args(["both"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 0
            assert mock_cls.call_count == 2  # SYCL + CUDA

    def test_run_build_single_failure(self, tmp_path: Path, capsys) -> None:
        """run_build_command should return 1 when single backend fails."""
        args = parse_build_args(["sycl"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(False)
            mock_pipeline.run.return_value = BuildResult(
                success=False,
                error_message="Compilation failed",
            )
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Build failed" in captured.err

    def test_run_build_both_partial_failure(self, tmp_path: Path, capsys) -> None:
        """run_build_command should continue on failure for 'both' and return 1."""
        args = parse_build_args(["both"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_success = self._make_mock_pipeline(True)
            mock_failure = self._make_mock_pipeline(False)
            mock_failure.run.return_value = BuildResult(
                success=False,
                error_message="CUDA build failed",
            )
            mock_cls.side_effect = [mock_success, mock_failure]

            exit_code = run_build_command(args)

            assert exit_code == 1  # Overall failure

    def test_run_build_json_output_success(self, tmp_path: Path, capsys) -> None:
        """run_build_command --json should output JSON on success."""
        args = parse_build_args(["sycl", "--json"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["success"] is True
            assert len(parsed["artifacts"]) == 1
            assert parsed["artifacts"][0]["artifact_type"] == "llama-server"

    def test_run_build_json_output_failure(self, tmp_path: Path, capsys) -> None:
        """run_build_command --json should output JSON errors on failure."""
        args = parse_build_args(["sycl", "--json"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(False)
            mock_pipeline.run.return_value = BuildResult(
                success=False,
                error_message="Compilation failed",
            )
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert exit_code == 1
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["success"] is False
            assert len(parsed["errors"]) == 1
            assert "Compilation failed" in parsed["errors"][0]["error"]

    def test_run_build_dry_run(self, tmp_path: Path, capsys) -> None:
        """run_build_command --dry-run should set dry_run on pipeline."""
        args = parse_build_args(["sycl", "--dry-run"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            exit_code = run_build_command(args)

            assert mock_pipeline.dry_run is True
            assert exit_code == 0

    def test_run_build_artifact_path_converted(self, tmp_path: Path, capsys) -> None:
        """run_build_command JSON should convert Path values to strings."""
        args = parse_build_args(["sycl", "--json"])
        args.source_dir = tmp_path / "source"
        args.build_dir = tmp_path / "build"
        args.output_dir = tmp_path / "output"

        with patch("llama_cli.build_cli.BuildPipeline") as mock_cls:
            mock_pipeline = self._make_mock_pipeline(True)
            mock_cls.return_value = mock_pipeline

            run_build_command(args)

            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            # binary_path should be a string, not a Path object
            assert isinstance(parsed["artifacts"][0]["binary_path"], str)


# =============================================================================
# main
# =============================================================================


class TestMain:
    """Tests for main entry point."""

    def test_main_keyboard_interrupt(self, tmp_path: Path) -> None:
        """main should return 130 on KeyboardInterrupt."""
        with patch("llama_cli.build_cli.parse_build_args") as mock_parse:
            mock_parse.side_effect = KeyboardInterrupt()

            exit_code = main()

            assert exit_code == 130

    def test_main_generic_exception(self, tmp_path: Path) -> None:
        """main should return 1 on generic exception."""
        with patch("llama_cli.build_cli.parse_build_args") as mock_parse:
            mock_parse.side_effect = ValueError("something went wrong")

            exit_code = main()

            assert exit_code == 1

    def test_main_generic_exception_json_mode(self, tmp_path: Path, capsys) -> None:
        """main should output JSON on exception when --json flag is set."""
        args = argparse.Namespace(json=True)
        with patch("llama_cli.build_cli.parse_build_args") as mock_parse:
            mock_parse.return_value = args
            with patch("llama_cli.build_cli.run_build_command") as mock_run:
                mock_run.side_effect = ValueError("build error")

                exit_code = main()

                assert exit_code == 1
                captured = capsys.readouterr()
                parsed = json.loads(captured.out)
                assert parsed["success"] is False
                assert parsed["error"] == "build error"

    def test_main_success_path(self, tmp_path: Path) -> None:
        """main should delegate to run_build_command on success."""
        args = parse_build_args(["sycl"])
        with (
            patch.object(
                sys.modules["llama_cli.build_cli"],
                "parse_build_args",
                return_value=args,
            ),
            patch.object(
                sys.modules["llama_cli.build_cli"],
                "run_build_command",
                return_value=0,
            ) as mock_run,
        ):
            exit_code = main()
            mock_run.assert_called_once_with(args)
            assert exit_code == 0
