"""Tests for server runner module (server_runner.py).

Covers:
- _resolve_port: port resolution with defaults
- _normalize_main_args: argument normalization
- _build_target_configs: config building for different modes
- _run_dry_run_mode: dry-run handling
- run_build: build execution
- verify_risks: risk verification
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_cli.server_runner import (
    _build_target_configs,
    _normalize_main_args,
    _resolve_port,
    _run_dry_run_mode,
    run_build,
    verify_risks,
)
from llama_manager import Config, ServerConfig, ServerManager
from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildResult

# =============================================================================
# _resolve_port
# =============================================================================


class TestResolvePort:
    """Tests for _resolve_port helper."""

    def test_resolve_port_from_list(self) -> None:
        """_resolve_port should return port from list when available."""
        port = _resolve_port([8080, 8081], 0, 8080)
        assert port == 8080

    def test_resolve_port_from_list_secondary(self) -> None:
        """_resolve_port should return secondary port from list."""
        port = _resolve_port([8080, 8081], 1, 8081)
        assert port == 8081

    def test_resolve_port_default_when_empty(self) -> None:
        """_resolve_port should return default when list is empty."""
        port = _resolve_port([], 0, 8080)
        assert port == 8080

    def test_resolve_port_default_when_index_out_of_range(self) -> None:
        """_resolve_port should return default when index > list length."""
        port = _resolve_port([8080], 1, 8081)
        assert port == 8081

    def test_resolve_port_default_when_list_shorter(self) -> None:
        """_resolve_port should return default for missing ports."""
        port = _resolve_port([8080], 2, 9999)
        assert port == 9999


# =============================================================================
# _normalize_main_args
# =============================================================================


class TestNormalizeMainArgs:
    """Tests for _normalize_main_args helper."""

    def test_normalize_none_args(self) -> None:
        """_normalize_main_args with None should return sys.argv[1:]."""
        import sys

        result = _normalize_main_args(None)
        assert result == sys.argv[1:]

    def test_normalize_empty_args(self) -> None:
        """_normalize_main_args with empty list should return empty list."""
        result = _normalize_main_args([])
        assert result == []

    def test_normalize_valid_mode(self) -> None:
        """_normalize_main_args should pass through valid modes unchanged."""
        for mode in [
            "summary-balanced",
            "summary-fast",
            "qwen35",
            "both",
            "dry-run",
            "doctor",
            "build",
            "setup",
        ]:
            result = _normalize_main_args([mode])
            assert result == [mode]

    def test_normalize_flag_args(self) -> None:
        """_normalize_main_args should pass through flag args unchanged."""
        result = _normalize_main_args(["--port", "8080"])
        assert result == ["--port", "8080"]

    def test_normalize_skip_first_when_invalid_mode(self) -> None:
        """_normalize_main_args should skip first arg when it's not a valid mode."""
        result = _normalize_main_args(["not-a-mode", "summary-balanced"])
        assert result == ["summary-balanced"]

    def test_normalize_non_flag_non_mode_passes_through(self) -> None:
        """_normalize_main_args should pass through non-flag args that aren't modes."""
        # "8080" is not a valid mode and doesn't start with "-",
        # so it should be skipped
        result = _normalize_main_args(["8080"])
        assert result == []


# =============================================================================
# _build_target_configs
# =============================================================================


class TestBuildTargetConfigs:
    """Tests for _build_target_configs helper."""

    def test_build_configs_summary_balanced(self) -> None:
        """_build_target_configs should create summary-balanced config."""
        cfg = Config()
        configs = _build_target_configs("summary-balanced", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "summary-balanced"

    def test_build_configs_qwen35(self) -> None:
        """_build_target_configs should create qwen35 config."""
        cfg = Config()
        configs = _build_target_configs("qwen35", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "qwen35-coding"

    def test_build_configs_both(self) -> None:
        """_build_target_configs should create two configs for 'both'."""
        cfg = Config()
        configs = _build_target_configs("both", [], cfg)
        assert len(configs) == 2
        assert configs[0].alias == "summary-balanced"
        assert configs[1].alias == "qwen35-coding"

    def test_build_configs_with_ports(self) -> None:
        """_build_target_configs should use provided ports."""
        cfg = Config()
        configs = _build_target_configs("qwen35", [9999], cfg)
        assert len(configs) == 1
        assert configs[0].port == 9999

    def test_build_configs_unknown_mode(self) -> None:
        """_build_target_configs should return empty list for unknown mode."""
        cfg = Config()
        configs = _build_target_configs("unknown-mode", [], cfg)
        assert configs == []

    def test_build_configs_summary_fast(self) -> None:
        """_build_target_configs should create summary-fast config."""
        cfg = Config()
        configs = _build_target_configs("summary-fast", [], cfg)
        assert len(configs) == 1
        assert configs[0].alias == "summary-fast"


# =============================================================================
# _run_dry_run_mode
# =============================================================================


class TestRunDryRunMode:
    """Tests for _run_dry_run_mode helper."""

    def test_dry_run_missing_mode(self) -> None:
        """_run_dry_run_mode should return 1 when dry_run_mode is missing."""
        parsed = argparse.Namespace(mode="dry-run", ports=[], acknowledge_risky=False)
        exit_code = _run_dry_run_mode(parsed, acknowledged=False)
        assert exit_code == 1

    @patch("llama_cli.server_runner.dry_run")
    def test_dry_run_with_valid_mode(self, mock_dry_run: MagicMock) -> None:
        """_run_dry_run_mode should call dry_run with correct args."""
        parsed = argparse.Namespace(
            mode="dry-run",
            dry_run_mode="both",
            ports=[8080, 8081],
            acknowledge_risky=True,
        )
        exit_code = _run_dry_run_mode(parsed, acknowledged=True)
        assert exit_code == 0
        mock_dry_run.assert_called_once_with("both", "8080", "8081", acknowledged=True)

    @patch("llama_cli.server_runner.dry_run")
    def test_dry_run_with_single_port(self, mock_dry_run: MagicMock) -> None:
        """_run_dry_run_mode should handle single port."""
        parsed = argparse.Namespace(
            mode="dry-run",
            dry_run_mode="summary-balanced",
            ports=[8080],
            acknowledge_risky=False,
        )
        exit_code = _run_dry_run_mode(parsed, acknowledged=False)
        assert exit_code == 0
        mock_dry_run.assert_called_once_with("summary-balanced", "8080", "", acknowledged=False)


# =============================================================================
# run_build
# =============================================================================


class TestRunBuild:
    """Tests for run_build function."""

    def test_run_build_sycl(self, tmp_path: Path) -> None:
        """run_build should create correct build config for SYCL."""
        with patch("llama_cli.server_runner.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.server_runner.BuildPipeline") as mock_pipeline_cls:
                mock_result = BuildResult(success=True)
                mock_pipeline_cls.return_value.run.return_value = mock_result
                mock_pipeline_cls.return_value.dry_run = False

                exit_code = run_build("sycl", dry_run=False)

                assert exit_code == 0
                # Verify pipeline was created with correct config
                call_args = mock_pipeline_cls.call_args
                build_config = call_args[0][0]
                assert isinstance(build_config, BuildConfig)
                assert build_config.backend == BuildBackend.SYCL

    def test_run_build_cuda(self, tmp_path: Path) -> None:
        """run_build should create correct build config for CUDA."""
        with patch("llama_cli.server_runner.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.server_runner.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(success=True)
                mock_pipeline_cls.return_value.dry_run = False

                exit_code = run_build("cuda", dry_run=False)

                assert exit_code == 0
                call_args = mock_pipeline_cls.call_args
                build_config = call_args[0][0]
                assert build_config.backend == BuildBackend.CUDA
                # CUDA uses build_cuda directory
                assert "build_cuda" in str(build_config.build_dir)

    def test_run_build_failure(self, tmp_path: Path) -> None:
        """run_build should return 1 on build failure."""
        with patch("llama_cli.server_runner.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.server_runner.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(
                    success=False,
                    error_message="Build failed",
                )
                mock_pipeline_cls.return_value.dry_run = False

                exit_code = run_build("sycl", dry_run=False)

                assert exit_code == 1

    def test_run_build_dry_run_mode(self, tmp_path: Path) -> None:
        """run_build should set dry_run on pipeline."""
        with patch("llama_cli.server_runner.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.server_runner.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(success=True)

                run_build("sycl", dry_run=True)

                assert mock_pipeline_cls.return_value.dry_run is True


# =============================================================================
# verify_risks
# =============================================================================


class TestVerifyRisks:
    """Tests for verify_risks function."""

    def test_verify_risks_no_risks(self) -> None:
        """verify_risks should do nothing when no risks detected."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"
        manager.is_risk_acknowledged.return_value = True

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = ["warning_bypass"]

        # Should not raise
        verify_risks(manager, [cfg], acknowledged=True)

    def test_verify_risks_with_risks_acknowledged(self) -> None:
        """verify_risks should acknowledge risks when flag is set."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = []

        with patch("llama_cli.server_runner.detect_risky_operations", return_value=["risky_op"]):
            verify_risks(manager, [cfg], acknowledged=True)

        # Should have added the risk label
        assert "warning_bypass" in cfg.risky_acknowledged

    def test_verify_risks_without_acknowledgment(self) -> None:
        """verify_risks should print warning when risks not acknowledged."""
        manager = MagicMock(spec=ServerManager)
        manager.begin_launch_attempt.return_value = "attempt-123"
        manager.issue_ack_token.return_value = "token-abc"
        manager.is_risk_acknowledged.return_value = False

        cfg = ServerConfig(
            model="/path/to/model.gguf",
            alias="test",
            device="CUDA",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        cfg.risky_acknowledged = []

        with (
            patch("llama_cli.server_runner.detect_risky_operations", return_value=["risky_op"]),
            patch("llama_cli.server_runner.input", return_value="y"),
        ):
            verify_risks(manager, [cfg], acknowledged=False)

        # Should have called acknowledge_risk
        manager.acknowledge_risk.assert_called_once()
