"""Tests for nvtop GPU stats collection."""

import json
from unittest.mock import MagicMock, patch

from llama_manager.gpu_telemetry import collect_nvtop_stats


class TestCollectNvtopStats:
    """Tests for collect_nvtop_stats function."""

    def test_successful_nvtop_call_returns_gpu_data(self) -> None:
        """Should return GPU data when nvtop succeeds."""
        gpu_data = [
            {
                "device_name": "NVIDIA GeForce RTX 3090",
                "gpu_util": "72%",
                "mem_util": "4.2 GB / 8.0 GB",
                "temp": "65°C",
                "power_draw": "250W",
            }
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(gpu_data),
                stderr="",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "NVIDIA GeForce RTX 3090"
        assert result["gpu_util"] == "72%"
        assert result["mem_util"] == "4.2 GB / 8.0 GB"
        assert result["temp"] == "65°C"
        assert result["power"] == "250W"

    def test_nvtop_device_index_out_of_range_fallback(self) -> None:
        """Should fall back to psutil when device_index is out of range."""
        gpu_data = [{"device_name": "GPU 0"}]  # Only one GPU
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(gpu_data),
                stderr="",
            )
            result = collect_nvtop_stats(5)  # Out of range

        assert result["device"] == "GPU 5"
        assert result["gpu_util"] == "N/A"

    def test_nvtop_timeout_fallback(self) -> None:
        """Should fall back to psutil on timeout."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["nvtop", "-s"], timeout=1)
            result = collect_nvtop_stats(0)

        assert result["device"] == "GPU 0"
        assert result["gpu_util"] == "N/A"

    def test_nvtop_json_decode_error_fallback(self) -> None:
        """Should fall back to psutil on JSON decode error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not json",
                stderr="",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "GPU 0"

    def test_nvtop_return_code_error_fallback(self) -> None:
        """Should fall back to psutil when nvtop returns non-zero."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="nvtop error",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "GPU 0"

    def test_nvtop_non_list_output_fallback(self) -> None:
        """Should fall back to psutil when output is not a list."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"not": "a list"}',
                stderr="",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "GPU 0"

    def test_nvtop_non_dict_gpu_entry_fallback(self) -> None:
        """Should fall back to psutil when GPU entry is not a dict."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='["not a dict"]',
                stderr="",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "GPU 0"

    def test_nvtop_missing_keys_use_defaults(self) -> None:
        """Should use default values when GPU data is missing keys."""
        gpu_data = [{}]  # Empty GPU entry
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(gpu_data),
                stderr="",
            )
            result = collect_nvtop_stats(0)

        assert result["device"] == "Unknown"
        assert result["gpu_util"] == "N/A"
        assert result["mem_util"] == "N/A"
        assert result["temp"] == "N/A"
        assert result["power"] == "N/A"

    def test_fallback_uses_psutil(self) -> None:
        """Fallback should use psutil for CPU and memory."""
        with (
            patch("subprocess.run") as mock_run,
            patch("psutil.cpu_percent", return_value=42.0),
            patch("psutil.virtual_memory") as mock_mem,
        ):
            mock_run.side_effect = RuntimeError("forced fallback")
            mock_mem.return_value = MagicMock(percent=55.0)
            result = collect_nvtop_stats(0)

        assert result["cpu"] == "42%"
        assert result["mem"] == "55%"
