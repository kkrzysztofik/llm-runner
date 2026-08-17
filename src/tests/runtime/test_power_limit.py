"""Unit tests for orchestration.power_limit — mocked subprocess, no GPU."""

from __future__ import annotations

from unittest.mock import Mock, patch

from llama_manager.orchestration.power_limit import (
    apply_nvidia_power_limit,
    cuda_ordinals,
)


class TestCudaOrdinals:
    def test_parses_multi_device(self) -> None:
        assert cuda_ordinals("CUDA0,CUDA1") == [0, 1]

    def test_parses_dotted_cuda(self) -> None:
        assert cuda_ordinals("cuda:0") == [0]

    def test_empty_device_defaults_to_zero(self) -> None:
        assert cuda_ordinals("") == [0]

    def test_auto_defaults_to_zero(self) -> None:
        assert cuda_ordinals("auto") == [0]

    def test_parses_nonzero_indices(self) -> None:
        assert cuda_ordinals("cuda:1,2") == [1, 2]

    def test_sycl_returns_empty(self) -> None:
        assert cuda_ordinals("SYCL0") == []

    def test_garbage_returns_empty(self) -> None:
        assert cuda_ordinals("not-a-device") == []


class TestApplyNvidiaPowerLimit:
    def test_zero_watts_is_noop(self) -> None:
        warn = Mock()
        with patch("subprocess.run") as run:
            apply_nvidia_power_limit("cuda:0", 0, warn)
        run.assert_not_called()
        warn.assert_not_called()

    def test_applies_to_each_ordinal(self) -> None:
        warn = Mock()
        with patch("subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")) as run:
            apply_nvidia_power_limit("CUDA0,CUDA1", 290, warn)
        assert run.call_count == 2
        assert run.call_args_list[0].args[0] == [
            "sudo",
            "-n",
            "nvidia-smi",
            "-i",
            "0",
            "-pl",
            "290",
        ]
        assert run.call_args_list[1].args[0] == [
            "sudo",
            "-n",
            "nvidia-smi",
            "-i",
            "1",
            "-pl",
            "290",
        ]
        warn.assert_not_called()

    def test_nonzero_exit_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", return_value=Mock(returncode=1, stdout="", stderr="denied")):
            apply_nvidia_power_limit("CUDA0,CUDA1", 290, warn)
        assert warn.call_count == 2

    def test_oserror_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", side_effect=OSError("no sudo")):
            apply_nvidia_power_limit("cuda:0", 290, warn)
        warn.assert_called_once()

    def test_timeout_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", side_effect=TimeoutError("hung")):
            apply_nvidia_power_limit("cuda:0", 290, warn)
        warn.assert_called_once()

    def test_sycl_device_noop(self) -> None:
        warn = Mock()
        with patch("subprocess.run") as run:
            apply_nvidia_power_limit("SYCL0", 290, warn)
        run.assert_not_called()
        warn.assert_not_called()
