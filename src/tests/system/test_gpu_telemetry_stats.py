"""Tests for GPUStats, collect_gpu_stats, and helpers in gpu_telemetry.stats."""

from unittest.mock import MagicMock, patch

import pytest

from llama_manager.gpu_telemetry.common import GpuTelemetrySelector
from llama_manager.gpu_telemetry.stats import (
    GPUStats,
    _is_real_value,
    collect_gpu_stats,
    collector_for_config,
    get_gpu_identifier,
    make_gpu_collector,
    selector_for_config,
    selectors_for_config,
)


class TestGPUStatsUpdate:
    """Tests for GPUStats.update() - util delta parsing and logging."""

    def test_update_skips_when_interval_not_elapsed(self) -> None:
        """Should skip collector call when update_interval not elapsed."""
        collector = MagicMock(return_value={"gpu_util": "50%"})
        stats = GPUStats(device_index=0, collector=collector)
        stats.last_update = 0.0
        stats.update_interval = 10.0

        stats.update()
        stats.update()

        assert collector.call_count == 1

    def test_update_parses_string_util_with_percent(self) -> None:
        """Should parse gpu_util string like '50%' to float."""
        collector = MagicMock(return_value={"gpu_util": "50%"})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 45.0

        stats.update()

        assert stats._prev_gpu_util == 50.0

    def test_update_handles_invalid_percent_string(self) -> None:
        """Should default to 0.0 when gpu_util string can't be parsed."""
        collector = MagicMock(return_value={"gpu_util": "not_a_number"})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 50.0

        stats.update()

        assert stats._prev_gpu_util is None

    def test_update_parses_numeric_util(self) -> None:
        """Should handle gpu_util as int/float."""
        collector = MagicMock(return_value={"gpu_util": 75.5})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 60.0

        stats.update()

        assert stats._prev_gpu_util == 75.5

    def test_update_logs_util_delta_when_significant(self) -> None:
        """Should log debug when gpu util delta > 5%."""
        collector = MagicMock(return_value={"gpu_util": "80%"})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 50.0

        with patch("llama_manager.gpu_telemetry.stats.logger") as mock_logger:
            stats.update()
            mock_logger.debug.assert_called()

    def test_update_logs_util_delta_with_int_values(self) -> None:
        """Should log debug when int gpu util delta > 5%."""
        collector = MagicMock(return_value={"gpu_util": 90})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 50.0

        with patch("llama_manager.gpu_telemetry.stats.logger") as mock_logger:
            stats.update()
            mock_logger.debug.assert_called()

    def test_update_does_not_log_small_delta(self) -> None:
        """Should not log when delta <= 5%."""
        collector = MagicMock(return_value={"gpu_util": "53%"})
        stats = GPUStats(device_index=0, collector=collector)
        stats._prev_gpu_util = 50.0

        with patch("llama_manager.gpu_telemetry.stats.logger") as mock_logger:
            stats.update()
            mock_logger.debug.assert_not_called()


class TestGPUStatsProperties:
    """Tests for gpu_util and memory_util properties."""

    def test_gpu_util_returns_cached_value(self) -> None:
        """gpu_util property should return cached stats."""
        collector = MagicMock(return_value={"gpu_util": "75%", "mem_util": "50%"})
        stats = GPUStats(device_index=0, collector=collector)

        result = stats.gpu_util

        assert result == "75%"

    def test_memory_util_returns_cached_value(self) -> None:
        """memory_util property should return cached stats."""
        collector = MagicMock(return_value={"gpu_util": "50%", "mem_util": "80%"})
        stats = GPUStats(device_index=0, collector=collector)

        result = stats.memory_util

        assert result == "80%"

    def test_gpu_util_returns_na_when_missing(self) -> None:
        """gpu_util should return 'N/A' when not in stats."""
        collector = MagicMock(return_value={})
        stats = GPUStats(device_index=0, collector=collector)

        assert stats.gpu_util == "N/A"

    def test_memory_util_returns_na_when_missing(self) -> None:
        """memory_util should return 'N/A' when not in stats."""
        collector = MagicMock(return_value={})
        stats = GPUStats(device_index=0, collector=collector)

        assert stats.memory_util == "N/A"


class TestGPUStatsFormat:
    """Tests for format_stats_text()."""

    def test_format_with_gpu_util(self) -> None:
        """Should include GPU line when gpu_util is present."""
        collector = MagicMock(return_value={"gpu_util": "75%", "mem_util": "50%"})
        stats = GPUStats(device_index=0, collector=collector)

        text = stats.format_stats_text()

        assert "GPU: 75%" in text
        assert "Mem: 50%" in text

    def test_format_without_gpu_util_falls_to_cpu(self) -> None:
        """Should include CPU line when gpu_util is N/A."""
        collector = MagicMock(return_value={"cpu": "42%", "mem": "55%"})
        stats = GPUStats(device_index=0, collector=collector)

        text = stats.format_stats_text()

        assert "CPU: 42%" in text
        assert "Mem: 55%" in text

    def test_format_includes_temp_when_present(self) -> None:
        """Should include temp line when temp is present."""
        collector = MagicMock(return_value={"gpu_util": "75%", "mem_util": "50%", "temp": "65C"})
        stats = GPUStats(device_index=0, collector=collector)

        text = stats.format_stats_text()

        assert "Temp: 65C" in text

    def test_format_includes_power_when_present(self) -> None:
        """Should include power line when power key exists."""
        collector = MagicMock(return_value={"gpu_util": "75%", "mem_util": "50%", "power": "250W"})
        stats = GPUStats(device_index=0, collector=collector)

        text = stats.format_stats_text()

        assert "Power: 250W" in text

    def test_format_excludes_power_when_na(self) -> None:
        """Should not include power line when power is 'N/A'."""
        collector = MagicMock(return_value={"gpu_util": "75%", "mem_util": "50%", "power": "N/A"})
        stats = GPUStats(device_index=0, collector=collector)

        text = stats.format_stats_text()

        assert "Power:" not in text


class TestGetGpuIdentifier:
    """Tests for get_gpu_identifier()."""

    def test_cuda_backend_returns_identifier(self) -> None:
        """Should return identifier for cuda backend."""
        collector = MagicMock(return_value=[{"name": "RTX 3090", "index": 0}])
        result = get_gpu_identifier("cuda", gpu_collector=collector)

        assert isinstance(result, str)
        assert "nvidia" in result.lower()

    def test_sycl_backend_returns_identifier(self) -> None:
        """Should return identifier for sycl backend."""
        collector = MagicMock(return_value=[{"name": "Intel Arc", "index": 0}])
        result = get_gpu_identifier("sycl", gpu_collector=collector)

        assert isinstance(result, str)
        assert "intel" in result.lower()

    def test_unsupported_backend_raises(self) -> None:
        """Should raise ValueError for unsupported backend."""
        with pytest.raises(ValueError, match="unsupported backend"):
            get_gpu_identifier("rocm")

    def test_empty_device_list_raises(self) -> None:
        """Should raise IndexError when collector returns empty list."""
        collector = MagicMock(return_value=[])
        with pytest.raises(IndexError, match="empty device list"):
            get_gpu_identifier("cuda", gpu_collector=collector)


class TestIsRealValue:
    """Tests for _is_real_value()."""

    def test_none_is_not_real(self) -> None:
        assert _is_real_value(None) is False

    def test_na_is_not_real(self) -> None:
        assert _is_real_value("N/A") is False

    def test_empty_string_is_not_real(self) -> None:
        assert _is_real_value("") is False

    def test_valid_string_is_real(self) -> None:
        assert _is_real_value("75%") is True

    def test_zero_is_real(self) -> None:
        assert _is_real_value(0) is True

    def test_zero_float_is_real(self) -> None:
        assert _is_real_value(0.0) is True


class TestCollectGpuStats:
    """Tests for collect_gpu_stats() merge and early-break logic."""

    def test_sycl_uses_level_zero_first(self) -> None:
        """SYCL selector should try L0 collector first."""
        selector = GpuTelemetrySelector(backend="sycl", ordinal=0)
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_level_zero_stats",
                return_value={"gpu_util": "50%", "mem_util": "30%"},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                return_value={"gpu_util": "90%"},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_xpu_smi_stats",
                return_value={"gpu_util": "80%"},
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["gpu_util"] == "50%"

    def test_cuda_uses_nvidia_smi_first(self) -> None:
        """CUDA selector should try nvidia-smi collector first."""
        selector = GpuTelemetrySelector(backend="cuda", ordinal=0)
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvidia_smi_stats",
                return_value={"gpu_util": "75%", "mem_util": "40%"},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                return_value={"gpu_util": "90%"},
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["gpu_util"] == "75%"

    def test_merge_fills_missing_metrics(self) -> None:
        """Later collectors should fill missing keys from earlier ones."""
        selector = GpuTelemetrySelector(backend="sycl", ordinal=0)

        def mock_l0(_sel):
            # L0 provides only mem_util (no gpu_util so collection continues)
            return {"mem_util": "30%"}

        def mock_nvtop(_sel):
            # nvtop provides gpu_util + temp
            return {"gpu_util": "50%", "temp": "65C"}

        def mock_xpu(_sel):
            return {}

        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_level_zero_stats",
                mock_l0,
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                mock_nvtop,
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_xpu_smi_stats",
                mock_xpu,
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["gpu_util"] == "50%"
        assert result["mem_util"] == "30%"
        assert result["temp"] == "65C"

    def test_merge_skips_real_value_in_earlier(self) -> None:
        """Later collectors should not override real values from earlier."""
        selector = GpuTelemetrySelector(backend="sycl", ordinal=0)
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_level_zero_stats",
                return_value={"gpu_util": "50%", "mem_util": "30%"},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                return_value={"gpu_util": "90%", "mem_util": "80%"},
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["gpu_util"] == "50%"
        assert result["mem_util"] == "30%"

    def test_early_break_on_real_gpu_util(self) -> None:
        """Should stop collecting once gpu_util has a real value."""
        selector = GpuTelemetrySelector(backend="sycl", ordinal=0)
        mock_nvtop = MagicMock(return_value={"mem_util": "30%"})
        mock_xpu = MagicMock(return_value={"gpu_util": "80%"})
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_level_zero_stats",
                return_value={"gpu_util": "50%"},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                mock_nvtop,
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_xpu_smi_stats",
                mock_xpu,
            ),
        ):
            collect_gpu_stats(selector)

        # L0 returns gpu_util=50%, so break after L0 — nvtop and xpu skipped
        mock_nvtop.assert_not_called()
        mock_xpu.assert_not_called()

    def test_falls_back_to_psutil_when_no_collectors_provide(self) -> None:
        """Should fall back to psutil_only_collector when all collectors fail."""
        selector = GpuTelemetrySelector(backend="cuda", ordinal=3)
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvidia_smi_stats",
                return_value=None,
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                return_value=None,
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["device"] == "GPU 3"
        assert result["gpu_util"] == "N/A"

    def test_empty_merged_dict_falls_back(self) -> None:
        """Should fall back when collectors return empty dicts."""
        selector = GpuTelemetrySelector(backend="cuda", ordinal=1)
        with (
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvidia_smi_stats",
                return_value={},
            ),
            patch(
                "llama_manager.gpu_telemetry.stats.collect_nvtop_stats_for_selector",
                return_value={},
            ),
        ):
            result = collect_gpu_stats(selector)

        assert result["device"] == "GPU 1"


class TestMakeGpuCollector:
    """Tests for make_gpu_collector()."""

    def test_returns_callable_bound_to_selector(self) -> None:
        """Should return a zero-argument callable."""
        selector = GpuTelemetrySelector(backend="cuda", ordinal=0)
        collector = make_gpu_collector(selector)

        assert callable(collector)
        result = collector()
        assert isinstance(result, dict)


class TestSelectorForConfig:
    """Tests for selector_for_config()."""

    def test_parses_cuda_device_string(self) -> None:
        """Should parse CUDA device string."""
        cfg = MagicMock()
        cfg.device = "cuda:0"
        cfg.main_gpu = 0

        selector = selector_for_config(cfg)

        assert selector.backend == "cuda"
        assert selector.ordinal == 0

    def test_parses_sycl_device_string(self) -> None:
        """Should parse SYCL device string."""
        cfg = MagicMock()
        cfg.device = "sycl:0"
        cfg.main_gpu = 0

        selector = selector_for_config(cfg)

        assert selector.backend == "sycl"
        assert selector.ordinal == 0

    def test_fallback_to_cuda_when_no_device(self) -> None:
        """Should default to cuda when device is empty."""
        cfg = MagicMock()
        cfg.device = ""
        cfg.main_gpu = 2

        selector = selector_for_config(cfg)

        assert selector.backend == "cuda"
        assert selector.ordinal == 2

    def test_selectors_for_config_returns_all_cuda_ordinals_with_main_first(self) -> None:
        """Multi-GPU CUDA configs should collect telemetry for every listed ordinal."""
        cfg = MagicMock()
        cfg.device = "CUDA:0,1"
        cfg.main_gpu = 1

        selectors = selectors_for_config(cfg)

        assert [selector.ordinal for selector in selectors] == [1, 0]
        assert [selector.backend for selector in selectors] == ["cuda", "cuda"]

    def test_collector_for_config_returns_devices_for_multi_cuda_config(self) -> None:
        """Multi-GPU collector should expose per-device snapshots for the panel."""
        cfg = MagicMock()
        cfg.device = "CUDA:0,1"
        cfg.main_gpu = 0

        def fake_collect(selector: GpuTelemetrySelector) -> dict[str, str]:
            return {
                "device": f"GPU {selector.ordinal}",
                "gpu_util": f"{selector.ordinal}0%",
                "mem_util": f"{selector.ordinal}5%",
            }

        with patch("llama_manager.gpu_telemetry.stats.collect_gpu_stats", fake_collect):
            result = collector_for_config(cfg)()

        assert result["device"] == "CUDA:0 GPU 0"
        assert [device["device"] for device in result["devices"]] == [
            "CUDA:0 GPU 0",
            "CUDA:1 GPU 1",
        ]
