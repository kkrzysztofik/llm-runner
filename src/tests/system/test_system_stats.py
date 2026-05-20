"""Tests for system stats collection."""

from unittest.mock import MagicMock, patch

from llama_manager.system_stats import (
    collect_cpu_percentages,
    collect_memory_usage,
    collect_system_info,
)


class TestCollectCpuPercentages:
    """Tests for collect_cpu_percentages."""

    def test_returns_list_of_floats(self) -> None:
        """Should return a list of float values."""
        with patch("psutil.cpu_percent", return_value=[42.5]):
            result = collect_cpu_percentages()
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_single_value_when_not_percpu(self) -> None:
        """Should return one value when percpu=False."""
        with patch("psutil.cpu_percent", return_value=33.0):
            result = collect_cpu_percentages(percpu=False)
        assert len(result) == 1

    def test_returns_multiple_values_when_percpu(self) -> None:
        """Should return one value per CPU when percpu=True."""
        with patch("psutil.cpu_percent", return_value=[10.0, 20.0, 30.0, 40.0]):
            result = collect_cpu_percentages(percpu=True)
        assert len(result) == 4


class TestCollectMemoryUsage:
    """Tests for collect_memory_usage."""

    def test_returns_mem_and_swp(self) -> None:
        """Should return both mem and swp keys."""
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.swap_memory") as mock_swap,
        ):
            mock_mem.return_value = MagicMock(percent=75.0, used=8e9, total=16e9)
            mock_swap.return_value = MagicMock(percent=25.0, used=2e9, total=8e9)
            result = collect_memory_usage()

        assert "mem" in result
        assert "swp" in result

    def test_mem_has_required_fields(self) -> None:
        """Mem entry should have label, percent, and value_text."""
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.swap_memory") as mock_swap,
        ):
            mock_mem.return_value = MagicMock(percent=50.0, used=4e9, total=8e9)
            mock_swap.return_value = MagicMock(percent=10.0, used=1e9, total=10e9)
            result = collect_memory_usage()

        assert result["mem"]["label"] == "Mem"
        assert isinstance(result["mem"]["percent"], float)
        assert isinstance(result["mem"]["value_text"], str)

    def test_swp_has_required_fields(self) -> None:
        """Swp entry should have label, percent, and value_text."""
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.swap_memory") as mock_swap,
        ):
            mock_mem.return_value = MagicMock(percent=50.0, used=4e9, total=8e9)
            mock_swap.return_value = MagicMock(percent=10.0, used=1e9, total=10e9)
            result = collect_memory_usage()

        assert result["swp"]["label"] == "Swp"
        assert isinstance(result["swp"]["percent"], float)
        assert isinstance(result["swp"]["value_text"], str)


class TestCollectSystemInfo:
    """Tests for collect_system_info."""

    def test_returns_required_keys(self) -> None:
        """Should return tasks, threads, running, load_values, and uptime."""
        with (
            patch("psutil.boot_time", return_value=1000000.0),
            patch("psutil.process_iter") as mock_iter,
            patch("psutil.getloadavg", return_value=(1.0, 2.0, 3.0)),
        ):
            mock_iter.return_value = []
            result = collect_system_info()

        assert "tasks" in result
        assert "threads" in result
        assert "running" in result
        assert "load_values" in result
        assert "uptime" in result

    def test_uptime_format(self) -> None:
        """Uptime should be formatted as HH:MM:SS."""
        with (
            patch("psutil.boot_time", return_value=1000000.0),
            patch("psutil.process_iter") as mock_iter,
            patch("psutil.getloadavg", return_value=(1.0, 2.0, 3.0)),
        ):
            mock_iter.return_value = []
            result = collect_system_info()

        uptime = result["uptime"]
        assert isinstance(uptime, str)
        parts = uptime.split(":")
        assert len(parts) == 3

    def test_load_values_tuple(self) -> None:
        """Load values should be a tuple of 3 floats."""
        with (
            patch("psutil.boot_time", return_value=1000000.0),
            patch("psutil.process_iter") as mock_iter,
            patch("psutil.getloadavg", return_value=(1.5, 2.5, 3.5)),
        ):
            mock_iter.return_value = []
            result = collect_system_info()

        assert result["load_values"] == (1.5, 2.5, 3.5)

    def test_load_values_none_when_not_available(self) -> None:
        """Load values should be None when getloadavg is unavailable."""
        with (
            patch("psutil.boot_time", return_value=1000000.0),
            patch("psutil.process_iter") as mock_iter,
            patch("psutil.getloadavg", side_effect=OSError("no load avg")),
        ):
            mock_iter.return_value = []
            result = collect_system_info()

        assert result["load_values"] is None

    def test_task_caching(self) -> None:
        """Task stats should be cached for 1.5 seconds."""
        # Clear any stale cache from previous tests
        from llama_manager.system_stats import _get_task_stats

        if hasattr(_get_task_stats, "_cache"):
            del _get_task_stats._cache  # type: ignore[attr-defined]
        if hasattr(_get_task_stats, "_cache_ts"):
            del _get_task_stats._cache_ts  # type: ignore[attr-defined]
        call_count = 0

        def _mock_iter(*args: object, **kwargs: object) -> list[object]:
            nonlocal call_count
            call_count += 1
            return []

        with (
            patch("psutil.boot_time", return_value=1000000.0),
            patch("psutil.process_iter", side_effect=_mock_iter),
            patch("psutil.getloadavg", return_value=(1.0, 2.0, 3.0)),
        ):
            # Verify the patch is active
            import psutil  # noqa: PLC0415

            mock_iter = psutil.process_iter  # type: ignore[assignment]
            assert hasattr(mock_iter, "side_effect")

            # First call should trigger process_iter
            collect_system_info()
            assert call_count == 1, f"Expected 1 call after first collect, got {call_count}"
            # Second and third calls should use cache
            collect_system_info()
            collect_system_info()

        # Should only have called process_iter once due to caching
        assert call_count == 1
