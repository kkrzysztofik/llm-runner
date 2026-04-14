"""Tests for GPU collectors module."""

from unittest.mock import MagicMock, patch

import psutil

from llama_cli.gpu_collectors import _get_cpu_percent, _get_memory_percent


class TestGetCpuPercent:
    """Tests for _get_cpu_percent function."""

    def test_get_cpu_percent_success(self):
        """Test successful CPU percent retrieval."""
        with patch("psutil.cpu_percent", return_value=45.7) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 45.7
            assert isinstance(result, float)
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_access_denied(self):
        """Test graceful handling of AccessDenied exception."""
        with patch("psutil.cpu_percent", side_effect=psutil.AccessDenied()) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_no_such_process(self):
        """Test graceful handling of NoSuchProcess exception."""
        with patch("psutil.cpu_percent", side_effect=psutil.NoSuchProcess(pid=123)) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_general_exception(self):
        """Test graceful handling of general Exception."""
        with patch("psutil.cpu_percent", side_effect=Exception("test error")) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)


class TestGetMemoryPercent:
    """Tests for _get_memory_percent function."""

    def test_get_memory_percent_success(self):
        """Test successful memory percent retrieval."""
        mock_mem = MagicMock()
        mock_mem.percent = 62.3
        with patch("psutil.virtual_memory", return_value=mock_mem) as mock_mem_func:
            result = _get_memory_percent()
            assert result == 62.3
            assert isinstance(result, float)
            mock_mem_func.assert_called_once()

    def test_get_memory_percent_access_denied(self):
        """Test graceful handling of AccessDenied exception."""
        with patch("psutil.virtual_memory", side_effect=psutil.AccessDenied()) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()

    def test_get_memory_percent_no_such_process(self):
        """Test graceful handling of NoSuchProcess exception."""
        with patch("psutil.virtual_memory", side_effect=psutil.NoSuchProcess(pid=123)) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()

    def test_get_memory_percent_general_exception(self):
        """Test graceful handling of general Exception."""
        with patch("psutil.virtual_memory", side_effect=Exception("test error")) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()
