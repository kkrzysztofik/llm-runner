---
description: Run pytest tests with coverage
agent: python-qa
---

Run pytest tests:

```bash
# Basic test run
uv run pytest

# With coverage report
uv run pytest --cov --cov-report=term-missing

# Specific test file
uv run pytest tests/test_config.py

# Specific test function
uv run pytest tests/test_config.py::test_config_default_values

# Verbose output
uv run pytest -v

# With output capture
uv run pytest -s
```

## Test Structure

Tests are organized in `tests/`:

- `test_config.py`: Config, ServerConfig, config builders
- `test_server.py`: Validators, build_server_cmd

## Testing Patterns

### Testing Validators
```python
def test_validate_port_invalid_low(capsys):
    """validate_port should exit with code 1 for port < 1"""
    with pytest.raises(SystemExit) as exc_info:
        validate_port(0, "test_port")
    assert exc_info.value.code == 1
    
    captured = capsys.readouterr()
    assert "error: test_port must be between 1 and 65535" in captured.err
```

### Testing with Subprocess Mock
```python
from unittest.mock import patch, MagicMock

@patch("subprocess.run")
def test_gpu_stats_nvtop(mock_run):
    """GPUStats should parse nvtop JSON output"""
    mock_run.return_value = MagicMock(
        stdout='[{"device_name": "Intel Arc B580", "gpu_util": "45%"}]'
    )
    gpu = GPUStats(device_index=1)
    gpu.update()
    assert gpu.stats["device"] == "Intel Arc B580"
```

### Testing with tmp_path
```python
def test_require_model_not_found(tmp_path):
    """require_model should exit if model doesn't exist"""
    fake_model = tmp_path / "nonexistent.gguf"
    
    with pytest.raises(SystemExit) as exc_info:
        require_model(str(fake_model))
    
    assert exc_info.value.code == 1
```

## Quality Gate

```bash
uv run pytest --cov --cov-report=term-missing
```

Aim for >90% coverage on `llama_manager/` module.
