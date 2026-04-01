---
description: Run linter and formatter checks
agent: python-reviewer
---

Run ruff linting and formatting checks:

```bash
# Check linting
uv run ruff check .

# Check formatting
uv run ruff format --check .
```

## Quick Fixes

```bash
# Auto-fix fixable lint issues
uv run ruff check --fix .

# Auto-format code
uv run ruff format .

# Fix all issues
uv run ruff check --fix && uv run ruff format .
```

## What This Checks

- **E**: Pycodestyle errors
- **W**: Pycodestyle warnings
- **F**: Pyflakes errors
- **I**: isort (import ordering)
- **UP**: Pyupgrade (modern Python syntax)
- **B**: Flake8-bugbear
- **C4**: Flake8-comprehensions

## Common Issues

### Import Ordering
```python
# WRONG
from llama_manager.config import Config
import os
import sys

# CORRECT
import os
import sys
from llama_manager.config import Config
```

### Type Annotations
```python
# WRONG
from typing import List, Optional

def foo(items: List[str]) -> Optional[str]:
    ...

# CORRECT (Python 3.10+)
def foo(items: list[str]) -> str | None:
    ...
```

### Line Length
```python
# WRONG (101 chars)
def build_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build llama-server command arguments with all options."""

# CORRECT (100 chars max)
def build_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build llama-server command arguments."""
```
