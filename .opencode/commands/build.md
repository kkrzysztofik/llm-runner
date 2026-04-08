---
description: Run full CI pipeline - lint, typecheck, and test
agent: build
---

# Build Command

Run the full CI pipeline:

```bash
# Lint and format
uv run ruff check .
uv run ruff format --check .

# Type check
uv run pyright

# Run tests
uv run pytest
```

Show output from each step. All must pass.

## What This Checks

- **Lint**: Pycodestyle errors, pyflakes, isort, flake8-bugbear, flake8-comprehensions
- **Format**: Code formatting (100 char line length)
- **Type checking**: Pyright static type analysis
- **Tests**: All pytest tests with coverage

## Quick Fixes

```bash
# Auto-fix lint issues
uv run ruff check --fix .

# Auto-format code
uv run ruff format .

# Fix all issues
uv run ruff check --fix && uv run ruff format .
```

## Pre-commit Integration

To run on every commit:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
