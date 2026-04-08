---
description: Run all pre-commit hooks
agent: python-reviewer
---

# Pre-commit Command

Run all pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

## What This Runs

Pre-commit hooks defined in `.pre-commit-config.yaml`:

- **ruff-check**: Lint checking
- **ruff-format**: Code formatting
- **pyright**: Type checking

## Setup

Install pre-commit hooks:

```bash
uv run pre-commit install
```

This installs hooks that run automatically on `git commit`.

## Running Specific Hooks

```bash
# Run only ruff check
uv run pre-commit run ruff-check --all-files

# Run only formatting
uv run pre-commit run ruff-format --all-files

# Run only type checking
uv run pre-commit run pyright --all-files
```

## Pre-commit vs CI

**Local (pre-commit)**:

- Runs on every commit
- Fast feedback
- Auto-fixes where possible

**CI (build command)**:

- Full pipeline
- All checks enforced
- Required for merge

## Common Issues

### Hook Fails on Clean Code

```bash
# Re-run to ensure all fixes applied
uv run pre-commit run --all-files

# Or skip hooks temporarily (not recommended)
git commit --no-verify
```

### Add New Hook

1. Add to `.pre-commit-config.yaml`
2. Install: `uv run pre-commit install`
3. Run: `uv run pre-commit run --all-files`

## Integration with Editor

Most editors integrate with pre-commit:

- **VS Code**: Pylance + ruff extension
- **Neovim**: null-ls + pre-commit-nvim
- **PyCharm**: Built-in pre-commit support
