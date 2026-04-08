# llm-runner — Copilot Instructions

This is a Python 3.12 TUI project managing llama.cpp inference servers across
Intel SYCL and NVIDIA CUDA GPUs. Managed with `uv`. See
[AGENTS.md](../AGENTS.md) for full reference.

## Quick Rules

- **Library vs CLI**: `llama_manager/` is pure library (no argparse, no Rich).
  `llama_cli/` owns all I/O.
- **Config pattern**: use `Config` + `ServerConfig` dataclasses. Factory functions
  in `config_builder.py`.
- **Type hints**: always annotate function signatures. Use `list[str]` not
  `List[str]`.
- **Line length**: 100 chars. Formatter: `ruff format`.
- **Imports**: stdlib → third-party → first-party (enforced by ruff/isort).

## Dev Commands

```bash
uv sync --extra dev          # install everything
uv run ruff check --fix .    # lint
uv run ruff format .         # format
uv run pyright               # type check
uv run pytest                # tests
```

## Tests

- Pure unit tests only — no subprocess, no GPU.
- Use `capsys` + `pytest.raises(SystemExit)` for validator edge cases.
- Test files: `tests/test_config.py`, `tests/test_server.py`.
