# Quickstart

## 1) Install dependencies

```bash
uv sync --extra dev
```

Optional:

```bash
source .venv/bin/activate
```

## 2) Verify the environment

```bash
uv run llm-runner setup check
uv run llm-runner doctor check
```

These commands confirm the toolchain, venv, locks, reports, and cached profile
state.

## 3) Preview the launch commands

```bash
uv run llm-runner dry-run both
```

You should see the resolved argv, ports, model paths, redacted environment,
and validation output for both slots. Dry-run is the recommended way to
validate profile resolution before launching — it shows exactly what
`llama-server` commands will be executed, with all tuning parameters resolved
from the built-in profile registry.

## 4) Launch a model pair

```bash
uv run llm-runner both
```

Default mapping:

- `summary-balanced` → Intel SYCL / GPU 1 / port `8080`
- `qwen35` → NVIDIA CUDA / GPU 0 / port `8081`

Launch modes are data-driven: profiles and run groups are defined in
`config_builder.py` and resolved at runtime. The `both` group launches two
profiles simultaneously; single-profile modes (`summary-balanced`,
`summary-fast`, `qwen35`) launch one server in the foreground.

## 5) Check the running servers

```bash
uv run llm-runner smoke both
```

If you only want one slot:

```bash
uv run llm-runner smoke slot summary-balanced
```

## 6) Build or repair the environment

```bash
uv run llm-runner build sycl --dry-run
uv run llm-runner build sycl
uv run llm-runner setup venv
uv run llm-runner doctor repair --dry-run
```

By default, builds clone or reuse `llama.cpp` at
`$XDG_CACHE_HOME/llm-runner/llama.cpp`. Use `LLAMA_CPP_ROOT` or
`--source-dir` to point at an explicit checkout.

## 7) Optional profiling

```bash
uv run llm-runner profile summary-balanced balanced
```

## 8) Optional TUI

```bash
uv run python src/run_models_tui.py both
```

## What to expect

- Logs stream live in the TUI or CLI output
- Sensitive values are redacted automatically
- Runtime files live under XDG/`LLM_RUNNER_RUNTIME_DIR`
- `llama.cpp` source and binaries live under `$XDG_CACHE_HOME/llm-runner/llama.cpp`
- Build provenance goes to `$XDG_STATE_HOME/llm-runner/builds`
- Failure reports go to `$XDG_DATA_HOME/llm-runner/reports`

## If something fails

Run these three commands first:

```bash
uv run llm-runner setup check
uv run llm-runner doctor check
uv run llm-runner dry-run both
```

Then read the structured `error_code`, `failed_check`, `why_blocked`, and
`how_to_fix` fields in the output.
