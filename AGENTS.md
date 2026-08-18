# AGENTS.md — llm-runner

> Comprehensive reference for AI agents (OpenCode, GitHub Copilot) working in this repository.

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:

- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

---

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

---

## Project Overview

**llm-runner** is a Python TUI application for managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) inference server instances across heterogeneous GPU hardware (Intel Arc SYCL + NVIDIA CUDA). It provides a live Textual terminal dashboard for real-time log streaming, GPU stats, and configuration display.

### Hardware Targets

| Role | Hardware | Backend |
| ------ | ---------- | --------- |
| Summary models (Qwen 3.5-2B / 0.8B) | Intel Arc B580 (GPU 1) | SYCL (SYCL0) |
| Code / reasoning model (Qwen 3.5-35B) | NVIDIA RTX 3090 (GPU 0) | CUDA |

---

## Repository Layout

```bash
llm-runner/
├── src/
│   ├── llama_cli/              # CLI layer (entry points, argument parsing, TUI)
│   │   ├── cli_parser.py       # argparse dispatch for all modes/subcommands
│   │   ├── server_runner.py    # main() + cli_main() — the `llm-runner` entry point
│   │   ├── ui_output.py        # User-facing output helpers (rich Console)
│   │   ├── commands/           # Subcommands: build, doctor, dry-run, profile, setup, smoke
│   │   └── tui/                # Textual TUI (app, controller, viewmodel, modals, components/)
│   ├── llama_manager/          # Core library (no I/O except sys.stderr)
│   │   ├── benchmark/          # llama-bench runner + output parsing
│   │   ├── build_pipeline/     # llama.cpp clone/configure/build orchestration
│   │   ├── common/             # Shared helpers (file ops, security, text, validators)
│   │   ├── config/             # Config, profile registry, persistence, profile cache
│   │   ├── gpu_telemetry/      # GPU stats (nvidia-smi / Level Zero SYCL)
│   │   ├── metadata/           # GGUF metadata extraction
│   │   ├── orchestration/      # Slot lifecycle (manager, launcher, lockfiles)
│   │   ├── probe/              # Smoke probes + git provenance
│   │   ├── reports/            # Failure reports + log rotation
│   │   ├── toolchain/          # Build toolchain detection
│   │   ├── validation/         # Config validators + command builders
│   │   ├── dry_run.py          # Dry-run domain service
│   │   ├── log_buffer.py       # Thread-safe log buffer with autoscroll
│   │   ├── logging_setup.py    # Loguru backend with stdlib bridge
│   │   ├── model_index.py      # Disk cache for scanned GGUF model metadata
│   │   ├── profile_orchestrator.py  # GPU profiling orchestration
│   │   ├── risk_ack.py         # Risk acknowledgement logic (UI-agnostic)
│   │   ├── setup_venv.py       # venv setup for the build environment
│   │   ├── slot_manager.py     # Slot CRUD operations
│   │   ├── slot_profile_store.py  # Persistent store for custom slot profiles
│   │   ├── slot_state.py       # Slot state transitions + runtime liveness
│   │   ├── slot_stats.py       # Slot runtime stats (metrics parsing/persistence)
│   │   ├── smoke.py            # Smoke target resolution + probe execution
│   │   └── system_stats.py     # System statistics collection via psutil
│   └── tests/                  # Unit tests (cli/, config/, smoke/, system/, tui/, ...)
├── pyproject.toml          # Build config, deps, ruff/pyright/pytest settings
├── .python-version         # 3.14
├── .pre-commit-config.yaml # ruff + pyright hooks
└── .github/workflows/ci.yml
```

---

## Development Setup

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install all deps (including dev extras)
uv sync --extra dev

# Activate (optional — prefix commands with `uv run` instead)
source .venv/bin/activate
```

### Key Commands

| Task                | Command |
| ------------------- | ------- |
| Run linter          | `uv run ruff check .` |
| Auto-fix lint       | `uv run ruff check --fix .` |
| Format code         | `uv run ruff format .` |
| Type check          | `uv run pyright` |
| Run tests           | `uv run pytest` |
| Run tests + coverage | `uv run pytest --cov --cov-report=term-missing` |
| Install pre-commit hooks | `uv run pre-commit install` |
| Run all pre-commit hooks | `uv run pre-commit run --all-files` |
| Launch TUI (dry run) | `uv run llm-runner dry-run both` |
| Launch summary model | `uv run llm-runner summary-balanced` |
| Launch all models   | `uv run llm-runner both` |

---

## Architecture Principles

### Separation of Concerns
- `llama_manager/` is a **pure library** — no `argparse`, no `Rich`, no `subprocess` at module level. Functions take typed parameters and return values or mutate state explicitly.
- `llama_cli/` owns all user-facing I/O: argument parsing, TUI rendering, signal handling.
- `tests/` are pure unit tests — no subprocesses, no GPU, no file system side effects beyond what `tmp_path` provides.

### Config Dataclasses

`Config` holds hardware-specific defaults (paths, ports, GPU settings).
`ServerConfig` holds per-instance launch parameters. The profile registry in
`config/profiles.py` maps slot IDs/aliases to profiles; `resolve_profile_config`
translates a profile into a `ServerConfig`.

```python
# Correct pattern — resolve via the profile registry
registry = create_default_profile_registry(config)
sc = resolve_profile_config(registry, "summary-balanced")
cmd = build_server_cmd(sc)
```

### Error Handling
Validation functions in `src/llama_manager/validation/validators.py` (`validate_port`, `validate_ports`, `require_model`, `require_executable`, `validate_server_config`) return `ErrorDetail | None` — `None` means valid; a non-`None` `ErrorDetail` carries `error_code`, `failed_check`, `why_blocked`, and `how_to_fix` (see `src/llama_manager/config/errors.py`). The CLI layer surfaces failures to the user. Tests assert on the returned `ErrorDetail`/`None`, not on `SystemExit`.

---

## Code Conventions

### Python Style

- **Python ≥ 3.14**, type hints on all new functions.
- Line length: 100 chars (ruff enforced).
- Imports: stdlib → third-party → first-party, sorted by ruff/isort.
- Use `|` union syntax (`str | None`) not `Optional[str]` for new code.
- Dataclasses preferred over plain dicts for structured config.

### Naming

- Module-level constants: `UPPER_SNAKE_CASE`
- Functions: `lower_snake_case`
- Classes: `PascalCase`
- Private helpers: `_leading_underscore`

### Type Annotations

- Annotate all function signatures (params + return type).
- Use `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]` (PEP 585).
- `build_server_cmd` returns `list[str]` — keep it that way (subprocess-safe).

---

## Testing Guidelines

- All tests live in `tests/`.
- No subprocess spawning in tests — mock or stub hardware-dependent paths.
- Validators return `ErrorDetail | None` (see the "Error Handling" section) — tests assert on the returned value (`error_code`, `failed_check`, `why_blocked`, `how_to_fix`) or `None`, not on `SystemExit`/`sys.stderr`.
- Use `capsys` only when testing CLI-layer output, not validator return values.
- Tests must pass in CI (ubuntu-latest, Python 3.14) without GPU hardware.
- Name test functions descriptively: `test_<what>_<condition>`.

---

## CI / Pre-commit

All three CI checks must pass before merging:

1. **lint** — `ruff check` + `ruff format --check`
2. **typecheck** — `pyright` (standard mode)
3. **test** — `pytest` with coverage

Additionally:

- **audit** — `uv run pip-audit` for known CVEs in dependencies. CI ignores
  `CVE-2026-3219` and `PYSEC-2026-196` in `pip` because `pip` is only a transitive
  **dev** dependency of `pip-audit` (via `pip-api`), not a runtime dependency of
  llm-runner; revisit when upgrading `pip-audit` / `pip-api`.
- **SonarCloud** — quality gate / SAST on pushes and same-repository pull
  requests when `SONAR_TOKEN` is set (fork PRs are skipped — secrets unavailable).
- **CodeQL** — GitHub Default Setup code scanning; apply
  `.github/codeql/codeql-config.yml` by setting repository property
  `github-codeql-config-file` to that path, then re-saving Default Setup (see
  `.github/codeql/README.md`).

Pre-commit hooks run the same ruff and pyright checks locally on every commit.

### Agent Guardrail: Mandatory Local Gate Before Commit/Push

If you are an AI agent making code changes, you **must** run this exact gate
before any `git commit` or `git push`:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Hard rules for agents:

1. **Do not commit or push if either command fails.**
2. **Fix failures first, then re-run both commands until green.**
3. **Report in your final message that the gate was run and passed.**
4. If the user explicitly instructs you to skip this gate, quote that instruction
   in your final message and call out the risk.

---

## Dependency Security Policy

### CI Dependency Scan

CI runs `uv run pip-audit` on every push and pull request to detect known CVEs
in dependencies. Dependabot opens weekly update PRs for `pip` and
`github-actions`. SAST is covered by CodeQL Default Setup and SonarCloud — not
by a third-party SCA/SAST vendor CLI in this workflow.

### Local Pre-release Check

Before merging or releasing, run:

```bash
uv run pip-audit
```

### Vulnerability Response Cadence

| Severity | Response Target |
| -------- | --------------- |
| Critical | Immediately — patch or pin within 24h |
| High     | Within 1 week |
| Medium   | Within 1 month |
| Low      | Included in routine dependency refresh |

### Routine Dependency Refresh

Quarterly (or before major releases), update all dependencies:

```bash
uv lock --upgrade
uv sync
uv run pip-audit
```

Review `pip-audit` output and update dependencies via `uv add --upgrade-package <pkg>`.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes,
ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** → `ast-grep`
- Need raw speed or **hunting text** → `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

---

## Common Pitfalls

- `ServerConfig.server_bin` defaults to `""` — `build_server_cmd` only falls back to `Config().paths.llama_server_bin_intel` when `server_bin` is `None`, so provide an explicit path in tests to avoid needing the binary on disk.
- `n_gpu_layers` is typed as `int | str` to support `"all"` for CUDA. Keep it that way.
- Do not import from `llama_cli` inside `llama_manager` — the dependency is one-way.
- Comma-form `except A, B:` is PEP 758-legal on Python ≥3.14 and is the form `ruff format` enforces on the `py314` target (it rewrites the parenthesized tuple form back to comma form). Do not flag it as a Python 2 SyntaxError; the project uses it throughout (31 occurrences).
- The TUI uses Textual for rendering and key handling; keep blocking subprocess/log work off the app thread and route UI output through widgets or controller state.

---

## Out of Scope

- Model weights and binary paths are local to the developer's machine — do not hardcode new paths, use `Config` defaults.
- GPU driver setup, SYCL environment variables (`ONEAPI_DEVICE_SELECTOR`), and CUDA library paths are handled by shell wrapper scripts (`run_opencode_models.sh`), not Python.

## Active Technologies
- Python 3.14+ + textual, rich renderables, psutil, pytest, ruff, pyright (001-prd-mvp-spec)
- Local runtime files under resolved runtime dir (`LLM_RUNNER_RUNTIME_DIR` else `$XDG_RUNTIME_DIR/llm-runner`) for lockfiles + JSON artifacts (001-prd-mvp-spec)
- Python 3.14+ + stdlib (`subprocess`, `pathlib`, `venv`, `json`, `dataclasses`, `threading`), textual, rich renderables, psutil (002-build-setup)
- Local filesystem only (source tree + XDG cache/state/data directories) (002-build-setup)

## Recent Changes
- 001-prd-mvp-spec: Added Python 3.14+ + textual, rich renderables, psutil, pytest, ruff, pyright

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan:
specs/001-m4-op-hardening/plan.md
<!-- SPECKIT END -->

## Learned User Preferences

- Remove smoke testing and llama.cpp GPU profiling from the **TUI only**; keep `llm-runner smoke` and `llm-runner profile` CLI commands and backend libraries unless the user explicitly requests full removal.
- When removing smoke or profiling, confirm TUI-only vs CLI/libraries scope before deleting shared modules.
- Do not edit attached Speckit/plan files during “implement the plan” work—change code and tests only.
- Ask scope-clarifying questions before large removals (smoke/profile, profiling cache) rather than assuming full deletion.
- Keep llama.cpp build output routed through captured/buffered UI (build wizard `RichLog` with timestamps/colour, `markup=False`; result/errors as Rich `Text` with escaped brackets, not markup strings); avoid flashing raw Loguru stderr over Textual; show live progress and clear build-failure handling.
- System-health datetime row: use Textual `Digits` for block digital time—not analog wall clock or `textual-hires-canvas`.
- Datetime header layout: `LLM_RUNNER_LOGO` (wordmark + robot) on the left; date and digital time on the far right with a flex spacer—do not put date on the left beside the logo.
- Logo wordmark block letters must read **LLM**, not LIM—verify spacing in `_LLM_BLOCK` / `LLM_RUNNER_LOGO` when refining the mascot.
- TUI header logo: R2-D2-inspired mascot, horizontal rainbow on the LLM wordmark, no separate "runner" label under the robot; robot height should match the wordmark block.
- Run profile create/edit: keep port, ubatch size, GPU layers, threads, and server binary in a collapsed **Advanced** section by default.
- When adding profile/server fields, align with `run_opencode_models.sh`; expose global defaults in the Config modal; use Select/Checkbox for enumerated values (e.g. Config modal `source flavor` from `SOURCE_FLAVOR_DEFAULTS`: `upstream`, `beellama`) and Input for freeform or wide numeric ranges; keep `git_remote`/`git_branch` as Inputs for manual overrides.
- Textual `Select` in profile/config modals: style `SelectCurrent` via `profile-select`/`config-select` so the chosen value renders inside the control at Input height; do not add summary-label workarounds beside Select widgets.

## Learned Workspace Facts

- For build wizard binary display and readiness badges, run `llama-server --version` and parse the `version:` line; do not substitute git `source_head_sha` as the binary version. Prefer `git_commit_sha` from `build-artifact.json`; else take the last parenthesized hex (7–40 chars), not the first (build number). Compare up to 8 chars to source HEAD; missing or mismatched binary commit ⇒ needs_update.
- Build wizard step 1: mount immediately with parallel SYCL/CUDA `get_build_status` on `@work(thread=True)`; show Loading… until `call_from_thread` applies results on `STEP_SELECT` only—never call `BackendStatusCard.set_status` after leaving step 1 (avoids `NoMatches` on detached `.build-backend-header`).
- TUI builds: wrap pipeline work in `suppress_build_pipeline_stderr_for_tui()`; Stop sets `build_cancel_event` and kills the active stage via `run_command_with_cancel` (process-group termination, not dismiss-only).
- When `BuildConfig.jobs` is unset, `cmake --build` uses `-j` from `os.cpu_count()`.
- SYCL `llama-server --version` probes need oneAPI via `get_build_env_cmd()` in `build_pipeline/utils.py` (sources `/opt/intel/oneapi/setvars.sh` when present).
- Build wizard “Artifact” means provenance JSON at `builds_dir/{sycl|cuda}/build-artifact.json`; untracked binaries fall back to `llama_server_bin_intel` / `llama_server_bin_nvidia` on Config.
- `SOURCE_FLAVOR_DEFAULTS` in `build_pipeline/models.py` maps `upstream` / `beellama` to git remote URLs and branches; Config modal source-flavor Select and build orchestration resolve URLs from it. TUI build wizard passes empty `git_remote_url` in `config_overrides` to keep flavor-resolved URLs—`_merge_config_overrides` must seed all base `BuildConfig` fields before applying non-empty overrides.
- Default runtime binaries live under `llama_cpp_root`: SYCL at `build/bin/llama-server`, CUDA at `build_cuda/bin/llama-server`; provenance JSON lives under XDG state `builds_dir`.
- `LLM_RUNNER_LOGO` and `DigitalClockWidget` live in `digital_clock.py`; `DateTimeWidget` mounts logo left (rainbow `_LLM_BLOCK`, R2-D2 `_ROBOT_BLOCK`, `markup=True`) with date + `Digits` on the right; clock ticks on 1s `set_interval` (not the dashboard 250ms loop); date uses `%a %Y-%m-%d`.
- TUI bottom bar: Textual built-in `Footer` (`show_command_palette=False`) in `textual_app.py`, with `check_action` + `refresh_bindings` for mode-aware bindings (replaced `CommandMenu`).
- Bare `llm-runner` / `parse_args([])` launches standalone TUI via `_default_tui_namespace()` in `cli_parser.py`; `_normalize_main_args` must not strip bare run-group names as the program name.
- Run profile Advanced fields live in a collapsed Textual `Collapsible` in `run_profile_modal.py` (pattern/CSS from build wizard via `.profile-advanced-options`).
