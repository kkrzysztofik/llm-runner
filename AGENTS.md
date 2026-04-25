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

**llm-runner** is a Python TUI application for managing multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) inference server instances across heterogeneous GPU hardware (Intel Arc SYCL + NVIDIA CUDA). It provides a live Rich-based terminal dashboard for real-time log streaming, GPU stats, and configuration display.

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
│   │   ├── cli_parser.py       # argparse modes: both, summary-balanced, summary-fast, qwen35, dry-run
│   │   ├── server_runner.py    # main() + cli_main() entry point
│   │   ├── tui_app.py          # Rich Live TUI (signal handling, layout)
│   │   └── dry_run.py          # Print resolved commands without launching
│   ├── llama_manager/          # Core library (no I/O except sys.stderr)
│   │   ├── config.py           # Config + ServerConfig dataclasses
│   │   ├── config_builder.py   # Factory functions: create_*_cfg()
│   │   ├── server.py           # build_server_cmd() + validators
│   │   ├── process_manager.py  # ServerManager — subprocess lifecycle
│   │   ├── gpu_stats.py        # GPUStats (nvidia-smi / sycl-ls parsing)
│   │   ├── log_buffer.py       # Thread-safe real-time log streaming
│   │   └── colors.py           # Terminal colour constants
│   ├── tests/
│   │   ├── test_config.py      # Config, ServerConfig, config builders
│   │   └── test_server.py      # Validators, build_server_cmd
│   ├── run_models_tui.py       # TUI entry point
│   └── run_opencode_models.py  # CLI entry point
├── pyproject.toml          # Build config, deps, ruff/pyright/pytest settings
├── .python-version         # 3.12
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
`ServerConfig` holds per-instance launch parameters. The factory functions in
`config_builder.py` translate a `Config` into a `ServerConfig` for a given mode.

```python
# Correct pattern — only override what you need
sc = create_summary_balanced_cfg(port=8080, threads=4)
cmd = build_server_cmd(sc)
```

### Error Handling
Validation functions (`validate_port`, `validate_threads`, `validate_ports`) call `sys.exit(1)` on failure after printing to `sys.stderr`. This is intentional — they are user-input guards at the CLI boundary. Do not add try/except around them in test code; use `pytest.raises(SystemExit)` instead.

---

## Code Conventions

### Python Style

- **Python ≥ 3.12**, type hints on all new functions.
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
- Use `capsys` fixture to capture and assert on `sys.stderr` output from validators.
- Use `pytest.raises(SystemExit)` for testing validator exit paths; assert `exc.value.code == 1`.
- Tests must pass in CI (ubuntu-latest, Python 3.12) without GPU hardware.
- Name test functions descriptively: `test_<what>_<condition>`.

---

## CI / Pre-commit

All three CI checks must pass before merging:

1. **lint** — `ruff check` + `ruff format --check`
2. **typecheck** — `pyright` (standard mode)
3. **test** — `pytest` with coverage

Additionally, an audit job runs `uv run pip-audit` to check for known CVEs in
dependencies.

Pre-commit hooks run the same ruff and pyright checks locally on every commit.

---

## Dependency Security Policy

### CI Dependency Scan

CI automatically runs `uv run pip-audit` on every push and pull request to detect
known CVEs in dependencies. The audit job does not block merging but provides
visibility into potential vulnerabilities.

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

## Issue Tracking (br / beads-rust)

Issues for this project are tracked with **[br](https://github.com/Dicklesworthstone/beads_rust)** — a local-first, non-invasive issue tracker storing data in SQLite with JSONL export for git. Issues live in `.beads/` at the root of the repo.

**`br` is non-invasive — it NEVER executes git commands automatically.** You are always responsible for staging and committing `.beads/`.

**Never touch `.beads/` directly** — always use the `br` CLI.

### br CLI Commands

| Task | Command |
| ------ | --------- |
| See actionable (unblocked) work | `RUST_LOG=error br ready` |
| List all open issues | `RUST_LOG=error br list --status open` |
| Create an issue | `br create "Title" --type task --priority 2` |
| Quick-capture (returns ID only) | `br q "Title"` |
| Show issue details | `br show br-abc123` |
| Mark in progress | `br update br-abc123 --status in_progress` |
| Close with reason | `br close br-abc123 --reason "Fixed in commit abc"` |
| Close multiple at once | `br close br-abc123 br-def456 --reason "Done"` |
| Add dependency (A blocked by B) | `br dep add br-A br-B` |
| Export to JSONL for git | `br sync --flush-only` |

### Workflow Pattern

1. **Pick work** — `RUST_LOG=error br ready` shows unblocked, open issues sorted
by priority
2. **Claim** — `br update <id> --status in_progress`
3. **Implement** the task
4. **Close** — `br close <id> --reason "..."` (be specific — reference the
commit or file changed)
5. **Sync + commit** — see session end checklist below

`br ready` only surfaces issues that have no open blockers. Use `br dep add
<child> <parent>` to declare that one issue must wait for another.

### Issue Types and Priority

**Types:** `task`, `bug`, `feature`, `epic`, `question`, `docs`

**Priority** (use numbers, not words):

| Number | Meaning                 |
| ------ | ----------------------- |
| 0      | Critical / blocking everything |
| 1      | High — current sprint   |
| 2      | Normal (default)        |
| 3      | Low                     |
| 4      | Backlog                 |

### Agent Usage — Always Use `--json`

**CRITICAL:** Always pass `--json` when parsing `br` output programmatically.
Plain output format depends on terminal state and may include ANSI codes that
break parsing.

```bash
# CORRECT — stable, parseable output
RUST_LOG=error br ready --json
RUST_LOG=error br list --status open --json
RUST_LOG=error br show br-abc123 --json

# WRONG — output varies with terminal state
br ready | head -1
```bash

`RUST_LOG=error` suppresses internal Rust dependency logs while keeping clean
stdout. **Always include it** in automated or agent-driven commands.

### Session End Checklist

Before ending any work session, run in order:

```bash
git status                  # Check what changed
git add <changed-files>     # Stage code changes first
br sync --flush-only        # Export issues to JSONL
git add .beads/             # Stage beads changes
git commit -m "..."         # Commit code + beads together
git push
```bash

### Best Practices

- Run `RUST_LOG=error br ready` at the start of each session to find available
work
- Use `br q "Title"` for fast capture when you discover tasks during
implementation — fill in details later
- Set `--type` and `--priority` explicitly on `br create`; defaults are `task` /
`2`
- Always sync and commit `.beads/` at session end — stale JSONL means lost issue
state in git history

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

- `ServerConfig.server_bin` defaults to `""` — `build_server_cmd` falls back to `Config().llama_server_bin_intel`. Provide an explicit path in tests to avoid needing the binary on disk.
- `n_gpu_layers` is typed as `Union[int, str]` to support `"all"` for CUDA. Keep it that way.
- Do not import from `llama_cli` inside `llama_manager` — the dependency is one-way.
- The TUI uses `Rich` `Live` context manager; never call `console.print()` while `Live` is active — use `layout` updates or `log` panel instead.

---

## Out of Scope

- Model weights and binary paths are local to the developer's machine — do not hardcode new paths, use `Config` defaults.
- GPU driver setup, SYCL environment variables (`ONEAPI_DEVICE_SELECTOR`), and CUDA library paths are handled by shell wrapper scripts (`run_opencode_models.sh`), not Python.

## Active Technologies
- Python 3.12+ + rich, psutil, pytest, ruff, pyright (001-prd-mvp-spec)
- Local runtime files under resolved runtime dir (`LLM_RUNNER_RUNTIME_DIR` else `$XDG_RUNTIME_DIR/llm-runner`) for lockfiles + JSON artifacts (001-prd-mvp-spec)
- Python 3.12+ + stdlib (`subprocess`, `pathlib`, `venv`, `json`, `dataclasses`, `threading`), rich, psutil (002-build-setup)
- Local filesystem only (source tree + XDG cache/state/data directories) (002-build-setup)

## Recent Changes
- 001-prd-mvp-spec: Added Python 3.12+ + rich, psutil, pytest, ruff, pyright

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan:
specs/001-m4-op-hardening/plan.md
<!-- SPECKIT END -->
