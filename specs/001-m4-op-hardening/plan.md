# Implementation Plan: M4 — Operational Hardening and Smoke Verification

**Branch**: `003-m4-op-hardening` | **Date**: 2026-04-23 | **Spec**: [spec.md](specs/001-m4-op-hardening/spec.md)
**Input**: Feature specification from `/specs/001-m4-op-hardening/spec.md`

## Summary

M4 adds operational hardening and smoke verification to llm-runner — a Python TUI application for managing multiple llama.cpp inference server instances across heterogeneous GPU hardware (Intel Arc SYCL + NVIDIA CUDA). The feature adds:

1. **Smoke verification** (`smoke both` / `smoke slot <id>`) — OpenAI-compatible endpoint probing with sequential slot validation, model ID resolution, and composite reporting.
2. **GGUF metadata extraction** — Pinned `gguf` PyPI dependency for header-only parsing with prefix cap and timeout.
3. **Graceful shutdown** — SIGTERM → SIGKILL escalation with orphan prevention and exit code 130.
4. **Hardware/VRAM warnings** — Non-standard hardware acknowledgment, VRAM heuristic with confirmation path.
5. **Slot state machine** — Six operational states (idle, launching, running, degraded, crashed, offline) with explicit transitions.
6. **Lockfiles** — Per-slot lockfiles with stale detection (5 min, psutil-based).
7. **Extended exit code families** — Doctor (1-9), smoke (10-19), process manager (130).
8. **Rotating log + redaction** — Command audit trail with secret redaction and report directories.

## Technical Context

**Language/Version**: Python 3.12+ (project `.python-version`)  
**Primary Dependencies**: `rich` (TUI), `psutil` (process management), `httpx` (smoke HTTP probes, new), `gguf` PyPI (metadata extraction, new, per PRD FR-014)  
**Storage**: Local filesystem — XDG runtime dir for lockfiles (`$XDG_RUNTIME_DIR/llm-runner/`), `~/.config/llm-runner/` for hardware allowlist, `~/.local/share/llm-runner/reports/` for smoke/doctor reports  
**Testing**: `pytest` with `--cov`, `ruff check`, `ruff format --check`, `pyright` — all CI gates  
**Target Platform**: Linux server (ubuntu-latest CI, developer workstation with Intel Arc + NVIDIA GPU)  
**Project Type**: CLI + TUI desktop application (pure Python, no web framework)  
**Performance Goals**: Launch + shutdown cycle < 120s (SC-001a); shutdown initiates within 1s of user request (SC-001b); GGUF metadata extraction < 5s under 32 MiB cap (SC-003); TUI poll interval ≤ 1000ms (NFR-006)  
**Constraints**: `llama_manager` must remain a pure library (no argparse, no Rich, no subprocess at module level); smoke probes use `httpx` (not stdlib `urllib`); GGUF parsing uses pinned `gguf` PyPI dependency (not custom parser); only `llama.cpp` runtime supported (vLLM explicitly rejected); lockfiles prevent concurrent mutating commands; no subprocess spawning in tests  
**Scale/Scope**: 2 GPU slots (arc_b580 on Intel SYCL, rtx3090 on NVIDIA CUDA); 2 models (summary Qwen 3.5-2B, code Qwen 3.5-35B); sequential smoke probing across slots

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Code quality impact is explicit**: Changes span `llama_manager/` (metadata extraction, shutdown, lockfiles, VRAM heuristic) and `llama_cli/` (smoke command, hardware warnings, TUI key handler). `llama_manager` remains pure library. All new code passes `ruff` + `pyright`. No compatibility shims or variant-file forks.
- [x] **Testing plan is explicit**: Unit tests for GGUF metadata extraction with committed synthetic fixtures (FR-008), smoke phase logic with mocked HTTP (FR-003), shutdown signal flow (FR-010), VRAM/heuristic decision trees (FR-013, FR-016). No subprocess spawning of real servers in CI. Regression test for smoke CLI parser integration.
- [x] **UX consistency impact is explicit**: CLI and TUI expose consistent smoke results (FR-020 JSON schema parity). Dry-run shows OpenAI flag bundles and compatibility matrix. Diagnostics attribute failures to slot → model → process → phase with remediation hints (NFR-002). Hardware warnings require explicit acknowledgment in both TUI (`y`/`n` key) and CLI (`--ack-nonstandard-hardware`).
- [x] **Runtime safety and observability impact is explicit**: Rotating log includes command, timestamp, exit code, truncated output (FR-016). Secret patterns redacted. Report directories under predictable path. Exit code families documented (Appendix B). Shutdown escalates SIGTERM→SIGKILL with orphan prevention. Lockfiles prevent concurrent mutating commands.
- [x] **Merge gates are explicit**: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, and `uv run pytest` included in validation steps. CI audit job runs `uv run pip-audit` for dependency CVEs.

## Project Structure

### Documentation (this feature)

```text
specs/001-m4-op-hardening/
├── spec.md              # Feature specification (308 lines, complete)
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0 output (to be generated)
├── data-model.md        # Phase 1 output (to be generated)
├── quickstart.md        # Phase 1 output (to be generated)
├── contracts/           # Phase 1 output (to be generated)
│   └── smoke-api.md     # Smoke probe API contract (httpx-based)
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── llama_manager/
│   ├── config.py           # Config + ServerConfig dataclasses (ADD: smoke, hardware, lockfile fields)
│   ├── config_builder.py   # Factory functions (ADD: smoke config builders)
│   ├── server.py           # build_server_cmd() + validators (MODIFY: VRAM heuristic)
│   ├── process_manager.py  # ServerManager — subprocess lifecycle (MODIFY: shutdown escalation, lockfiles)
│   ├── gpu_stats.py        # GPUStats (MODIFY: VRAM query)
│   ├── metadata.py         # NEW: GGUF metadata extraction (pinned gguf dep)
│   ├── smoke.py            # NEW: Smoke probe logic (httpx-based)
│   ├── log_buffer.py       # Existing (unchanged)
│   └── colors.py           # Existing (unchanged)
├── llama_cli/
│   ├── cli_parser.py       # MODIFY: smoke subcommands, --json, --api-key, --model-id
│   ├── server_runner.py    # MODIFY: smoke command entry point
│   ├── tui_app.py          # MODIFY: slot state machine, hardware warnings, TUI key handler
│   ├── smoke_cli.py        # NEW: smoke CLI orchestrator
│   └── dry_run.py          # MODIFY: show smoke flag bundles
├── tests/
│   ├── test_config.py      # MODIFY: smoke/hardware/lockfile config fields
│   ├── test_server.py      # MODIFY: VRAM heuristic validators
│   ├── test_metadata.py    # NEW: GGUF metadata extraction tests
│   ├── test_smoke.py       # NEW: smoke probe logic tests (mocked HTTP)
│   ├── test_process_manager.py # NEW: shutdown + lockfile tests
│   └── fixtures/
│       ├── gguf_v3_valid.bin      # NEW: valid GGUF v3 with all required keys
│       ├── gguf_v3_no_name.bin    # NEW: valid GGUF v3 missing general.name
│       ├── gguf_corrupt.bin       # NEW: corrupt file (bad magic bytes)
│       ├── gguf_truncated.bin     # NEW: truncated file (valid header, no KV data)
│       └── gguf_v4_unsupported.bin # NEW: valid GGUF v4 (expected error)
scripts/
└── generate_gguf_fixtures.py  # NEW: synthetic GGUF fixture generator

pyproject.toml  # MODIFY: add httpx, gguf dependencies
```

**Structure Decision**: The existing `src/llama_manager/` + `src/llama_cli/` split is preserved. New modules `metadata.py` and `smoke.py` live in `llama_manager` as pure library code. The smoke CLI orchestrator lives in `llama_cli/smoke_cli.py`. Test fixtures are committed as binary files under `tests/fixtures/`, generated by `scripts/generate_gguf_fixtures.py`.

## Complexity Tracking

> No constitution violations — all changes align with architecture boundaries and coding standards.
