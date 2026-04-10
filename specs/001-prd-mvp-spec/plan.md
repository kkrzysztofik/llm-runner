# Implementation Plan: PRD M1 — Slot-First Launch & Dry-Run

**Branch**: `001-prd-mvp-spec` | **Date**: 2026-04-10 | **Spec**: `/specs/001-prd-mvp-spec/spec.md`
**Input**: Feature specification from `/specs/001-prd-mvp-spec/spec.md`

## Summary

Implement PRD M1 slot-first launch/dry-run behavior with deterministic configuration precedence,
per-slot lock ownership checks, structured actionable validation errors, canonical dry-run output,
and JSON observability artifacts. The implementation must enforce M1 backend eligibility (`llama_cpp`
only), preserve CLI/TUI contract parity, and meet stated p95 validation and dry-run timing budgets.

## Technical Context

**Language/Version**: Python 3.12+  
**Primary Dependencies**: rich, psutil, pytest, ruff, pyright  
**Storage**: Local runtime files under resolved runtime dir (`LLM_RUNNER_RUNTIME_DIR` else `$XDG_RUNTIME_DIR/llm-runner`) for lockfiles + JSON artifacts  
**Testing**: pytest unit/regression tests with deterministic fixtures and mocked process/port/runtime-dir behavior  
**Target Platform**: Linux workstation (Intel Arc SYCL + NVIDIA CUDA)  
**Project Type**: Python CLI/TUI application plus pure-library core  
**Performance Goals**: Dry-run p95 ≤250 ms (single slot) / ≤400 ms (two slots); lock/port validation p95 ≤150 ms per slot  
**Constraints**: `llama_manager` remains pure library; no module-level subprocess side effects; FR-005 multi-error contract; FR-007 redaction and permission rules (`0600` files, `0700` dirs)  
**Scale/Scope**: Single-operator workstation, 1–2 concurrent slots, M1-only scope (M0 and M2+ deferred)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Code quality impact is explicit: planned changes remain in `llama_manager/` (core logic) and
      `llama_cli/` (presentation/interaction) with one-way dependency preserved; validation includes
      `ruff` + `pyright`.
- [x] Testing plan is explicit: add/update deterministic tests for slot validation, dry-run schema,
      multi-blocker FR-005 errors, lock integrity, artifact persistence/redaction, and degraded launch
      behavior using `uv run pytest`.
- [x] UX consistency impact is explicit: CLI and TUI share identical FR-005 fields, redaction policy,
      and canonical dry-run semantics; presentation may differ but contract remains aligned.
- [x] Runtime safety and observability impact is explicit: runtime-dir resolution, lock ownership,
      risk acknowledgement, redaction, and artifact failure attribution are covered as blocking checks.
- [x] Merge gates are explicit: `uv run ruff check .`, `uv run ruff format --check .`, and
      `uv run pyright` are mandatory validation steps.

### Post-Design Re-check (after Phase 1 artifacts)

- [x] `research.md` resolves planning decisions for precedence, redaction, deterministic output, and performance verification.
- [x] `data-model.md` documents entities, validations, and state transitions aligned to FR-001..FR-011.
- [x] `contracts/` captures CLI/TUI-facing contracts for FR-003, FR-005, and FR-007 parity.
- [x] `quickstart.md` includes concrete validation and quality-gate execution steps.

## Project Structure

### Documentation (this feature)

```text
specs/001-prd-mvp-spec/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── actionable-error-contract.md
│   ├── dry-run-canonical-contract.md
│   └── observability-artifact-contract.md
└── tasks.md              # Produced later by /speckit.tasks
```

### Source Code (repository root)

```text
llama_manager/
├── config.py
├── config_builder.py
├── server.py
├── process_manager.py
├── gpu_stats.py
└── (new modules may be added only where needed for validators/observability)

llama_cli/
├── cli_parser.py
├── dry_run.py
├── server_runner.py
└── tui_app.py

tests/
├── test_config.py
├── test_server.py
└── (new focused test modules for lock/artifact/error contracts)
```

**Structure Decision**: Keep the existing single-repo Python architecture; implement core behavior in
`llama_manager/`, wire user-facing flows in `llama_cli/`, and enforce behavior through deterministic
tests in `tests/`.

## Complexity Tracking

No constitution violations identified at planning time; no exceptions requested.
