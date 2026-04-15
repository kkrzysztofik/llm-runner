# Implementation Plan: PRD M2 - Build Wizard + Setup Pipeline

**Branch**: `002-build-setup` | **Date**: 2026-04-15 | **Spec**: `/specs/002-build-setup/spec.md`
**Input**: Feature specification from `/specs/002-build-setup/spec.md`

## Summary

Deliver PRD milestone M2 with a serialized llama.cpp build pipeline (SYCL/CUDA), toolchain
diagnostics, setup venv management, provenance capture, and redacted failure reporting. The
implementation follows spec decisions from `research.md`, models entities and invariants in
`data-model.md`, and preserves M1 conventions for FR-005 actionable errors and CLI/TUI consistency.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: stdlib (`subprocess`, `pathlib`, `venv`, `json`, `dataclasses`, `threading`), rich, psutil
**Storage**: Local filesystem only (source tree + XDG cache/state/data directories)
**Testing**: pytest with mocking (`pytest.raises`, `capsys`, monkeypatch); no real GPU or subprocess dependence in tests
**Target Platform**: Linux workstation (anchored Intel Arc B580 + NVIDIA RTX 3090)
**Project Type**: Single-project Python CLI/TUI application
**Performance Goals**: Correctness and determinism over throughput; no explicit latency/SLA target in M2
**Constraints**: Serialized builds only, file lock required, no runtime venv mutation, no sudo automation, redacted/truncated failure logs, fail-fast on non-retryable errors
**Scale/Scope**: Single-operator local workflow; two build backends (`sycl`, `cuda`) plus `both` orchestration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Code quality impact is explicit: changes stay within `llama_manager` (core) and `llama_cli` (I/O), with typed interfaces and no architecture boundary violations.
- [x] Testing plan is explicit: each user story has deterministic unit/regression coverage and validation via `uv run pytest`.
- [x] UX consistency impact is explicit: CLI/TUI use M1-aligned FR-005 structured errors, clear progress states, and explicit dry-run/setup behavior.
- [x] Runtime safety and observability impact is explicit: lockfile serialization, signal-safe cleanup, redaction, provenance/failure artifacts, and documented exit semantics.
- [x] Merge gates are explicit: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` are part of completion criteria.

**Gate Status (pre-design)**: PASS
**Gate Status (post-design)**: PASS (validated against `research.md`, `data-model.md`, `quickstart.md`, and contracts)

## Project Structure

### Documentation (this feature)

```text
specs/002-build-setup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── cli-json-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── llama_cli/
│   ├── cli_parser.py
│   ├── server_runner.py
│   ├── tui_app.py
│   ├── build_cli.py          # planned M2 addition
│   └── setup_cli.py          # planned M2 addition
├── llama_manager/
│   ├── config.py
│   ├── build_pipeline.py     # planned M2 addition
│   ├── toolchain.py          # planned M2 addition
│   ├── setup_venv.py         # planned M2 addition
│   ├── reports.py            # planned M2 addition
│   └── [existing M1 modules]
└── tests/
    ├── test_build_pipeline.py    # planned M2 addition
    ├── test_toolchain.py         # planned M2 addition
    ├── test_setup_venv.py        # planned M2 addition
    ├── test_reports.py           # planned M2 addition
    └── [existing M1 tests]
```

**Structure Decision**: Keep the existing single-project layout. Add M2 build/setup/report modules
inside `llama_manager`, expose them via new CLI entry modules in `llama_cli`, and add focused test
modules under `src/tests` to preserve the established architecture and test conventions.

## Complexity Tracking

No constitution violations require exception handling in this plan.
