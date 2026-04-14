# Implementation Plan: PRD M2 — Build Wizard + Setup Pipeline

**Branch**: `002-build-setup` | **Date**: 2026-04-15 | **Spec**: `/specs/002-build-setup/spec.md`
**Input**: Feature specification from `/specs/002-build-setup/spec.md`
**Milestone Scope**: M2 only — TUI build pipeline, toolchain diagnostics, setup venv, build provenance, failure reports. Not M0, M3, or M4.

## Summary

Implement PRD M2 build wizard and setup pipeline with serialized SYCL/CUDA builds, preflight
toolchain checks, structured failure reporting, build provenance tracking, and venv creation. The
implementation must enforce serialized execution, produce FR-005 actionable errors for all failure
paths, and maintain the core/CLI boundary established in M1.

**Note**: This plan covers M2 milestone only. M0 (documentation), M1 (slot-first, completed),
M3 (profiling), and M4 (operational) are deferred to follow-on specifications.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: rich, psutil, pytest, ruff, pyright (no new deps for M2)
**Storage**: XDG directories — cache (`$XDG_CACHE_HOME/llm-runner/`), state (`$XDG_STATE_HOME/llm-runner/`), data (`$XDG_DATA_HOME/llm-runner/`)
**Testing**: pytest unit tests with mocked subprocess calls; no GPU or real build tools in CI
**Target Platform**: Linux workstation (Intel Arc SYCL + NVIDIA CUDA)
**Project Type**: Python CLI/TUI application plus pure-library core
**Performance Goals**: Preflight check p95 ≤2 seconds; build pipeline startup ≤1 second before subprocess
**Constraints**: `llama_manager` remains pure library; no `argparse`/`Rich`/module-level subprocess; FR-005 multi-error contract; FR-007 redaction and permission rules

## Constitution Check

*GATE: Must pass before implementation begins.*

- [x] Code quality impact is explicit: planned changes add new modules to `llama_manager/` (build_pipeline, toolchain, setup_venv, reports) and `llama_cli/` (build_cli, setup_cli); core/CLI boundary preserved; validation includes `ruff` + `pyright`.
- [x] Testing plan is explicit: add/update deterministic tests for build pipeline, toolchain detection, provenance, failure reports, and setup venv using `uv run pytest` with mocked subprocess calls.
- [x] UX consistency impact is explicit: new `build` and `setup` CLI subcommands follow M1 conventions; FR-005 structured errors reused from M1; exit codes follow PRD Appendix B.
- [x] Runtime safety and observability impact is explicit: build locks prevent concurrent execution; provenance records build origin; failure reports with redaction; no auto-build on launch.
- [x] Merge gates are explicit: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` are mandatory validation steps.

## Project Structure

### Documentation (this feature)

```text
specs/002-build-setup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── llama_manager/
│   ├── config.py              # MODIFY: add XDG paths, build config fields, new ErrorCodes
│   ├── config_builder.py      # MODIFY: add build config factory functions
│   ├── server.py              # EXISTING: ErrorDetail, redact_sensitive
│   ├── process_manager.py     # EXISTING: lock patterns, runtime dir
│   ├── build_pipeline.py      # NEW: BuildConfig, BuildArtifact, BuildProgress, BuildPipeline
│   ├── toolchain.py           # NEW: ToolchainStatus, detect_toolchain(), get_toolchain_hints()
│   ├── setup_venv.py          # NEW: get_venv_path(), create_venv(), check_venv_integrity()
│   └── reports.py             # NEW: write_failure_report(), rotate_reports()
├── llama_cli/
│   ├── cli_parser.py          # MODIFY: add build/setup subcommands
│   ├── server_runner.py       # MODIFY: add subcommand dispatch
│   ├── build_cli.py           # NEW: build_main() entry point
│   └── setup_cli.py           # NEW: setup_main() entry point
└── tests/
    ├── test_build_pipeline.py # NEW: BuildPipeline unit tests (mocked subprocess)
    ├── test_toolchain.py      # NEW: ToolchainStatus, detect_toolchain unit tests
    ├── test_setup_venv.py     # NEW: venv creation/integrity tests
    ├── test_reports.py        # NEW: failure report, rotation, redaction tests
    ├── test_build_cli.py      # NEW: CLI integration tests (mocked pipeline)
    └── test_setup_cli.py      # NEW: CLI integration tests (mocked setup)
```

**Structure Decision**: Keep the existing single-repo Python architecture; implement core build logic in
`src/llama_manager/`, wire CLI flows in `src/llama_cli/`, and enforce behavior through deterministic
tests in `src/tests/`.

## Complexity Tracking

No constitution violations identified at planning time; no exceptions requested.
