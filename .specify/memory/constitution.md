<!--
Sync Impact Report
- Version change: template-unversioned -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. Code Quality Is a Release Gate
  - Template Principle 2 -> II. Testing Standards Are Non-Negotiable
  - Template Principle 3 -> III. User Experience Consistency Across CLI and TUI
  - Template Principle 4 -> IV. Deterministic Configuration and Runtime Safety
  - Template Principle 5 -> V. Observability and Failure Transparency
- Added sections:
  - Engineering Standards
  - Workflow and Review Gates
- Removed sections: None
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ✅ .specify/templates/commands/*.md (no files present)
  - ✅ README.md (reviewed; no constitution-reference changes required)
- Deferred TODOs: None
-->
# llm-runner Constitution

## Core Principles

### I. Code Quality Is a Release Gate

- Changes MUST preserve architecture boundaries: `llama_manager` remains a pure core library and
  `llama_cli` owns user-facing I/O.
- New and changed interfaces MUST be type-annotated and aligned with repository conventions in
  `AGENTS.md`.
- Every merged change MUST pass `uv run ruff check .`, `uv run ruff format --check .`, and
  `uv run pyright`.
- Compatibility shims, speculative abstractions, and variant-file forks (for example `*_v2.py`)
  MUST NOT be introduced.
Rationale: early-stage velocity depends on clean structure, strict static checks, and low tech debt.

### II. Testing Standards Are Non-Negotiable

- Every behavior change MUST add or update automated tests in `tests/`.
- Bug fixes MUST include a regression test that fails before the fix and passes after the fix.
- Changes are complete only when `uv run pytest` passes, including updated assertions for new
  validation and error paths.
- Tests MUST be deterministic in CI and MUST NOT depend on local GPUs, external services, or
  unmanaged subprocess state.
Rationale: deterministic tests are the primary protection for rapid iteration on serving workflows.

### III. User Experience Consistency Across CLI and TUI

- User-facing flows MUST follow PRD terminology and behaviors for `doctor`, `setup`, `smoke`,
  dry-run, and slot-first operation.
- CLI and TUI MUST expose consistent meanings for status, errors, warnings, and remediation steps.
- Dry-run and diagnostics outputs MUST be explicit enough for users to reconstruct launch intent,
  including effective ports, backends, and risk notes.
- Risky operations MUST require explicit acknowledgement or confirmation; safe local defaults MUST
  remain the baseline.
Rationale: operators depend on predictable, explainable workflows more than hidden automation.

### IV. Deterministic Configuration and Runtime Safety

- Slot-level invariants from the PRD are mandatory: one active model per slot and slot-owned bind
  host and port behavior.
- Configuration resolution order MUST be deterministic and documented so identical inputs produce
  identical commands.
- Runtime serve paths MUST NOT mutate Python environments or install packages implicitly.
- Shutdown paths MUST clean up child processes to prevent orphaned GPU workloads.
Rationale: deterministic runtime behavior reduces operational risk and debugging time.

### V. Observability and Failure Transparency

- Logs and errors MUST identify failures by slot, model, process stage, and actionable next step.
- Setup and diagnostic outputs MUST redact secrets while preserving local troubleshooting value.
- Build, profile, and smoke workflows MUST preserve provenance and report artifacts required for
  reproducibility.
- Commands MUST return documented, non-ambiguous exit codes for automation.
Rationale: transparent failures and durable diagnostics are required for reliable solo operations.

## Engineering Standards

- Python 3.12+ and project tooling in `pyproject.toml` are the required baseline.
- `ruff`, `pyright`, and `pytest` are mandatory quality gates for merge readiness.
- All new work MUST favor direct, in-place improvements over compatibility wrappers or duplicate
  replacement files.
- Repository guidance in `AGENTS.md` and product behaviors in `PRD.md` are normative references
  for implementation choices.

## Workflow and Review Gates

- Every plan MUST include an explicit Constitution Check before implementation starts.
- Every spec MUST include independently testable user scenarios and measurable success criteria.
- Every task list MUST include quality-gate and test tasks; test execution is never optional.
- Code review MUST verify architecture boundaries, test coverage updates, and CLI/TUI consistency.
- Any temporary exception to these principles MUST be documented in the active plan with a
  time-bounded remediation step.

## Governance

- This constitution is the highest-priority engineering policy for this repository.
- Amendments MUST update this file, include a Sync Impact Report, and propagate required changes to
  dependent templates and guidance docs in the same change set.
- Versioning follows semantic rules: MAJOR for incompatible governance changes, MINOR for new
  principle/section requirements, PATCH for clarifications that do not change obligations.
- Compliance review is required in planning and code review: unresolved violations block merge until
  waived with explicit rationale in the plan.
- `AGENTS.md` and `PRD.md` remain authoritative implementation references, but must not contradict
  this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-04-08 | **Last Amended**: 2026-04-08
