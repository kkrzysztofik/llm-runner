# Quickstart — Validate PRD M1 Slot-First Launch & Dry-Run

## Prerequisites

- Python 3.12 environment set up (`uv sync --extra dev`)
- Feature branch: `001-prd-mvp-spec`
- Spec inputs prepared in `specs/001-prd-mvp-spec/spec.md`

## 1) Run deterministic dry-run

```bash
uv run llm-runner dry-run both
```

Expected outcomes:

- Canonical per-slot dry-run fields present (FR-003)
- `command_args` shown as ordered argv tokens
- `vllm_eligibility` row included and blocked in M1 when backend is `vllm`
- Sensitive env values redacted by key rule; filesystem paths remain visible

## 2) Verify launch-blocking validation contract

Trigger known blockers (e.g., duplicate port or invalid backend) and verify FR-005 response includes:

- `error_code`
- `failed_check`
- `why_blocked`
- `how_to_fix`
- optional `docs_ref`

If multiple blocking checks fail, verify all are returned in `errors[]`.

## 3) Verify degraded one-slot behavior

Run with one unavailable slot (conflict/lock) and confirm:

- Available slot remains launch-eligible
- Unavailable slot emits warning and clear remediation

## 4) Verify lockfile and artifact handling

Check runtime outputs under resolved runtime directory:

- Lockfiles at `slot-{slot_id}.lock` with owner metadata (`pid`, `port`, `started_at`)
- One JSON artifact per dry-run/launch attempt in `artifacts/`
- Permissions: files `0600`, directories `0700`

## 5) Run required quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All four must pass before implementation is considered complete.
