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

- Per-slot FR-003 canonical payload includes all of:
  `slot_id`, `binary_path`, `command_args`, `model_path`, `bind_address`, `port`,
  `environment_redacted`, `openai_flag_bundle`, `hardware_notes`,
  `vllm_eligibility`, `warnings`, and `validation_results`
- `command_args` remains ordered argv tokens (stable order per slot)
- `warnings` is present for each slot (empty list when none)
- `validation_results` is present for each slot with `passed` and `checks`
- `vllm_eligibility` is present and marks `eligible=false` for M1
- Sensitive env values are redacted by key rule (`KEY|TOKEN|SECRET|PASSWORD|AUTH`);
  filesystem paths remain visible

## 2) Verify launch-blocking validation contract

Trigger known blockers (e.g., duplicate port or invalid backend) and verify FR-005 response includes:

- `error_code`
- `failed_check`
- `why_blocked`
- `how_to_fix`
- optional `docs_ref`

For full-block failures (`launch blocked - no slots could be launched`), verify
multi-error output repeats this block for each error:

```text
<error_code>
  failed_check: <failed_check>
  why_blocked: <why_blocked>
  how_to_fix: <how_to_fix>
```

If multiple checks fail, verify all are present (one repeated block per failing check).

## 3) Verify degraded one-slot behavior

Run with one unavailable slot (conflict/lock) and confirm:

- Available slot remains launch-eligible
- CLI emits degraded preface:
  `warning: launch degraded - some slots blocked`
- Each blocked slot emits warning lines in this format:
  `warning: slot <slot_id>: <error_code> - <why_blocked>`

## 4) Verify lockfile and artifact handling

Check runtime outputs under resolved runtime directory:

- Lockfiles at `slot-{slot_id}.lock` with owner metadata (`pid`, `port`, `started_at`)
- JSON artifacts at `artifacts/artifact-{timestamp}.json` (one per dry-run/launch attempt)
- Permissions: files `0600`, directories `0700`

Runtime verification steps:

1. Capture the runtime path resolution source:
   - `LLM_RUNNER_RUNTIME_DIR` if set, otherwise `$XDG_RUNTIME_DIR/llm-runner`
2. Run dry-run and confirm a line like:
   - `Artifact written: <runtime-dir>/artifacts/artifact-YYYYMMDDTHHMMSSZ.json`
3. Verify the file exists and naming matches `artifact-{timestamp}.json`
   in the `artifacts/` subdirectory.

## 5) Run required quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All four must pass before implementation is considered complete.
