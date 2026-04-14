# Contract: FR-005 Actionable Errors

## Purpose

Defines the launch/dry-run blocking error payload shared by CLI and TUI semantics.

## Single Error Object

| Field | Type | Required | Notes |
|---|---|---|---|
| `error_code` | string | yes | Stable machine-readable code |
| `failed_check` | string | yes | Validation check identifier |
| `why_blocked` | string | yes | Human-readable explanation |
| `how_to_fix` | string | yes | Actionable remediation |
| `docs_ref` | string | no | Optional path/URL reference |

## Lock Integrity Failure Modes

| Failure Mode              | error_code                    | failed_check           |
| ------------------------- | ----------------------------- | ---------------------- |
| malformed_content         | LOCKFILE_INTEGRITY_FAILURE    | lockfile_integrity     |
| unreadable_content        | LOCKFILE_INTEGRITY_FAILURE    | lockfile_integrity     |
| indeterminate_owner       | LOCKFILE_INTEGRITY_FAILURE    | lockfile_integrity     |
| start_time_mismatch       | LOCKFILE_INTEGRITY_FAILURE    | lockfile_integrity     |
| atomicity_violation       | LOCKFILE_INTEGRITY_FAILURE    | lockfile_integrity     |

- **why_blocked determinism**: `why_blocked` MUST include the failure mode token (e.g., `malformed_content`,
  `unreadable_content`, `indeterminate_owner`, `start_time_mismatch`, `atomicity_violation`) for
  deterministic operator verification across runs.

## Multi-Error Response

```json
{
  "errors": [
    {
      "error_code": "BACKEND_NOT_ELIGIBLE",
      "failed_check": "vllm_launch_eligibility",
      "how_to_fix": "change backend to 'llama_cpp' for M1",
      "why_blocked": "vllm is not launch-eligible in PRD M1"
    }
  ]
}
```

## Required Semantics

- If multiple launch blockers are found in one resolution pass, all must be included.
- CLI and TUI must preserve the same contract fields and meanings.
- Canonical `vllm` blocked remediation fields are mandatory in M1.

## Canonical vllm Blocked Response (FR-011)

```json
{
  "errors": [
    {
      "error_code": "BACKEND_NOT_ELIGIBLE",
      "failed_check": "vllm_launch_eligibility",
      "how_to_fix": "change backend to 'llama_cpp' for M1",
      "why_blocked": "vllm is not launch-eligible in PRD M1"
    }
  ]
}
```

**Key Requirements**:
- `error_code` MUST be `BACKEND_NOT_ELIGIBLE` (uppercase with underscores)
- `failed_check` MUST be `vllm_launch_eligibility` (lowercase with underscores)
- `why_blocked` MUST include "not launch-eligible in PRD M1" verbatim
- `how_to_fix` MUST provide explicit correction to `llama_cpp`
- `docs_ref` is optional but recommended pointing to `specs/001-prd-mvp-spec/spec.md#fr-011`

## Canonical Error Code Reference (M1)

| error_code | failed_check | When Used |
| --- | --- | --- |
| `BACKEND_NOT_ELIGIBLE` | `vllm_launch_eligibility` | vllm backend selected in M1 |
| `DUPLICATE_SLOT` | `slot_uniqueness` | Multiple slots with same slot_id |
| `INVALID_SLOT_ID` | `slot_id_format` | slot_id contains non-allowed characters |
| `LOCKFILE_INTEGRITY_FAILURE` | `lockfile_integrity` | Malformed, unreadable, or indeterminate lock |
| `ARTIFACT_PERSISTENCE_FAILURE` | `artifact_persistence` | Cannot write artifact with required permissions |
| `RUNTIME_DIR_UNAVAILABLE` | `runtime_dir_resolution` | Neither LLM_RUNNER_RUNTIME_DIR nor XDG_RUNTIME_DIR usable |
| `PORT_CONFLICT` | `port_availability` | Port already in use |
| `MODEL_NOT_FOUND` | `model_source` | Model file does not exist |

## Deterministic Formatting Rules

- **error_code**: MUST be uppercase with underscores, no spaces or special characters.
- **failed_check**: MUST be lowercase with underscores, no spaces or special characters.
- **why_blocked/how_to_fix**: Plain text only; no markdown, no inline code ticks, no HTML.
- **docs_ref**: If present, MUST be a relative path (e.g., `specs/001-prd-mvp-spec/spec.md`) or valid URL.
- **Field ordering**: JSON object keys in error objects MUST be serialized in alphabetical order.
- **Error ordering**: When multiple errors exist, order by `failed_check` ascending, then by `error_code` ascending.
