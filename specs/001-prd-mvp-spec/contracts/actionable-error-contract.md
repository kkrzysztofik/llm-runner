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
      "why_blocked": "vllm is not launch-eligible in PRD M1",
      "how_to_fix": "change backend to 'llama_cpp' for M1"
    }
  ]
}
```

## Required Semantics

- If multiple launch blockers are found in one resolution pass, all must be included.
- CLI and TUI must preserve the same contract fields and meanings.
- Canonical `vllm` blocked remediation fields are mandatory in M1.
