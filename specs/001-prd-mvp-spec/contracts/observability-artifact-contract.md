# Contract: FR-007 Observability Artifact

## Purpose

Defines required JSON artifact persisted for each launch/dry-run attempt.

## Required Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `timestamp` | string | yes | Creation time for artifact |
| `slot_scope` | array[string] | yes | Slot IDs included in this attempt |
| `resolved_command` | object | yes | Slot→argv representation |
| `validation_results` | object | yes | Structured validation outcomes |
| `warnings` | array[string] | yes | Non-blocking warnings |
| `environment_redacted` | object | yes | Redacted environment snapshot |

## Path + Permission Requirements

- Artifact directory: `$LLM_RUNNER_RUNTIME_DIR/artifacts/` when set and usable, otherwise `$XDG_RUNTIME_DIR/llm-runner/artifacts/`.
- Directory permissions must be `0700` (owner only).
- Artifact file permissions must be `0600` (owner only).
- Any persistence failure is launch-blocking (`failed_check=artifact_persistence`).

## Redaction Rules

- Redact values when env key contains: `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `AUTH` (case-insensitive).
- Keep filesystem paths visible.
