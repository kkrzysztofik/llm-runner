# Contract: FR-007 Observability Artifact

## Purpose

Defines required JSON artifact persisted for each launch/dry-run attempt.

## Required Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `artifact_id` | string | yes | Unique identifier (UUID or timestamp-based) |
| `timestamp` | string | yes | Creation time for artifact (ISO 8601 UTC) |
| `slot_scope` | array[string] | yes | Slot IDs included in this attempt |
| `resolved_command` | object | yes | Slot→argv representation (per-slot) |
| `validation_results` | object | yes | Structured validation outcomes (per-slot) |
| `warnings` | array[string] | yes | Non-blocking warnings |
| `environment_redacted` | object | yes | Redacted environment snapshot |
| `attempt_id` | string | yes | Launch attempt identifier (for risk acknowledgement tracking) |
| `hardware_warning` | object | no | FR-017 hardware warning (when non-anchor topology detected) |

## Path + Permission Requirements

- Artifact directory: `$LLM_RUNNER_RUNTIME_DIR/artifacts/` when set and usable, otherwise `$XDG_RUNTIME_DIR/llm-runner/artifacts/`.
- Directory permissions must be `0700` (owner only).
- Artifact file permissions must be `0600` (owner only).
- Any persistence failure is launch-blocking (`failed_check=artifact_persistence`).

## Hardware Warning Object (FR-017)

When non-anchor hardware is detected, include:

```json
{
  "hardware_warning": {
    "warning_type": "non_anchor_hardware",
    "detected_config": {
      "gpu_devices": [...],
      "sycl_devices": [...]
    },
    "expected_config": {
      "gpu_devices": [{"vendor": "Intel", "model": "Arc B580"}, {"vendor": "NVIDIA", "model": "RTX 3090"}],
      "sycl_devices": [...]
    },
    "acknowledged": true | false,
    "ack_token": "ephemeral-token" | null
  }
}
```

**Note**: Hardware fingerprint computation uses `lspci` + `sycl-ls` output hash (M4 milestone).

## Redaction Rules

- **Trigger patterns** (case-insensitive substring match on env key):
  - `KEY`
  - `TOKEN`
  - `SECRET`
  - `PASSWORD`
  - `AUTH`
- **Redaction format**: Replace value with `"[REDACTED]"` (string literal).
- **Visible fields**: Filesystem paths (e.g., `MODEL_PATH`, `CONFIG_DIR`) remain visible even if they contain sensitive-looking substrings.
- **Timestamp format**: ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
- **Filename rules**: Artifact filenames must be filesystem-safe (alphanumeric, hyphen, underscore only; no spaces or special characters).
