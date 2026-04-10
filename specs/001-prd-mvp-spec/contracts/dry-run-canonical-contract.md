# Contract: FR-003 Canonical Dry-Run Schema

## Required Per-Slot Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `slot_id` | string | yes | Normalized slot identifier |
| `binary_path` | string | yes | Effective server binary path |
| `command_args` | array[string] | yes | Ordered raw argv tokens |
| `model_path` | string | yes | Effective model path |
| `bind_address` | string | yes | Effective bind host |
| `port` | integer | yes | Effective port |
| `environment_redacted` | object | yes | Redacted env snapshot |
| `openai_flag_bundle` | object | yes | OpenAI compatibility flags/values |
| `hardware_notes` | object | yes | Includes backend/device info |
| `vllm_eligibility` | object | yes | M1 launch-eligibility row |
| `warnings` | array[string] | yes | Non-blocking operator warnings |
| `validation_results` | object | yes | Blocking + non-blocking validation output |

## `hardware_notes` Contract

- Required fields: `backend`, `device_id`, `device_name`
- Optional fields: `driver_version`, `runtime_version`

## `vllm_eligibility` Contract (M1)

- Includes eligibility status and actionable blocked details.
- In M1, `vllm` is surfaced as non-eligible for launch with FR-011 remediation guidance.

## Field Type Alignment

All field types in this contract MUST align with Python type annotations used in the implementation.
Specifically:
- `slot_id`, `binary_path`, `model_path`, `bind_address`: Python `str` → JSON string
- `port`: Python `int` → JSON integer
- `command_args`: Python `list[str]` → JSON array of strings
- `environment_redacted`, `openai_flag_bundle`, `hardware_notes`, `vllm_eligibility`, `validation_results`:
  Python `dict` → JSON object
- `warnings`: Python `list[str]` → JSON array of strings

Type conversions MUST be explicit and deterministic; no implicit casting or type coercion is allowed.

## `openai_flag_bundle` Schema

- **Type**: object (dict[str, str | int | bool])
- **Required keys**: none (bundle is opt-in based on configuration)
- **Allowed keys for M1**:
  - `--port` (int): OpenAI-compatible port override
  - `--host` (str): OpenAI-compatible host override
  - `--chat-format` (str): chat template identifier
  - `--openai` (bool): OpenAI compatibility mode flag
- Keys MUST retain leading `--` and map directly to effective CLI-style OpenAI flags.
- Unknown keys are FR-003 canonical-schema violations in M1.
- **Deterministic serialization requirement**:
  - JSON object keys MUST be serialized in sorted ascending key order
  - Nested dict values follow the same sorted-key serialization
  - All string values are UTF-8 encoded; no surrogate characters
  - New allowed keys require spec/contract update before use

## Determinism Rules

- Identical inputs must produce identical canonical output content.
- Human-readable output and machine-parseable output both derive from this same canonical schema.
- **errors[] ordering**: When FR-005 returns multiple errors in `validation_results.errors[]`,
  entries are ordered first by slot configuration sequence (slot_id iteration order), then by
  `failed_check` ascending within each slot.
- **warnings[] ordering**: Warnings are ordered by slot configuration sequence (slot_id iteration order).
- **SC-003/FR-003 consistency**: Deterministic ordering rules apply identically to both SC-003
  (override resolution verification) and FR-003 (dry-run canonical schema) to ensure consistent
  operator verification across repeated runs with identical inputs.
