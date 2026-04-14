# Data Model — PRD M1 Slot-First Launch & Dry-Run

## Entity: M1ConfigSchema

- **Fields**:
  - `schema_version: int` (must be `1`)
  - `workstation_defaults: WorkstationDefaults`
  - `slots: list[SlotConfig]` (keyed by `slot_id` semantics)
  - `profiles: dict[str, OperationalProfile]` (optional)
- **Validation rules**:
  - `schema_version == 1`
  - `slot_id` uniqueness after normalization
  - At least one configured slot for launch/dry-run modes

## Entity: WorkstationDefaults

- **Fields**:
  - `backend: str`
  - `bind_address: str`
  - `port_base: int | None`
  - `environment: dict[str, str]`
  - `openai_defaults: dict[str, str | int | bool]`
- **Validation rules**:
  - Backend value may include future backends, but M1 launch eligibility enforced later
  - Bind/port must pass network safety checks
  - `openai_defaults.*` keys are M1-limited to `--port`, `--host`, `--chat-format`, and `--openai`
    (CLI-style keys with leading `--`)

## Entity: SlotConfig

- **Fields**:
  - `slot_id: str`
  - `model_path: str`
  - `backend: str`
  - `bind_address: str`
  - `port: int`
  - `profile: str | None`
  - `environment_overrides: dict[str, str]`
- **Validation rules**:
  - `slot_id` normalized + filesystem-safe
  - `model_path` required for launch
  - Duplicate bind/port collisions are launch-blocking
  - `backend=vllm` is accepted in config but blocked for M1 launch

## Entity: OperationalProfile

- **Fields**:
  - `name: str`
  - `guidance: dict[str, str | int | bool]`
  - `risk_notes: list[str]`
- **Guidance key schema**:
  - Allowed key families: `backend`, `bind_address`, `port`, `threads`, `context_length`,
    `environment.*` (prefixed keys for env overrides), `openai_defaults.*` (prefixed keys for
    OpenAI flag bundle defaults)
  - Unknown keys in guidance are launch-blocking with `failed_check=config_profile` and
    `how_to_fix=remove unknown guidance key X`
- **Scope and specificity rules**:
  - Profile guidance keys are global at profile scope and apply to all referenced slots unless
    superseded by explicit override.
  - At the profile merge layer, profile values overwrite slot/workstation values for the same key
    path.
  - Specificity ordering for key paths is: `slot.<field>` > `workstation.<field>` > global/default
    key path.
  - Explicit override always applies after profile merge and wins over all profile-provided values.
- **Validation rules**:
  - Applied as a distinct precedence layer
  - Unknown profile reference in slot config is launch-blocking
  - Precedence: profile guidance takes precedence over slot/workstation config when a field is
    defined in both; explicit override takes precedence over profile guidance
  - Conflict resolution: when multiple fields conflict within the same precedence layer, the most
    specific field (per-slot vs workstation-wide) wins
  - Validation timing: profile guidance fields are validated against domain constraints (e.g.,
    port range, valid backend values) at resolution time, after profile merge but before explicit
    override application. Invalid guidance is launch-blocking with FR-005.
  - Explicit override interaction: when an explicit override is provided for a field, the override
    value is used regardless of profile guidance validity; invalid profile guidance for overridden
    fields does not block launch.

## Entity: ResolvedLaunchIntent

- **Fields**:
  - `slot_id: str`
  - `binary_path: str`
  - `command_args: list[str]`
  - `model_path: str`
  - `bind_address: str`
  - `port: int`
  - `environment_redacted: dict[str, str]`
  - `openai_flag_bundle: dict[str, str | int | bool]`
  - `hardware_notes: HardwareNotes`
  - `vllm_eligibility: VllmEligibilityRow`
  - `warnings: list[str]`
  - `validation_results: ValidationReport`

## Entity: HardwareNotes

- **Fields**:
  - `backend: str` (e.g., `llama.cpp`, `vllm`)
  - `device_id: str` (SYCL device index or CUDA device ID)
  - `device_name: str` (human-readable device name)
  - `driver_version: str | None` (optional, when available)
  - `runtime_version: str | None` (optional, when available)
- **Validation rules**:
  - `backend` must be a supported backend value
  - `device_id` must match enumeration from hardware discovery tools
  - `driver_version` and `runtime_version` are optional but preferred for diagnostics

## Entity: VllmEligibilityRow

- **Fields**:
  - `eligible: bool` (whether vllm backend can launch in M1)
  - `backend: str` (backend name being evaluated, typically `vllm`)
  - `reason: str` (human-readable explanation for eligibility decision)
  - `remediation: str | None` (actionable guidance if `eligible=false`)
  - `supported_backends: list[str] | None` (list of backends this slot supports)
- **Validation rules**:
  - In M1, `eligible` is always `false` for vllm backend
  - `reason` must explicitly state M1 does not support vllm launches
  - `remediation` must reference experimental gate or backend enablement steps

## Entity: ValidationError / ValidationReport

- **ValidationError fields**:
  - `error_code: str`
  - `failed_check: str`
  - `why_blocked: str`
  - `how_to_fix: str`
  - `docs_ref: str | None`
- **ValidationReport fields**:
  - `errors: list[ValidationError]`
  - `warnings: list[str]`
  - `eligible: bool`
- **Validation rules**:
  - If one or more blocking conditions occur, all blockers are included in `errors`

## Entity: SlotLockState

- **Fields**:
  - `slot_id: str`
  - `lock_path: str`
  - `pid: int`
  - `port: int`
  - `started_at: str`
  - `owner_status: "live" | "stale" | "indeterminate"`
- **Validation rules**:
  - Lock owner considered live only if PID exists, owns expected port, and start-time matches
  - Malformed/unreadable/indeterminate lock checks are launch-blocking (`failed_check=lockfile_integrity`)

## Entity: GGUFMetadata (M4 - GGUF Parsing)

- **Fields**:
  - `general_name: str | None`
  - `architecture: str`
  - `tokenizer_type: str`
  - `embedding_length: int`
  - `block_count: int`
  - `context_length: int`
  - `attention_head_count: int`
  - `attention_head_count_kv: int`
  - `file_path: str`
  - `normalized_stem: str`
  - `parse_status: "success" | "timeout" | "corrupt" | "unsupported"`
- **Validation rules**:
  - Parse timeout (5s default) treated as `unsupported/corrupt`
  - Missing `general_name` falls back to normalized filename stem for smoke model_id
  - Prefix read cap (32 MiB default) enforced to avoid full weight loading
  - Test fixtures required for CI validation (AC-018)

## Entity: RiskAcknowledgementRecord

- **Fields**:
  - `slot_id: str`
  - `risk_type: "low_port" | "non_loopback_bind" | "warning_bypass"`
  - `acknowledged: bool`
  - `attempt_id: str`
- **State rules**:
  - Session/attempt scoped only; never persisted across launch attempts

## Entity: HardwareAllowlistEntry (FR-017)

- **Fields**:
  - `machine_fingerprint: str`
  - `pci_device_ids: list[str]`
  - `granted_at: str`
  - `granted_by: str` (user identifier)
  - `expires: str | None` (None = permanent until fingerprint changes)
- **Validation rules**:
  - Invalidated automatically when fingerprint/PCI set changes
  - JSON file under `~/.config/llm-runner/hardware-allowlist.json`
  - User must re-ack when hardware changes

## Entity: SessionSnooze (FR-017)

- **Fields**:
  - `snooze_id: str` (ephemeral)
  - `created_at: str`
  - `expires_at: str` (end of session or reboot)
  - `hardware_fingerprint: str`
- **Validation rules**:
  - Ephemeral file under `$XDG_RUNTIME_DIR/llm-runner/` or `/tmp/llm-runner/`
  - Suppresses duplicate warnings until exit/reboot
  - Never replaces persistent allowlist

## Entity: ObservabilityArtifact

- **Fields**:
  - `artifact_id: str`
  - `timestamp: str`
  - `slot_scope: list[str]`
  - `resolved_command: dict[str, list[str]]`
  - `validation_results: ValidationReport`
  - `warnings: list[str]`
  - `environment_redacted: dict[str, str]`
- **Validation rules**:
  - Stored under resolved runtime artifact dir
  - File permission `0600`, directory permission `0700`
  - Persistence failures are launch-blocking (`failed_check=artifact_persistence`)

## Entity: ReadmeMarker (M0 - Documentation Generation)

- **Fields**:
  - `section_name: str`
  - `content: str`
  - `start_marker: str` (e.g., `<!-- readme:workstation -->`)
  - `end_marker: str` (e.g., `<!-- /readme:workstation -->`)
- **Validation rules**:
  - Section names are alphanumeric with hyphens only
  - Content is extracted verbatim between markers
  - Used by gendoc.py for README injection (M0 milestone, not M1)

## Entity: HardwareFingerprint

- **Fields**:
  - `gpu_devices: list[GPUDevice]`
  - `sycl_devices: list[SYCLDevice]`
  - `fingerprint_hash: str` (SHA256 of device enumeration)
- **GPUDevice fields**: `vendor`, `model`, `pci_address`, `driver_version`
- **SYCLDevice fields**: `vendor`, `model`, `backend`, `device_id`
- **Validation rules**:
  - Fingerprint changes when any device attribute changes
  - Used for hardware-allowlist matching (FR-017)
  - Computed via `lspci` + `sycl-ls` (M4 milestone)

## Relationships

- `M1ConfigSchema` contains many `SlotConfig` entries.
- `SlotConfig` optionally references one `OperationalProfile`.
- Each `SlotConfig` resolves to one `ResolvedLaunchIntent` per dry-run/launch attempt.
- Each `ResolvedLaunchIntent` produces one `ValidationReport` and one `ObservabilityArtifact`.
- Each `SlotConfig` may have one active `SlotLockState` during runtime.

## State Transitions

1. `ConfigLoaded` → `Validated` (schema + slot/network/backend checks)
2. `Validated` → `Blocked` (one or more FR-005 errors) **or** `LaunchReady`
3. `LaunchReady` → `RiskPending` (if risky conditions detected) **or** `LockCheck`
4. `RiskPending` → `Blocked` (no acknowledgement) **or** `LockCheck`
5. `LockCheck` → `Blocked` (live owner or lockfile integrity failure) **or** `Launchable`
6. `Launchable` → `Launched` (process started) + `ArtifactPersisted`
7. Any persistence failure → `Blocked` with `failed_check=artifact_persistence`
