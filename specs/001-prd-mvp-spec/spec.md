# Feature Specification: PRD M1 — Slot-First Launch & Dry-Run

**Feature Branch**: `001-prd-mvp-spec`  
**Created**: 2026-04-08  
**Status**: Draft  
**Input**: User description: "based on PRD.md"

## Clarifications

### Session 2026-04-09

- Q: Should Spec-001 remain M1-scoped or be expanded to full PRD MVP? → A: Keep Spec-001 M1-scoped and explicitly label it as M1.
- Q: How should M0 (documentation generation) be handled relative to Spec-001? → A: Defer M0 to a separate spec and keep a dependency/reference note from Spec-001.
- Q: What dry-run detail level is required for Spec-001? → A: Require full deterministic dry-run detail (binary path, exact args, model path, slot, effective ports, merged env redacted, OpenAI flag bundle, hardware notes, vLLM matrix row).
- Q: How should backend selection behave in M1? → A: Only llama_cpp is launch-eligible in M1; vllm may appear in config but MUST be reported as non-eligible and blocked with actionable error.
- Q: What format should launch-blocking errors use? → A: Use a structured actionable error contract with `error_code`, `failed_check`, `why_blocked`, `how_to_fix`, and optional `docs_ref`.
- Q: How should lock ownership be detected for stale-lock cleanup? → A: Use per-slot lockfiles at `$XDG_RUNTIME_DIR/llm-runner/slot-{slot_id}.lock` with `pid`, `port`, `started_at`; owner is live only if PID exists and still owns the expected port.
- Q: Which M1 conditions are "risky operations" requiring acknowledgement? → A: Port <1024, non-loopback bind, and manual override that bypasses a warning.
- Q: What redaction policy should M1 dry-run and artifacts use? → A: Redact values for keys containing KEY/TOKEN/SECRET/PASSWORD/AUTH as `[REDACTED]`; filesystem paths remain visible.
- Q: How should slot/workstation config be defined for deterministic precedence in M1? → A: Use explicit schema_version:1 with workstation defaults and slot entries keyed by slot_id; after precedence resolution, user override wins per field.
- Q: How should M1 observability artifacts be stored? → A: Persist one JSON artifact per launch/dry-run under `$XDG_RUNTIME_DIR/llm-runner/artifacts/` with timestamp + slot scope, including resolved command, validation results, warnings, and redacted env snapshot.
- Q: How should path redaction be resolved between dry-run detail and observability rules? → A: Keep exact filesystem paths visible in dry-run and artifacts; only sensitive key values are redacted.
- Q: What file permission requirements should apply to lockfiles and artifacts? → A: Require owner-only permissions (`0600` files, `0700` directories); if secure permissions cannot be enforced, block launch with actionable error.
- Q: What denominator should SC-002 use for the 95% actionable-error threshold? → A: Use all FR-005 launch-blocking validation outcomes produced in launch/dry-run paths during M1 acceptance tests.
- Q: What evidence source should SC-003 use for deterministic operator verification? → A: Require deterministic dry-run output for identical inputs and matching FR-007 artifact fields (`resolved_command`, `validation_results`, `warnings`) across repeated runs.
- Q: How should malformed or unreadable lockfiles be handled in M1? → A: Treat as launch-blocking when lock integrity or stale-check cannot be completed, and return FR-005 actionable error with `failed_check=lockfile_integrity`.
- Q: What runtime directory resolution policy should M1 use when runtime dirs are unavailable? → A: Use `LLM_RUNNER_RUNTIME_DIR` first, then `$XDG_RUNTIME_DIR/llm-runner`; if neither is usable, block launch with FR-005 actionable error.
- Q: How should FR-005 represent multiple simultaneous launch blockers? → A: Return a structured multi-error response using `errors: [{error_code, failed_check, why_blocked, how_to_fix, docs_ref?}, ...]` when multiple blockers are present.
- Q: What canonical remediation content should be used when `vllm` is blocked in M1? → A: Use `error_code=BACKEND_NOT_ELIGIBLE`, `failed_check=vllm_launch_eligibility`, `why_blocked=vllm is not launch-eligible in PRD M1`, `how_to_fix=change backend to 'llama_cpp' for M1`.
- Q: How should dry-run output be formalized for determinism and measurability? → A: Define a canonical dry-run schema with required fields and allow both human-readable rendering and machine-parseable representation from that schema.
- Q: What parity requirement should apply between CLI and TUI outputs in M1? → A: Require contract-level parity (same FR-005 fields, redaction rules, and dry-run canonical schema semantics), while allowing presentation differences.
- Q: How should stale-owner verification handle race/uncertain states in M1? → A: Require atomic live-owner verification (`pid exists` + `owns expected port`) as one decision step; if indeterminate, block launch with FR-005 actionable error.
- Q: How should artifact persistence failures be handled in M1? → A: Treat disk/I/O/permission persistence failures as launch-blocking FR-005 errors with `failed_check=artifact_persistence`.
- Q: What M1 performance targets should apply to dry-run and validation? → A: Set p95 targets: dry-run resolution ≤250 ms (single slot) / ≤400 ms (two slots), lock/port validation ≤150 ms per slot.
- Q: How should M1 guard against PID reuse during stale-owner verification? → A: Treat lock owner as live only when PID exists, owns expected port, and process start-time matches lock `started_at`; mismatches/indeterminate checks are launch-blocking with `failed_check=lockfile_integrity`.
- Q: What structure should `hardware_notes` use in the dry-run canonical schema? → A: Use a structured object with required fields `backend`, `device_id`, `device_name`, and optional fields `driver_version`, `runtime_version`.
- Q: How should `command_args` encode quoting/escaping in the dry-run schema? → A: Represent `command_args` as ordered raw argv tokens (`list[str]`) preserving exact token boundaries; shell-escaped joined strings are non-normative.

### Session 2026-04-08

- Q: What should this feature include relative to the full PRD scope? → A: Narrow core only (slot launch + dry-run now; diagnostics/smoke later).
- Q: How should launch behave when one configured slot is unavailable? → A: Allow one-slot launch and warn that the second slot is unavailable.
- Q: What override precedence should be used for launch resolution? → A: defaults < slot/workstation config < profile guidance < explicit override.
- Q: How should lockfiles be handled during launch? → A: Auto-clear stale lock; block only if active owner is detected.
- Q: How long should risk acknowledgement persist? → A: Session-only (current launch attempt only).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Launch Models by Slot (Priority: P1)

As a solo operator, I can configure and launch one or two model-serving workloads using explicit
GPU slot ownership so I can run my local setup without manual script juggling.

**Why this priority**: This is the core value proposition and enables all other operational flows.

**Independent Test**: Configure valid slot assignments, run dry-run and launch, and confirm active
slots start with distinct bindings and clear status while unavailable slots are clearly warned.

**Acceptance Scenarios**:

1. **Given** two valid slot assignments, **When** I run launch for both slots, **Then** each slot
   starts at its declared bind and port without collision.
2. **Given** a duplicate slot assignment or occupied port, **When** I run launch, **Then** the
   system blocks startup and returns an actionable error.
3. **Given** one configured slot is unavailable, **When** I run launch, **Then** launch proceeds
   for the available slot and returns a warning for the unavailable slot.
4. **Given** a stale lockfile with no active owner, **When** I run launch, **Then** stale lock is
   cleared and launch proceeds.
5. **Given** multiple simultaneous blockers (e.g., duplicate slot, occupied port, missing model),
   **When** I run launch or dry-run, **Then** the system returns a structured multi-error FR-005
   response with all blockers included in an `errors` array, each containing required fields
   (`error_code`, `failed_check`, `why_blocked`, `how_to_fix`).
6. **Given** a degraded scenario where one slot has a stale lock and another slot has a port conflict,
   **When** I run launch, **Then** the stale lock is auto-cleared and launch proceeds for the available
   slot, while the port conflict blocks the second slot with an FR-005 actionable error.
7. **Given** a fully blocked scenario where all configured slots have non-recoverable blockers
    (e.g., active lock owner on one slot and occupied port on another), **When** I run launch,
    **Then** launch is blocked entirely with FR-005 multi-error response covering all blocker types.
8. **Given** all configured slots are blocked (e.g., duplicate slot, occupied ports, missing models),
    **When** I run launch, **Then** launch is blocked entirely with FR-005 multi-error response
    listing all blockers in the `errors` array before any partial launch is attempted.

**Slot Unavailability Causes**: When a configured slot cannot launch, the system MUST identify and report the specific cause. In M1, slot-unavailability causes are explicitly enumerated as: (1) port conflict (another process or slot binds the same port), (2) missing model source (model file does not exist at the configured path), (3) active lock owner (a live PID owns the slot lock and owns the expected port), or (4) backend non-eligibility (backend is not launch-eligible in M1, e.g., vllm). Each cause maps to a distinct FR-005 error response.

**Flow Classification (User Story 1)**:

- **Primary flow**: Scenario 1 (dual-slot successful launch).
- **Alternate flow**: Scenarios 3 and 6 (degraded one-slot launch with warning/remediation).
- **Exception/full-block flow**: Scenarios 5, 7, and 8 (all-slot blocked outcomes with FR-005 multi-error).

---

### User Story 2 - Resolve Launch Blocking Errors Early (Priority: P2)

As an operator, I can detect and resolve launch-blocking configuration issues before starting
workloads so startup failures are predictable and actionable.

**Why this priority**: Preventing avoidable startup failures is the next highest value after
successful launch.

**Independent Test**: Attempt launch with known invalid inputs and verify errors identify the
blocking condition and required correction.

**Acceptance Scenarios**:

1. **Given** duplicate slot assignment, **When** I run launch or dry-run, **Then** startup is
   blocked with a clear correction message.
2. **Given** a bind conflict, **When** I run launch, **Then** startup is blocked until the conflict
   is resolved or explicitly acknowledged.

---

### User Story 3 - Use Deterministic Overrides Safely (Priority: P3)

As a tuner, I can apply explicit overrides with deterministic precedence and receive risk prompts
for dangerous launch conditions.

**Why this priority**: Deterministic behavior and safety prompts reduce operational risk during
manual tuning.

**Independent Test**: Launch with conflicting default and override values and confirm the effective
result follows documented precedence with explicit risk acknowledgement gates.

**Acceptance Scenarios**:

1. **Given** explicit overrides are present, **When** launch intent is resolved, **Then** precedence
   is deterministic as defaults < slot/workstation config < profile guidance < explicit override.
2. **Given** launch conditions are risky, **When** I attempt startup, **Then** startup requires
   explicit acknowledgement before proceeding for that launch attempt only.

### Edge Cases

- One-slot degraded launch proceeds with a warning when one configured slot is unavailable.
- Full-block scenarios (all configured slots blocked) block launch entirely with FR-005 multi-error;
  partial launch is not attempted only when all configured slots are blocked.
- Conflicting values resolve deterministically using: defaults < slot/workstation config < profile guidance < explicit override.
- Stale lockfiles are auto-cleared; lockfiles with active owners block launch.
- Risk acknowledgements are non-persistent and apply only to the current launch attempt; in M1 they
  are required for port <1024, non-loopback bind, and warning-bypassing manual overrides.
- Dry-run and artifact outputs redact sensitive keys (KEY/TOKEN/SECRET/PASSWORD/AUTH) as
  `[REDACTED]`; filesystem paths remain visible.
- Lockfiles and runtime artifact directories use owner-only permissions (`0600` files, `0700`
  directories); inability to enforce secure permissions is launch-blocking.
- Malformed or unreadable lockfiles (or stale-check failures) are launch-blocking and return FR-005
  actionable errors with `failed_check=lockfile_integrity`.
- Blocked `vllm` selections use canonical FR-005 remediation fields: `error_code=BACKEND_NOT_ELIGIBLE`,
  `failed_check=vllm_launch_eligibility`, and `how_to_fix=change backend to 'llama_cpp' for M1`.
- Runtime directory resolution uses `LLM_RUNNER_RUNTIME_DIR` first, then
  `$XDG_RUNTIME_DIR/llm-runner`; if neither is writable/usable, launch is blocked with an FR-005
  actionable error.
- When multiple launch-blocking checks fail in one resolution pass, FR-005 returns all blockers in
  a structured `errors` array rather than only a single blocker.
- Live-owner stale-check uses an atomic decision step (`pid exists` + `owns expected port`); if the
  verification result is indeterminate, launch is blocked with FR-005 actionable error.
- PID reuse is guarded by requiring process start-time to match lock `started_at` during live-owner
  verification; mismatch/indeterminate verification is launch-blocking with
  `failed_check=lockfile_integrity`.
- Artifact persistence failures (disk full, I/O error, permission denied after runtime-dir
  resolution) are launch-blocking and return FR-005 actionable errors with
  `failed_check=artifact_persistence`.
- Dry-run `command_args` preserves exact argv token boundaries as `list[str]`; shell-escaped joined
  command strings are non-normative representations.
- Deterministic output: repeated identical inputs (same config, same hardware state) produce identical
  dry-run output and artifact field values (`resolved_command`, `validation_results`, `warnings`)
  with stable ordering across runs.
- Stable ordering: slot iteration order, warnings array, and validation_results array maintain
  deterministic ordering based on slot configuration sequence.
- Determinism scope: hardware/process-state changes (different live owners, changed runtime
  directory availability, changed device/runtime metadata) are treated as distinct inputs and are
  excluded from identical-input comparison assertions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST enforce slot-first orchestration where each running workload is bound to
  a declared slot and each slot owns its bind address and port. Slot ownership invariants: each
  slot_id is unique within a launch configuration (duplicate slot_ids are launch-blocking with
  `error_code="duplicate_slot"`), a slot's bind/port ownership is exclusive (no two slots may share
  the same port), and conflict resolution is explicit: when multiple slots target the same port,
  the system MUST block launch and return a multi-error FR-005 response listing all conflicting slots.
- **FR-002**: System MUST prevent invalid startup states, including duplicate slot assignment,
  missing model source, and conflicting network bindings. Slot ID normalization/boundary rules:
  slot_id values are normalized to lowercase canonical form (`slot_id.lower()` after stripping
  leading/trailing whitespace), only ASCII characters `a-z`, digits `0-9`, hyphen `-`, and underscore
  `_` are allowed in slot_id values (any other character is rejected with `error_code="invalid_slot_id"`
  and an FR-005 actionable error).
- **FR-003**: System MUST provide a deterministic dry-run mode that, before execution, prints for
  each slot: binary path, exact command arguments, model path, slot ID, effective bind/port,
  merged environment values with sensitive values redacted, OpenAI flag bundle, hardware notes,
  and a vLLM matrix row indicating launch eligibility in the current mode. Dry-run MUST conform to
  a canonical schema with required fields: `slot_id`, `binary_path`, `command_args`, `model_path`,
  `bind_address`, `port`, `environment_redacted`, `openai_flag_bundle`, `hardware_notes`,
  `vllm_eligibility`, `warnings`, and `validation_results`. M1 MAY present this as human-readable
  output, machine-parseable output, or both, but all representations MUST derive from the same
  canonical schema. M1 dry-run resolution latency target is p95 ≤250 ms for single-slot resolution
  and p95 ≤400 ms for two-slot resolution. `hardware_notes` MUST be an object containing required
  fields `backend`, `device_id`, `device_name` and optional fields `driver_version`,
  `runtime_version`. `command_args` MUST be represented as ordered raw argv tokens (`list[str]`)
  preserving exact token boundaries (e.g., `["model", "path/with spaces/model.gguf", "--threads",
  "4"]`); shell-escaped joined command strings are non-normative. `openai_flag_bundle` MUST be an
  object keyed by effective CLI-style OpenAI flags (including leading `--`) as defined in the
  dry-run contract.
  Output comparisons in acceptance tests MUST follow the deterministic edge-condition rules in
  the Edge Cases section.
- **FR-004**: System MUST allow degraded one-slot startup when one configured slot is unavailable,
  and MUST emit a clear warning identifying the unavailable slot.
- **FR-005**: System MUST return launch-blocking errors in a structured actionable format containing
  `error_code`, `failed_check`, `why_blocked`, `how_to_fix`, and optional `docs_ref` before startup
  proceeds. If multiple launch-blocking conditions are present, response MUST include
  `errors: [{error_code, failed_check, why_blocked, how_to_fix, docs_ref?}, ...]` covering all
  blockers detected in the current launch/dry-run resolution path.
- **FR-006**: System MUST apply deterministic override precedence in this order: defaults <
  slot/workstation config < profile guidance < explicit override. For M1, config MUST use
  `schema_version: 1`, workstation defaults, and slot entries keyed by `slot_id`; after
  precedence resolution, user override wins per field. Profile guidance MUST take precedence
  over slot/workstation config when a field is defined in both, and explicit override MUST
  take precedence over profile guidance. Within each precedence layer, fields are resolved
  by specificity: per-slot keys override workstation-wide keys, which override global/default keys.
  When multiple fields conflict within the same precedence layer and specificity, the most
  specific field (per-slot key over workstation-wide key over global/default key) wins.
- **FR-007**: System MUST preserve observability artifacts for launch and dry-run outcomes. Redaction
  MUST replace values with `[REDACTED]` for any key name containing `KEY`, `TOKEN`, `SECRET`,
  `PASSWORD`, or `AUTH` (case-insensitive). Filesystem paths MUST remain visible in M1. M1
  artifacts MUST be persisted as one JSON file per launch/dry-run under the resolved runtime
  artifact directory (`$LLM_RUNNER_RUNTIME_DIR/artifacts/` when set and usable, otherwise
  `$XDG_RUNTIME_DIR/llm-runner/artifacts/`) with timestamp + slot scope, containing resolved command,
  validation results, warnings, and a redacted environment snapshot. Artifact lifecycle: M1 uses
  timestamp-based filenames (`artifact-{timestamp}.json`) with collision behavior of overwrite
  (same timestamp is unique per launch invocation); retention window and cleanup responsibility are
  out of scope for M1 (operators may remove artifacts manually or via external process). If neither
  runtime directory is usable, the system MUST fail with a clear attribution to the unavailable path
  and return a launch-blocking FR-005 actionable error. Artifact files MUST be created with owner-only
  permissions (`0600`) and their runtime directories with owner-only permissions (`0700`);
  inability to enforce these permissions MUST return a launch-blocking actionable error.
  Artifact persistence failures (disk full, I/O error, permission denied after path resolution) MUST
  be launch-blocking and return FR-005 actionable error with
  `failed_check=artifact_persistence`. Artifact output comparisons MUST follow the deterministic
  edge-condition rules in the Edge Cases section.
- **FR-008**: System MUST treat runtime safety as default behavior, requiring explicit acknowledgement
  for risky operations; acknowledgement is valid only for the current launch attempt. In M1, risky
  operations are: (1) privileged port (port <1024), (2) non-loopback bind (bind address not
  `127.0.0.1` or `::1`), and (3) manual override that bypasses a warning. Acknowledgement scope
  explicitly covers: dry-run path (acknowledgement required before dry-run completes), and launch
  attempt path (acknowledgement required before subprocess execution); acknowledgement does NOT
  persist across separate launch/dry-run invocations.
- **FR-009**: System MUST auto-clear stale lockfiles when no active owner exists, and MUST block
  launch when a live lock owner is detected. Lockfiles MUST be per-slot under the resolved runtime
  lock directory (`$LLM_RUNNER_RUNTIME_DIR/` when set and usable, otherwise
  `$XDG_RUNTIME_DIR/llm-runner/`) at `slot-{slot_id}.lock` and include `pid`, `port`, `started_at`.
  If neither lock-directory candidate is usable, launch MUST be blocked with FR-005 actionable error
  and `failed_check=lockfile_integrity`; a
  lock owner is live only when the PID exists, still owns the expected port, and process start-time
  matches lock `started_at`. Lockfiles MUST use
  owner-only permissions (`0600`), and parent runtime directories MUST use owner-only permissions
  (`0700`). If lockfile integrity validation fails (malformed/unreadable content) or stale-owner
  verification cannot be completed, launch MUST be blocked and return an FR-005 actionable error
  with `failed_check=lockfile_integrity`. Stale-lock cleanup failure recovery requirements:
  (1) permission denied on lockfile read: block and return FR-005 with `failed_check=lockfile_integrity`,
  (2) concurrent writer during stale-check: block and return FR-005 with `failed_check=lockfile_integrity`,
  (3) malformed/unreadable lock content: block and return FR-005 with `failed_check=lockfile_integrity`.
  Automatic stale-lock recovery is prohibited for indeterminate ownership states (e.g., when
  PID/ownership/start-time verification is ambiguous); this prohibition includes race conditions,
  partial verification matches, and stale-check outcomes that cannot produce a definitive live/stale
  decision in one atomic step. such cases MUST block launch with FR-005. Live-owner verification
  MUST be evaluated as one atomic decision (`pid exists` + `owns expected port`); indeterminate
  verification states are launch-blocking. Lock/port validation latency target is p95 ≤150 ms per slot.
- **FR-010**: System MUST scope this feature to PRD M1 launch and dry-run core behavior; non-M1 work
  is deferred to separate follow-on specifications. M0 documentation generation is tracked as a
  separate dependent spec. M2 diagnostic tools, M3 smoke-testing workflow, and M4 multi-workstation
  orchestration are each scoped to dedicated follow-on specs named "PRD M2: Diagnostic Tools",
  "PRD M3: Smoke-Testing Workflow", and "PRD M4: Multi-Workstation Orchestration". This spec
  excludes M0 acceptance criteria.
- **FR-011**: System MUST accept backend values in configuration for forward compatibility, but in
  PRD M1 only `llama_cpp` is launch-eligible; `vllm` MUST be surfaced as non-eligible and blocked
  with an actionable correction message. `launch-eligible` is a validation-state concept in M1:
  it describes the system's eligibility decision at validation time, not a persistent runtime state.
  For blocked `vllm`, FR-005 response MUST include
  `error_code=BACKEND_NOT_ELIGIBLE`, `failed_check=vllm_launch_eligibility`,
  `why_blocked=vllm is not launch-eligible in PRD M1`, and
  `how_to_fix=change backend to 'llama_cpp' for M1`.

### Constitution Alignment *(mandatory)*

- **CA-001 Code Quality**: This feature MUST preserve core/CLI boundary integrity and pass all
  required static-quality gates before merge.
- **CA-002 Testing**: This feature MUST include deterministic automated tests for success paths,
  failure paths, and regression cases for each user story.
- **CA-003 UX Consistency**: This feature MUST keep terminology and outcomes consistent across
  command-line and terminal UI flows. M1 contract-level parity MUST include: identical FR-005
  structured error fields, identical FR-007 redaction rules, and identical FR-003 canonical dry-run
  schema semantics across CLI and TUI; rendering/presentation may differ.
- **CA-004 Safety and Observability**: This feature MUST provide actionable operator diagnostics,
  clear failure attribution, and redacted reporting outputs.

### Key Entities *(include if feature involves data)*

- **Slot Assignment**: A mapping of slot identity to bind settings and active model selection.
- **M1 Config Schema**: Versioned configuration model (`schema_version: 1`) containing workstation
  defaults and slot-specific entries keyed by `slot_id` used for deterministic merge resolution.
- **Operational Profile**: Preset and override guidance used to resolve effective runtime behavior.
- **Launch Validation Result**: A per-slot outcome record indicating launch eligibility, blocking
  errors, warnings, and remediation text.
- **M1 Observability Artifact**: Per-run JSON record persisted in
  the resolved runtime artifact directory (`$LLM_RUNNER_RUNTIME_DIR/artifacts/` when set and
  usable, otherwise `$XDG_RUNTIME_DIR/llm-runner/artifacts/`) with timestamp + slot scope,
  containing resolved command, validation results, warnings, and redacted environment snapshot.
- **Slot Lock State**: Per-slot runtime lock state indicating lock presence, owner activity, and
  stale-lock cleanup action, with lockfile metadata (`pid`, `port`, `started_at`) and live-owner
  detection outcome.
- **Risk Acknowledgement Record**: A runtime confirmation state indicating the operator accepted
  a risky launch condition for the current run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid launch attempts (one-slot or two-slot) complete without manual command
  editing.
- **SC-002**: At least 95% of FR-005 launch-blocking validation outcomes produced in launch/dry-run
  paths during M1 acceptance tests return an actionable correction message on first failure,
  including all required FR-005 fields. The denominator for this metric includes all FR-005
  multi-error responses (with one or more `errors` array entries) and single-error responses
  produced during M1 acceptance test execution.
- **SC-003**: 100% of override resolution cases produce deterministic, operator-verifiable results,
  evidenced by: (a) identical human-readable dry-run output for identical inputs, (b) identical
  machine-parseable dry-run output for identical inputs, and (c) matching FR-007 artifact fields
  (`resolved_command`, `validation_results`, `warnings`) across repeated runs.
- **SC-004**: 100% of dry-run outputs include all FR-003 canonical schema required fields for each
  resolved slot and apply FR-007 redaction rules.
- **SC-005**: 100% of launch and dry-run attempts either (a) produce an FR-007 JSON observability
  artifact with required redacted fields, or (b) fail with FR-005
  `failed_check=artifact_persistence` before execution proceeds.
- **SC-006**: M1 performance budgets are met: dry-run resolution p95 ≤250 ms (single slot) / ≤400
  ms (two slots), and lock/port validation p95 ≤150 ms per slot. The p95 timing windows are
  measured over all completed dry-run and lock/port validation invocations during M1 acceptance
  tests, excluding outlier failures (timeouts, CI noise) that are recorded separately. Single-slot,
  two-slot, and per-slot lock/port timing windows are measured and reported independently.

## Assumptions

- The target persona is a solo operator running on the anchored workstation profile described in
  `PRD.md`.
- This feature intentionally narrows scope to PRD M1 launch and dry-run behavior; M0 documentation
  generation and M2-M4 capabilities are deferred to later specs.
- M0 documentation generation is tracked as a separate dependent spec and may be planned in parallel,
  but it is not implemented or validated by Spec-001 acceptance criteria.
- When one configured slot is unavailable, operators still need usable one-slot launch capability
  with explicit warnings.
- Users prefer explicit confirmations for risky actions over silent automation.
- Risk acknowledgement does not persist across separate launch attempts.
- Functional readiness for this feature is defined by successful launch and deterministic dry-run,
  not full operational validation workflows.

## Addendum — Definitions & Measurement Notes (M1)

### SC-002 Denominator Clarification

For SC-002's 95% actionable-error threshold, the denominator counts **all FR-005 error objects**
(individual `errors[n]` entries) produced across all launch/dry-run validation runs during M1
acceptance tests, not the number of launch attempts or response objects. When FR-005 returns a
multi-error response with N error entries, those N entries contribute N items to the denominator.
The numerator counts how many of those error entries include all required FR-005 fields
(`error_code`, `failed_check`, `why_blocked`, `how_to_fix`).

### FR-009 Lockfile Atomicity Expectation (Requirement Note)

This requirement expects lockfile operations to behave atomically with respect to launch-state
decisions:

- **Write**: A lockfile must be fully written with valid content (`pid`, `port`, `started_at`)
  before any other process can observe it.
- **Read**: Ownership checks must read a consistent snapshot; partial or corrupted reads must not be
  treated as stale.
- **Update**: When a lock is acquired, cleared, or refreshed, the operation must not interleave with
  another process's read in a way that produces ambiguous live/stale results.

This is stated as a correctness requirement, not an implementation mandate; implementations may use
OS-level atomic primitives or file-locked transactions as long as the observable behavior matches
the expectation.

### Canonical Terminology (M1)

| Term | Definition (M1 scope) |
| --- | --- |
| **warning** | A non-blocking diagnostic emitted when a configuration or runtime condition deviates from the expected healthy state but does not prevent launch (e.g., one-slot unavailable in a two-slot config). |
| **launch-blocking error** | An FR-005 error that prevents any slot from starting; includes both single-slot full-block and multi-slot cases where all configured slots are blocked. |
| **risky operation** | A launch condition requiring explicit operator acknowledgement in M1: port <1024, non-loopback bind, or a manual override that bypasses a warning. |
| **degraded launch** | A one-slot launch that proceeds when only one of two configured slots is available; the unavailable slot generates a warning but does not block the running slot. |
| **full-block launch** | A launch outcome where zero configured slots can start due to one or more launch-blocking errors; no slots are started in this state. |
| **current launch attempt** | The runtime context covering a single invocation of the launch/dry-run resolution path; risk acknowledgements and in-memory state are scoped to this attempt and do not persist across invocations. Synonymous with "session-only" in M1. |
