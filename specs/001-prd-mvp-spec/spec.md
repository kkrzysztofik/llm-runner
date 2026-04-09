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
- Artifact persistence failures (disk full, I/O error, permission denied after runtime-dir
  resolution) are launch-blocking and return FR-005 actionable errors with
  `failed_check=artifact_persistence`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST enforce slot-first orchestration where each running workload is bound to
  a declared slot and each slot owns its bind address and port.
- **FR-002**: System MUST prevent invalid startup states, including duplicate slot assignment,
  missing model source, and conflicting network bindings.
- **FR-003**: System MUST provide a deterministic dry-run mode that, before execution, prints for
  each slot: binary path, exact command arguments, model path, slot ID, effective bind/port,
  merged environment values with sensitive values redacted, OpenAI flag bundle, hardware notes,
  and a vLLM matrix row indicating launch eligibility in the current mode. Dry-run MUST conform to
  a canonical schema with required fields: `slot_id`, `binary_path`, `command_args`, `model_path`,
  `bind_address`, `port`, `environment_redacted`, `openai_flag_bundle`, `hardware_notes`,
  `vllm_eligibility`, `warnings`, and `validation_results`. M1 MAY present this as human-readable
  output, machine-parseable output, or both, but all representations MUST derive from the same
  canonical schema. M1 dry-run resolution latency target is p95 ≤250 ms for single-slot resolution
  and p95 ≤400 ms for two-slot resolution.
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
  precedence resolution, user override wins per field.
- **FR-007**: System MUST preserve observability artifacts for launch and dry-run outcomes. Redaction
  MUST replace values with `[REDACTED]` for any key name containing `KEY`, `TOKEN`, `SECRET`,
  `PASSWORD`, or `AUTH` (case-insensitive). Filesystem paths MUST remain visible in M1. M1
  artifacts MUST be persisted as one JSON file per launch/dry-run under the resolved runtime
  artifact directory (`$LLM_RUNNER_RUNTIME_DIR/artifacts/` when set and usable, otherwise
  `$XDG_RUNTIME_DIR/llm-runner/artifacts/`) with timestamp + slot scope, containing resolved command,
  validation results, warnings, and a redacted environment snapshot. Artifact files MUST be created
  with owner-only permissions (`0600`) and their runtime directories with owner-only permissions
  (`0700`); inability to enforce these permissions MUST return a launch-blocking actionable error.
  Artifact persistence failures (disk full, I/O error, permission denied after path resolution) MUST
  be launch-blocking and return FR-005 actionable error with
  `failed_check=artifact_persistence`.
- **FR-008**: System MUST treat runtime safety as default behavior, requiring explicit acknowledgement
  for risky operations; acknowledgement is valid only for the current launch attempt. In M1, risky
  operations are: port <1024, non-loopback bind, and manual override that bypasses a warning.
- **FR-009**: System MUST auto-clear stale lockfiles when no active owner exists, and MUST block
  launch when a live lock owner is detected. Lockfiles MUST be per-slot under the resolved runtime
  lock directory (`$LLM_RUNNER_RUNTIME_DIR/` when set and usable, otherwise
  `$XDG_RUNTIME_DIR/llm-runner/`) at `slot-{slot_id}.lock` and include `pid`, `port`, `started_at`; a
  lock owner is live only when the PID exists and still owns the expected port. Lockfiles MUST use
  owner-only permissions (`0600`), and parent runtime directories MUST use owner-only permissions
  (`0700`). If lockfile integrity validation fails (malformed/unreadable content) or stale-owner
  verification cannot be completed, launch MUST be blocked and return an FR-005 actionable error
  with `failed_check=lockfile_integrity`. Live-owner verification MUST be evaluated as one atomic
  decision (`pid exists` + `owns expected port`); indeterminate verification states are
  launch-blocking. Lock/port validation latency target is p95 ≤150 ms per slot.
- **FR-010**: System MUST scope this feature to PRD M1 launch and dry-run core behavior; non-M1 work
  (including M0 documentation generation and M2-M4 capabilities) is deferred to separate follow-on
  specifications, and this spec excludes M0 acceptance criteria.
- **FR-011**: System MUST accept backend values in configuration for forward compatibility, but in
  PRD M1 only `llama_cpp` is launch-eligible; `vllm` MUST be surfaced as non-eligible and blocked
  with an actionable correction message. For blocked `vllm`, FR-005 response MUST include
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
  including all required FR-005 fields.
- **SC-003**: 100% of override resolution cases produce deterministic, operator-verifiable results,
  evidenced by identical dry-run output for identical inputs and matching FR-007 artifact fields
  (`resolved_command`, `validation_results`, `warnings`) across repeated runs.
- **SC-004**: 100% of dry-run outputs include all FR-003 canonical schema required fields for each
  resolved slot and apply FR-007 redaction rules.
- **SC-005**: 100% of launch and dry-run attempts either (a) produce an FR-007 JSON observability
  artifact with required redacted fields, or (b) fail with FR-005
  `failed_check=artifact_persistence` before execution proceeds.
- **SC-006**: M1 performance budgets are met: dry-run resolution p95 ≤250 ms (single slot) / ≤400
  ms (two slots), and lock/port validation p95 ≤150 ms per slot.

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
