# Feature Specification: PRD-Aligned MVP Control Plane

**Feature Branch**: `001-prd-mvp-spec`  
**Created**: 2026-04-08  
**Status**: Draft  
**Input**: User description: "based on PRD.md"

## Clarifications

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
- Risk acknowledgements are non-persistent and apply only to the current launch attempt.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST enforce slot-first orchestration where each running workload is bound to
  a declared slot and each slot owns its bind address and port.
- **FR-002**: System MUST prevent invalid startup states, including duplicate slot assignment,
  missing model source, and conflicting network bindings.
- **FR-003**: System MUST provide a dry-run mode that presents resolved launch intent in
  operator-readable form before execution.
- **FR-004**: System MUST allow degraded one-slot startup when one configured slot is unavailable,
  and MUST emit a clear warning identifying the unavailable slot.
- **FR-005**: System MUST return launch-blocking errors with actionable correction guidance before
  startup proceeds.
- **FR-006**: System MUST apply deterministic override precedence in this order: defaults <
  slot/workstation config < profile guidance < explicit override.
- **FR-007**: System MUST preserve observability artifacts for launch and dry-run outcomes, with
  sensitive values redacted.
- **FR-008**: System MUST treat runtime safety as default behavior, requiring explicit acknowledgement
  for risky operations; acknowledgement is valid only for the current launch attempt.
- **FR-009**: System MUST auto-clear stale lockfiles when no active owner exists, and MUST block
  launch when a live lock owner is detected.
- **FR-010**: System MUST scope this feature to launch and dry-run core behavior; guided diagnostics,
  setup mutation flows, and smoke verification are deferred to follow-on specifications.

### Constitution Alignment *(mandatory)*

- **CA-001 Code Quality**: This feature MUST preserve core/CLI boundary integrity and pass all
  required static-quality gates before merge.
- **CA-002 Testing**: This feature MUST include deterministic automated tests for success paths,
  failure paths, and regression cases for each user story.
- **CA-003 UX Consistency**: This feature MUST keep terminology and outcomes consistent across
  command-line and terminal UI flows.
- **CA-004 Safety and Observability**: This feature MUST provide actionable operator diagnostics,
  clear failure attribution, and redacted reporting outputs.

### Key Entities *(include if feature involves data)*

- **Slot Assignment**: A mapping of slot identity to bind settings and active model selection.
- **Operational Profile**: Preset and override guidance used to resolve effective runtime behavior.
- **Launch Validation Result**: A per-slot outcome record indicating launch eligibility, blocking
  errors, warnings, and remediation text.
- **Slot Lock State**: Per-slot runtime lock state indicating lock presence, owner activity, and
  stale-lock cleanup action.
- **Risk Acknowledgement Record**: A runtime confirmation state indicating the operator accepted
  a risky launch condition for the current run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid launch attempts (one-slot or two-slot) complete without manual command
  editing.
- **SC-002**: At least 95% of launch-blocking failures return an actionable correction message on
  first failure.
- **SC-003**: 100% of override resolution cases produce deterministic, operator-verifiable results.
- **SC-004**: At least 90% of dry-run reviews are accepted by operators as sufficiently clear to
  proceed without additional clarification.

## Assumptions

- The target persona is a solo operator running on the anchored workstation profile described in
  `PRD.md`.
- This feature intentionally narrows scope to launch and dry-run behavior from the PRD; diagnostics,
  setup mutation flows, and smoke verification are deferred to later specs.
- When one configured slot is unavailable, operators still need usable one-slot launch capability
  with explicit warnings.
- Users prefer explicit confirmations for risky actions over silent automation.
- Risk acknowledgement does not persist across separate launch attempts.
- Functional readiness for this feature is defined by successful launch and deterministic dry-run,
  not full operational validation workflows.
