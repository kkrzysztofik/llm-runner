# PRD vs Spec-001 Compliance Review

**Review Date:** 2026-04-09  
**Reviewed By:** Architect Agent  
**PRD Version:** MVP draft v0.3  
**Spec Version:** specs/001-prd-mvp-spec/spec.md (Draft, Created 2026-04-08)  
**Review Scope:** Full compliance assessment of PRD requirements against MVP spec-001
**Current Branch:** `001-prd-mvp-spec` (implements M1 scope only)

---

## Executive Summary

**Critical Finding:** Spec-001 is intentionally scoped narrower than the PRD MVP definition. The spec explicitly defers diagnostics (doctor), setup mutation flows, smoke verification, build pipelines, profiling, TUI monitoring, graceful shutdown, GGUF metadata parsing, and hardware acknowledgment mechanisms to "follow-on specifications."

**Coverage Assessment:**

- **Functional Requirements:** 3/19 fully covered (FR-001, FR-009, FR-011*), 3/19 partially covered (FR-003, FR-011, FR-016), 13/19 missing (68%)
- **Non-Functional Requirements:** 0/7 fully covered, 4/7 partially covered (57%), 3/7 missing (43%)
- **Acceptance Criteria:** 1/20 fully covered (AC-001), 3/20 partially covered (AC-005, AC-009, AC-016*), 16/20 missing (80%)

**Note:** *FR-011 in spec-001 covers backend eligibility (vllm guard) but not full CLI scripting (exit codes, --json). AC-016* is partially covered via FR-008 risk acknowledgement but not full VRAM heuristics.

**Current Branch Status (`001-prd-mvp-spec`):**
- Implements **M1 Milestone only**: Slot-first orchestration, deterministic override precedence, basic validation, dry-run mode, lockfile handling, risk acknowledgement
- **NOT** full PRD MVP completion — deferred items span M0, M2, M3, M4 milestones
- Title recommendation: Rename from "PRD-Aligned MVP Control Plane" to "PRD M1: Slot-First Launch & Dry-Run" to avoid stakeholder confusion

**PRD-Deferred Scope (Remaining Work):**
- **M0:** Documentation generation (FR-019)
- **M2:** Build pipeline (FR-004, FR-006), Setup (FR-005)
- **M3:** Profiling (FR-007, FR-008)
- **M4:** Smoke (FR-015), TUI monitoring (FR-010), Shutdown (FR-012), GGUF parsing (FR-014), Hardware acknowledgment (FR-017), Logging (FR-018), CLI scripting (FR-011), Backend abstraction (FR-002), Config schema (FR-013)

---

## A) Coverage Mapping

### Functional Requirements (FR-001..FR-019)

| ID | Status | Evidence/Notes |
| -- | ------ | -------------- |
| FR-001 | Covered | Spec FR-001: "System MUST enforce slot-first orchestration where each running workload is bound to a declared slot and each slot owns its bind address and port." |
| FR-002 | Missing | Spec does not mention backend selection, vLLM guards, or experimental gates. User Story 1-3 are slot-launch focused only. |
| FR-003 | Partially covered | Spec FR-003: "System MUST provide a dry-run mode that presents resolved launch intent in operator-readable form before execution." Missing: OpenAI flag bundle, vLLM matrix, merged env details, hardware notes. Spec explicitly defers "dry-run smoke" to follow-on specs. |
| FR-004 | Missing | Spec Section 1 (User Scenarios) and FR-010 explicitly state: "guided diagnostics, setup mutation flows, and smoke verification are deferred to follow-on specifications." Build pipeline is part of M2, not M1. |
| FR-005 | Missing | Toolchain/setup deferral noted in FR-010. No venv handling mentioned. |
| FR-006 | Missing | Provenance is part of M2 build wizard. Not addressed in spec. |
| FR-007 | Missing | Profiling explicitly deferred to M3. |
| FR-008 | Missing | Profiling persistence deferred. |
| FR-009 | Partially covered | Spec FR-006: "System MUST apply deterministic override precedence in this order: defaults < slot/workstation config < profile guidance < explicit override." Matches PRD. Missing: profile guidance integration (since profiling is deferred). |
| FR-010 | Missing | TUI monitoring is part of M4. Spec does not address TUI beyond "operator-readable" dry-run. |
| FR-011 | Partially covered | Spec does not mention exit codes, --json health/status, or doctor/smoke commands (all deferred). |
| FR-012 | Missing | Shutdown behavior not addressed. |
| FR-013 | Partially covered | Spec mentions "slot/workstation config" in FR-006 and FR-001 but does not address schema_version, merge rules, backup-on-migrate, or validation failures. |
| FR-014 | Missing | GGUF handling is not addressed. Part of M4 operational hardening. |
| FR-015 | Missing | Smoke is explicitly deferred in FR-010. Entire M4 "Operational hardening + smoke" milestone. |
| FR-016 | Partially covered | Spec FR-009: "System MUST auto-clear stale lockfiles when no active owner exists, and MUST block launch when a live lock owner is detected." Spec FR-002 mentions "conflicting network bindings" blocked. Missing: --force-bind mechanism, VRAM heuristics, lockfile paths, doctor --locks. |
| FR-017 | Missing | Spec mentions "one configured slot is unavailable" in User Story 1 Scenario 3, but this is about unavailable slots, not hardware topology warnings. No allowlist, session snooze, or --ack-nonstandard-hardware mentioned. |
| FR-018 | Partially covered | Spec FR-007: "System MUST preserve observability artifacts for launch and dry-run outcomes, with sensitive values redacted." Missing: rotating logs, report directory structure, mutation logging, secret redaction patterns. |
| FR-019 | Missing | Documentation generation not addressed. Part of M0 milestone. |

### Non-Functional Requirements (NFR-001..NFR-007)

| ID | Status | Evidence/Notes |
| -- | ------ | -------------- |
| NFR-001 | Partially covered | Spec FR-002: "System MUST prevent invalid startup states, including duplicate slot assignment, missing model source, and conflicting network bindings." Matches duplicate slot, missing model, port conflict. Missing: vLLM guard, GGUF validation, VRAM confirmation. |
| NFR-002 | Partially covered | Spec CA-004: "This feature MUST provide actionable operator diagnostics, clear failure attribution, and redacted reporting outputs." Missing: slot->model->process->phase hierarchy. |
| NFR-003 | Missing | Performance expectations not addressed. |
| NFR-004 | Partially covered | Spec FR-006 addresses deterministic override precedence. Missing: artifact SHA recording, reproducibility requirements. |
| NFR-005 | Missing | Runtime safety guarantees not addressed. |
| NFR-006 | Missing | TUI performance and monitoring defaults not addressed. |
| NFR-007 | Partially covered | Spec FR-007 mentions "sensitive values redacted" but not path redaction specifically. |

### Acceptance Criteria (AC-001..AC-020)

| ID | Status | Evidence/Notes |
| -- | ------ | -------------- |
| AC-001 | Covered | User Story 1 Scenario 2: "Given a duplicate slot assignment or occupied port, When I run launch, Then the system blocks startup and returns an actionable error." |
| AC-002 | Missing | Spec FR-003 only requires "resolved launch intent in operator-readable form." Missing: OpenAI flag bundle, vLLM matrix row, merged env, device mapping. |
| AC-003 | Missing | Backend selection (FR-002) not addressed. Doctor command deferred. |
| AC-004 | Missing | Hardware acknowledgment (FR-017) not addressed. Spec mentions "unavailable slots" but not hardware topology warnings. |
| AC-005 | Partially covered | Spec FR-002 mentions "conflicting network bindings" blocked. Missing: --force-bind mechanism. |
| AC-006 | Missing | Build pipeline (FR-004) explicitly deferred to follow-on specs. |
| AC-007 | Missing | Toolchain/setup (FR-005) explicitly deferred. |
| AC-008 | Missing | Profiling (FR-007, FR-008) explicitly deferred to M3. |
| AC-009 | Partially covered | Spec FR-006 defines deterministic precedence. Missing: explicit test coverage requirements. |
| AC-010 | Missing | GGUF handling (FR-014) not addressed. |
| AC-011 | Missing | TUI monitoring (FR-010) explicitly deferred to M4. |
| AC-012 | Missing | Shutdown (FR-012) not addressed. |
| AC-013 | Missing | Runtime error handling (part of M4) not addressed. |
| AC-014 | Missing | Smoke (FR-015) explicitly deferred to M4. |
| AC-015 | Missing | Documentation generation (FR-019) not addressed. |
| AC-016 | Missing | VRAM heuristics (FR-016) not addressed. |
| AC-017 | Missing | Smoke behavior not addressed. |
| AC-018 | Missing | GGUF fixtures (part of FR-014) not addressed. |
| AC-019 | Missing | Setup venv (FR-005) explicitly deferred. |
| AC-020 | Missing | OpenAI flag bundle (part of FR-003) not addressed. |

---

## B) Gap Report

### Major Gaps (PRD Items Not in Spec)

1. **Backend Abstraction & vLLM Guard (FR-002, AC-002, AC-003)**
   - Spec-001 does not mention backend selection, vLLM guards, or experimental gates.
   - PRD Section 1, 5 requires backend abstraction as MVP deliverable.
   - Impact: Spec-001 title claims "PRD-Aligned MVP" but omits core MVP requirement.

2. **Hardware Acknowledgment Flow (FR-017, AC-004)**
   - Spec-001 mentions "unavailable slots" but not hardware topology warnings, allowlist persistence, session snooze, or --ack-nonstandard-hardware.
   - PRD Section 10(AC-004) requires full hardware warning path as MVP completion gate.
   - Spec-001 only addresses slot unavailability (configuration), not hardware mismatch.

3. **GGUF Metadata Parsing (FR-014, AC-010, AC-018)**
   - Spec-001 does not mention GGUF parsing, prefix read caps, parse timeouts, filename normalization, or test fixtures.
   - PRD Section 5 explicitly states "GGUF-only models in MVP" with detailed parsing requirements.
   - Core MVP feature, not post-MVP.

4. **Smoke Verification (FR-015, AC-014, AC-017)**
   - Spec-001 explicitly defers smoke to "follow-on specifications."
   - PRD Section 10(AC-014) states "smoke both is authoritative MVP completion gate."
   - Direct contradiction: Spec-001 defers what PRD says is MVP-critical.

5. **Build Pipeline & Provenance (FR-004, FR-006, AC-006)**
   - Spec-001 defers TUI build pipeline to follow-on specs.
   - PRD Section 5 states "TUI-driven llama.cpp source build workflow... is in scope" for MVP.
   - Direct contradiction with PRD.

6. **Profiling & Presets (FR-007, FR-008, AC-008)**
   - Spec-001 defers profiling to follow-on specs.
   - PRD Section 5 states "Manual (TUI-triggered) profiling and persisted preset guidance" is in scope.
   - Direct contradiction with PRD.

7. **Graceful Shutdown (FR-012, AC-012)**
   - Spec-001 does not address shutdown behavior.
   - PRD Section 7(FR-012) requires graceful SIGTERM escalation and orphan process avoidance.

8. **TUI Monitoring (FR-010, AC-011)**
   - Spec-001 defers TUI monitoring to follow-on specs.
   - PRD Section 7(FR-010) requires TUI to show per-slot status, logs, GPU telemetry, backend, build job state, profile warnings, UNSAVED badge.

9. **CLI Scripting & Exit Codes (FR-011, AC-014)**
   - Spec-001 does not mention CLI scripting, --json output, or exit code conventions.
   - PRD Section 7(FR-011) requires stable --json for health/status and documented exit codes.

10. **Config Schema & Migration (FR-013)**
    - Spec-001 mentions "slot/workstation config" but not schema_version, merge rules, backup-on-migrate, or validation failure behavior.

11. **Documentation Generation (FR-019, AC-015)**
    - Spec-001 does not mention gendoc.py or README excerpt markers.

12. **Setup & venv Isolation (FR-005, AC-019)**
    - Spec-001 explicitly defers setup mutation flows.
    - PRD Section 7(FR-005) requires setup to create/reuse dedicated venv.

13. **Logging & Report Directories (FR-018, AC-007)**
    - Spec-001 mentions "sensitive values redacted" but not rotating logs, report directories, or mutation logging.

### Contradictions

| Issue | PRD Stance | Spec Stance |
| ----- | ---------- | ----------- |
| Scope definition | MVP includes: slot-first orchestration, TUI-driven builds, doctor/setup/smoke, manual profiling, GGUF parsing, TUI monitoring, graceful shutdown, hardware acknowledgment, smoke as MVP gate | Spec claims "PRD-Aligned MVP Control Plane" but explicitly defers: doctor, setup, smoke, build pipeline, profiling, TUI monitoring, shutdown, GGUF parsing, hardware acknowledgment |
| Smoke as MVP gate | Section 10(AC-014): "smoke both is authoritative MVP completion gate" | FR-010: "smoke verification are deferred to follow-on specifications" |
| Build pipeline | Section 5: "TUI-driven llama.cpp source build workflow... is in scope" | FR-010: build pipeline deferred |
| Profiling | Section 5: "Manual (TUI-triggered) profiling" is in scope | FR-010: profiling deferred |

### Ambiguities That Block Implementation

1. **"Operator-readable form"** (FR-003): What fields must dry-run show? Spec-001's requirement is too vague for testing.

2. **"Resolved launch intent"** (FR-003): What constitutes "resolved"? Defaults? Merged? Effective? Spec-001 doesn't define resolution algorithm or output format.

3. **"Slot/workstation config"** (FR-001, FR-006): What is a "workstation config" vs "slot config"? Spec-001 doesn't define config structure or inheritance model.

4. **"Risk acknowledgement"** (FR-008): What constitutes a "risky operation"? Spec-001 doesn't define risk categories or acknowledgement UX.

5. **"Observability artifacts"** (FR-007): What artifacts? Where stored? What format? Spec-001 doesn't define.

6. **"Sensitive values redacted"** (FR-007): What is "sensitive"? API keys? Paths? Model IDs? Spec-001 doesn't define redaction policy.

7. **"Live lock owner detected"** (FR-009): How is "live owner" detected? PID file? Network socket check? Spec-001 doesn't define lockfile mechanism.

8. **"Actionable correction message"** (SC-002): What makes a message "actionable"? Spec-001 doesn't define message format or required elements.

---

## C) Extension Recommendations

### P0 — Must Have for PRD Compliance

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| P0-01 | Rename spec title from "PRD-Aligned MVP Control Plane" to "PRD M1: Slot-First Launch & Dry-Run" | Current title misrepresents scope. Spec-001 only covers ~20% of MVP. Honesty prevents stakeholder confusion. |
| P0-02 | Add FR-020 (Backend abstraction): System MUST support backend selection (llama_cpp/vllm) via config; vllm MUST fail doctor unless experimental gate is set | PRD Section 1, 5 requires backend abstraction as MVP deliverable. Spec-001 omits entirely. |
| P0-03 | Expand FR-003 (Dry-run): Add requirement to show: binary path, exact command line, model path, slot, merged env, OpenAI flag bundle, effective ports, hardware notes, vLLM matrix row | PRD Section 7(FR-003) is explicit; spec-001's "operator-readable form" too vague for testing. |
| P0-04 | Add FR-021 (Hardware acknowledgment): System MUST warn on non-anchor hardware topology; require TUI Continue or CLI --ack-nonstandard-hardware; persist allowlist; implement session snooze | PRD Section 7(FR-017), 10(AC-004) requires full hardware warning path. |
| P0-05 | Add FR-022 (GGUF parsing): System MUST parse GGUF metadata with 32 MiB prefix cap, 5-second parse timeout; support unsharded files; filename normalization; repo includes synthetic test fixtures | PRD Section 5, 7(FR-014), 10(AC-010, AC-018) requires GGUF handling as MVP. |
| P0-06 | Clarify "slot/workstation config" in FR-001/FR-006: define schema with schema_version:1, model_id keys, slot keys, merge rules (user wins), inheritance model | Spec-001's "slot/workstation config" ambiguous. PRD Section 7(FR-013) is explicit. |

### P1 — Strongly Recommended

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| P1-01 | Add FR-023 (Graceful shutdown): System MUST handle SIGTERM with wait-and-escalate; abnormal TUI exit MUST terminate child servers | PRD Section 7(FR-012), 10(AC-012) requires shutdown handling. |
| P1-02 | Add FR-024 (Logging & reports): System MUST append mutating actions to rotating log with command/timestamp/exit code; redact secrets; drop report directory on failure | PRD Section 7(FR-018), 10(AC-007) requires detailed logging. |
| P1-03 | Add FR-025 (Exit codes): doctor MUST return 0/1/2; smoke MUST return 1/2/3; document all exit codes in CLI help | PRD Section 7(FR-011), Appendix B require exit code conventions. |
| P1-04 | Add FR-026 (VRAM heuristics): System MUST compare free VRAM against heuristic footprint; warn and require explicit confirm; not hard block except impossible headroom | PRD Section 7(FR-016), 10(AC-016) require VRAM warnings. |
| P1-05 | Add FR-027 (Port conflict handling): System MUST refuse port-in-use unless explicit --force-bind (logged dangerous); define per-slot lockfiles; doctor --locks surfaces holders | PRD Section 7(FR-016) requires detailed port/lock handling. |
| P1-06 | Add NFR-008 (Performance): Dry-run/config resolve MUST feel immediate (<100ms); long operations MUST show phase progress with ETA | PRD Section NFR-003 requires performance expectations. |
| P1-07 | Expand Success Criteria: SC-004 is unverifiable; replace with "Dry-run output includes all fields specified in FR-003 for 100% of cases." | SC-004 vague; PRD Section 7(FR-003) defines exact dry-run fields. |

### P2 — Nice-to-Have

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| P2-01 | Add FR-028 (Documentation generation): System supports gendoc.py to extract <!-- readme:... --> marker sections from PRD into README | PRD Section 7(FR-019), 10(AC-015), Appendix C require gendoc.py. |
| P2-02 | Add FR-029 (TUI monitoring): System MUST show per-slot status, logs, GPU telemetry, backend, build job state, profile warnings, UNSAVED badge | PRD Section 7(FR-010), 10(AC-011) require TUI monitoring. |
| P2-03 | Add FR-030 (Build pipeline & provenance): System MUST allow TUI-driven serialized llama.cpp builds, record remote URL + master tip SHA + timestamps, doctor --repair clears failed staging | PRD Section 5, 7(FR-004, FR-006), 10(AC-006) require build pipeline. |

---

## D) Proposed Docs Artifact Outline

**Title:** "PRD Spec-001 Coverage Summary"

**Structure:**

1. **Overview** (1 paragraph)
   - What spec-001 covers vs PRD MVP scope
   - Key takeaway: spec-001 = M1 only, not full MVP

2. **Coverage Matrix** (table)
   - Three-column table: Requirement | Spec Status | Evidence Link
   - Grouped by FR/NFR/AC sections

3. **What's Included** (bulleted list)
   - Slot-first orchestration (FR-001, AC-001)
   - Deterministic override precedence (FR-009, NFR-004)
   - Basic validation (FR-002, NFR-001)
   - Dry-run (FR-003, partial)
   - Lockfile handling (FR-009)
   - Risk acknowledgement (FR-008)

4. **What's Deferred** (bulleted list, with milestone mapping)
   - M0: Documentation generation (FR-019)
   - M1: Backend abstraction (FR-002), Config schema (FR-013) - partial
   - M2: Build pipeline (FR-004, FR-006), Setup (FR-005)
   - M3: Profiling (FR-007, FR-008)
   - M4: Smoke (FR-015), TUI monitoring (FR-010), Shutdown (FR-012), GGUF (FR-014), Hardware ack (FR-017), Logging (FR-018), CLI scripting (FR-011)

5. **Critical Gaps** (table)
   - Gap | Blocker? | Impact | Recommendation

6. **Implementation Path** (timeline view)
   - Current state: Spec-001 ready for M1 implementation
   - Next steps: Address P0 recommendations for MVP compliance
   - Long-term: M2-M4 specs to cover deferred items

7. **References**
   - PRD.md sections
   - Spec-001 sections
   - PRD milestone roadmap (Section 11)

---

## D) CI Quality Gates Note

All plans must address the following CI gates:

1. **ruff check** - All new FR/NFR definitions must follow existing code style
2. **ruff format --check** - Ensure consistent formatting
3. **pyright** - Type annotations on all new requirements
4. **pytest** - Test coverage requirements for each FR must be specified

**Specific concerns:**

- New FR definitions need testable acceptance criteria
- Success criteria must be measurable (avoid "sufficiently clear", "operator-verifiable" without definition)
- Any new dataclasses (e.g., HardwareWarning, GGUFMetadata, SmokeResult) need type annotations

---

## E) Current Branch Status Summary

**Branch:** `001-prd-mvp-spec`  
**Status:** Implements PRD Milestone M1 only — **NOT full PRD MVP**

### What's Implemented (M1 Scope)

| Category | Items |
| -------- | ----- |
| Core Orchestration | FR-001: Slot-first orchestration |
| Validation | FR-002 (partial), NFR-001 (partial) |
| Dry-Run | FR-003 (partial): Basic operator-readable output |
| Precedence | FR-009, NFR-004: Deterministic override precedence |
| Lockfiles | FR-009: Stale lock clearing, live lock blocking |
| Risk Mgmt | FR-008: Risk acknowledgement mechanism |

### What's Deferred (M0, M2, M3, M4)

| Milestone | Deferred Items |
| --------- | -------------- |
| M0 | FR-019: Documentation generation (gendoc.py) |
| M2 | FR-004: Build pipeline, FR-006: Provenance, FR-005: Setup/venv |
| M3 | FR-007, FR-008: Profiling & presets, GGUF parsing (FR-014) |
| M4 | FR-015: Smoke verification, FR-010: TUI monitoring, FR-012: Graceful shutdown, FR-017: Hardware acknowledgment, FR-018: Logging & reports, FR-011: CLI scripting, FR-002: Backend abstraction, FR-013: Config schema |

### Key Takeaway

**This branch represents a milestone delivery, not MVP completion.** The PRD's MVP includes features spanning M0-M4, but `001-prd-mvp-spec` only implements M1. Stakeholders should not assume full PRD compliance based on this branch alone.

**Recommendation:** Update spec title to "PRD M1: Slot-First Launch & Dry-Run" to accurately reflect scope and prevent confusion.
