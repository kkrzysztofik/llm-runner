# Spec-001 PRD Coverage Analysis

**Document Purpose:** This document provides a concise but complete mapping of what PRD.md requirements are covered by `specs/001-prd-mvp-spec/spec.md`, including coverage percentages, gap analysis, and recommended follow-on work.

**Key Finding:** Spec-001 intentionally scopes narrower than the PRD MVP definition. It covers **Milestone M1 only** (slot-first config + backend abstraction + validation), not the full MVP. The spec title "PRD-Aligned MVP Control Plane" is misleading and should be renamed to "PRD M1: Slot-First Launch & Dry-Run".

---

## Executive Summary

Spec-001 covers the core slot-first orchestration, validation, and dry-run capabilities (M1) but explicitly defers diagnostics (doctor), setup mutation flows, smoke verification, build pipelines, profiling, TUI monitoring, graceful shutdown, GGUF metadata parsing, hardware acknowledgment mechanisms, and CLI scripting to follow-on specifications. The spec represents approximately **16% of functional requirements**, **57% of non-functional requirements** (partially), and **20% of acceptance criteria** fully covered. Implementation teams should treat Spec-001 as an M1 milestone deliverable, not a complete MVP specification.

---

## Coverage Snapshot

| Category | Fully Covered | Partially Covered | Missing | % Coverage |
| -------- | ------------- | ----------------- | ------- | ---------- |
| **Functional Requirements** | 3/19 (16%) | 3/19 (16%) | 13/19 (68%) | 16% |
| **Non-Functional Requirements** | 0/7 (0%) | 4/7 (57%) | 3/7 (43%) | 57%* |
| **Acceptance Criteria** | 1/20 (5%) | 3/20 (15%) | 16/20 (80%) | 5% |
| **Overall MVP Coverage** | | | | **~15%** |

*\*NFR coverage is "partial" because spec-001 addresses some validation and observability patterns but misses performance, safety, and TUI-specific requirements.*

---

## What Spec-001 Covers Well

Spec-001 provides a solid foundation for the **M1 milestone** with clear requirements in these areas:

### Slot-First Orchestration

- **FR-001** (Spec): System MUST enforce slot-first orchestration where each running workload is bound to a declared slot and each slot owns its bind address and port.
- **AC-001** (PRD): Two models launch on distinct slots with slot-owned ports; duplicate slot assignment rejected.
- **Evidence:** User Story 1 (Launch Models by Slot) provides comprehensive scenarios for slot assignment validation.

### Deterministic Override Precedence

- **FR-006** (Spec): System MUST apply deterministic override precedence in this order: defaults < slot/workstation config < profile guidance < explicit override.
- **FR-009** (PRD): Explicit CLI/config overrides always win over preset defaults and profile guidance in a documented, deterministic order.
- **AC-009** (PRD): Override precedence covered by tests.
- **Evidence:** User Story 3 (Use Deterministic Overrides Safely) covers precedence and risk prompts.

### Basic Validation

- **FR-002** (Spec): System MUST prevent invalid startup states, including duplicate slot assignment, missing model source, and conflicting network bindings.
- **NFR-001** (PRD): Validation prevents duplicate slot assignment, invalid ports, missing models.
- **AC-005** (PRD): Port listener conflict → actionable error unless --force-bind.
- **Evidence:** User Story 2 (Resolve Launch Blocking Errors Early) covers duplicate slots and port conflicts.

### Lockfile Handling

- **FR-009** (Spec): System MUST auto-clear stale lockfiles when no active owner exists, and MUST block launch when a live lock owner is detected.
- **FR-016** (PRD): Per-slot lockfiles under runtime dir; doctor --locks surfaces holders.
- **Evidence:** User Story 1 Scenario 4 (stale lockfile handling).

### Risk Acknowledgement (Session-Only)

- **FR-008** (Spec): System MUST treat runtime safety as default behavior, requiring explicit acknowledgement for risky operations; acknowledgement is valid only for the current launch attempt.
- **Evidence:** User Story 3 Scenario 2 (risky conditions require explicit acknowledgement).

### Dry-Run (Core Only)

- **FR-003** (Spec): System MUST provide a dry-run mode that presents resolved launch intent in operator-readable form before execution.
- **Partial:** Spec defers OpenAI flag bundle, vLLM matrix, merged env details, hardware notes to follow-on specs.

### Observability Artifacts

- **FR-007** (Spec): System MUST preserve observability artifacts for launch and dry-run outcomes, with sensitive values redacted.
- **Partial:** Spec mentions "sensitive values redacted" but not rotating logs, report directories, or mutation logging (deferred to M4).

---

## Partially Covered

The following PRD requirements are partially addressed in Spec-001 but lack implementation details or are explicitly deferred:

| PRD Item | Spec-001 Coverage | Gap |
| -------- | ----------------- | --- |
| **FR-003** (Dry-run) | FR-003 requires "operator-readable form" | Missing: OpenAI flag bundle, vLLM matrix row, merged env details, hardware notes, device mapping. Spec defers "dry-run smoke" to follow-on specs. |
| **FR-009** (Override precedence) | FR-006 defines deterministic order | Missing: profile guidance integration (profiling is deferred to M3). |
| **FR-011** (CLI scripting) | Not addressed | Spec defers to follow-on specs; no --json, exit codes, or doctor/smoke commands. |
| **FR-013** (Config merge) | Mentions "slot/workstation config" | Missing: schema_version:1, merge rules by model_id/slot keys, backup-on-migrate, validation failures. |
| **FR-016** (Port/locks/VRAM) | FR-009 (locks) + FR-002 (port conflicts) | Missing: --force-bind mechanism, VRAM heuristics, lockfile paths, doctor --locks. |
| **FR-018** (Logging/reports) | FR-007 mentions "sensitive values redacted" | Missing: rotating logs, report directory structure, mutation logging, secret redaction patterns. |
| **NFR-001** (Validation) | FR-002 covers duplicate slot, missing model, port conflicts | Missing: vLLM guard, GGUF validation, VRAM confirmation. |
| **NFR-002** (Failure attribution) | CA-004 requires "actionable operator diagnostics" | Missing: slot→model→process→phase hierarchy. |
| **NFR-004** (Determinism) | FR-006 defines precedence | Missing: artifact SHA recording, reproducibility requirements. |
| **NFR-007** (Path redaction) | FR-007 mentions "sensitive values redacted" | Missing: --redact-paths flag, path redaction policy. |

---

## Not Covered (Needs Follow-on Specs)

Spec-001 explicitly defers the following PRD requirements to follow-on specifications. These map to M2, M3, and M4 milestones in the PRD roadmap.

### M0: Documentation Generation

| PRD Item | Description |
| -------- | ----------- |
| **FR-019** (gendoc.py) | Extract <!-- readme:... --> marker sections from PRD into README. |

### M1: Backend Abstraction & Config Schema (Partially Missing)

| PRD Item | Description |
| -------- | ----------- |
| **FR-002** (Backend selection) | MVP supports llama_cpp only; vllm may appear as parsed value but must not launch without experimental gate. Spec-001 omits backend selection entirely. |

### M2: Build Wizard & Setup

| PRD Item | Description |
| -------- | ----------- |
| **FR-004** (TUI build pipeline) | Serialized builds for intel-sycl and nvidia-cuda targets with preflight, progress, success/failure state. |
| **FR-005** (Toolchain & setup) | setup creates/reuses dedicated venv at $XDG_CACHE_HOME/llm-runner/venv; mutating steps only under confirmatory UX. |
| **FR-006** (Artifacts & provenance) | Built artifacts at predictable paths; each build records remote URL, master tip SHA, timestamps. |
| **AC-006** (Build artifacts) | TUI produces SYCL + CUDA artifacts with recorded SHAs; serialized execution enforced. |
| **AC-007** (Toolchain errors) | Toolchain missing → clear install guidance; setup log captures failures with redaction. |
| **AC-019** (Setup venv) | setup venv path created/verified; no serve-time pip drift. |

### M3: Profiling & Presets

| PRD Item | Description |
| -------- | ----------- |
| **FR-007** (Manual profiling) | User triggers profiling from TUI; tool invokes native bench tools as subprocesses. |
| **FR-008** (Profile persistence) | Profiles persist under ~/.cache/llm-runner/profiles/ keyed by GPU identifiers + backend + flavor. |
| **AC-008** (Profile cache) | Manual profile persists cache; staleness surfaces in TUI + doctor. |

### M4: Operational Hardening & Smoke

| PRD Item | Description |
| -------- | ----------- |
| **FR-010** (TUI monitoring) | TUI shows per-slot status, logs, GPU telemetry, backend (llama_cpp), build job state, profile warnings, UNSAVED badge. |
| **FR-011** (CLI scripting) | Stable --json for health/status; doctor, setup, smoke with documented exit codes. |
| **FR-012** (Graceful shutdown) | SIGTERM, wait, escalate; abnormal TUI exit must terminate child servers. |
| **FR-014** (GGUF metadata) | Parse metadata with pinned gguf PyPI dependency; 32 MiB prefix cap, 5-second parse timeout; filename normalization; test fixtures. |
| **FR-015** (Smoke OpenAI-compatible) | smoke both (MVP completion gate), smoke slot <slot_id>; listen → /v1/models → minimal chat completion; OpenAI flag bundle with chat template for Qwen-class. |
| **FR-017** (Hardware acknowledgment) | Non-anchor hardware → warn (not hard-fail); --ack-nonstandard-hardware; persistent allowlist; session snooze. |
| **FR-018** (Logging & reports) | Rotating log with command/timestamp/exit code; failure drops report directory. |
| **AC-002** (Dry-run details) | dry-run shows binary, args, model path, device mapping, merged env, OpenAI flag bundle, matrix row for vllm not enabled. |
| **AC-003** (vllm guard) | vllm in config fails doctor (or experimental gate only). |
| **AC-004** (Hardware warning) | Non-anchor topology warns; allowlist, session snooze; single-GPU warns and does not fake second running model. |
| **AC-012** (Graceful shutdown) | Graceful shutdown and abnormal exit both end without orphan servers. |
| **AC-013** (Isolated crash) | Isolated crash/OOM highlights failed slot without killing sibling. |
| **AC-014** (Smoke MVP gate) | smoke both is authoritative MVP gate; sequential probes pass; distinct exit code family from doctor. |
| **AC-015** (gendoc.py) | gendoc.py extracts marker sections from PRD into README. |
| **AC-016** (VRAM heuristic) | VRAM heuristic warns when free memory vs estimated load is risky; launch/smoke does not proceed without explicit confirmation. |
| **AC-017** (Smoke behavior) | smoke both and smoke slot behave per FR-015 and Appendix D (ordering, delays, timeouts, max_tokens). |
| **AC-018** (GGUF fixtures) | Committed synthetic GGUF fixtures + maintainer generator script documented; CI uses static fixtures only. |
| **AC-020** (OpenAI flag bundle) | dry-run for anchored models includes Qwen-class template/jinja flags in printed OpenAI flag bundle. |

---

## Contradictions / Scope Mismatch with PRD MVP

| Issue | PRD Stance | Spec-001 Stance | Severity |
| ----- | ---------- | --------------- | -------- |
| **Scope definition** | MVP includes: slot-first orchestration, TUI-driven builds, doctor/setup/smoke, manual profiling, GGUF parsing, TUI monitoring, graceful shutdown, hardware acknowledgment, smoke as MVP gate | Spec claims "PRD-Aligned MVP Control Plane" but explicitly defers: doctor, setup, smoke, build pipeline, profiling, TUI monitoring, shutdown, GGUF parsing, hardware acknowledgment | **High** – Misleading title |
| **Smoke as MVP gate** | Section 10(AC-014): "smoke both is authoritative MVP completion gate" | FR-010: "smoke verification are deferred to follow-on specifications" | **Critical** – Direct contradiction |
| **Build pipeline** | Section 5: "TUI-driven llama.cpp source build workflow... is in scope" | FR-010: build pipeline deferred to follow-on specs | **High** – Direct contradiction |
| **Profiling** | Section 5: "Manual (TUI-triggered) profiling" is in scope | FR-010: profiling deferred to follow-on specs | **High** – Direct contradiction |
| **Backend abstraction** | Section 1, 5: Backend abstraction (llama.cpp only) with vLLM future path is MVP | Not addressed in Spec-001 | **Medium** – Gap, not contradiction |
| **Hardware acknowledgment** | Section 7(FR-017), 10(AC-004): Hardware warning path required | Spec-001 mentions "unavailable slots" but not hardware topology warnings, allowlist, session snooze | **High** – Direct contradiction |

---

## Recommended Extensions (P0/P1/P2)

### P0 — Must Have for PRD Compliance

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| **P0-01** | Rename spec title from "PRD-Aligned MVP Control Plane" to "PRD M1: Slot-First Launch & Dry-Run" | Current title misrepresents scope. Spec-001 only covers ~20% of MVP. Honesty prevents stakeholder confusion. |
| **P0-02** | Add FR-020 (Backend abstraction): System MUST support backend selection (llama_cpp/vllm) via config; vllm MUST fail doctor unless experimental gate is set | PRD Section 1, 5 requires backend abstraction as MVP deliverable. Spec-001 omits entirely. |
| **P0-03** | Expand FR-003 (Dry-run): Add requirement to show: binary path, exact command line, model path, slot, merged env, OpenAI flag bundle, effective ports, hardware notes, vLLM matrix row | PRD Section 7(FR-003) is explicit; spec-001's "operator-readable form" too vague for testing. |
| **P0-04** | Add FR-021 (Hardware acknowledgment): System MUST warn on non-anchor hardware topology; require TUI Continue or CLI --ack-nonstandard-hardware; persist allowlist; implement session snooze | PRD Section 7(FR-017), 10(AC-004) requires full hardware warning path. |
| **P0-05** | Add FR-022 (GGUF parsing): System MUST parse GGUF metadata with 32 MiB prefix cap, 5-second parse timeout; support unsharded files; filename normalization; repo includes synthetic test fixtures | PRD Section 5, 7(FR-014), 10(AC-010, AC-018) requires GGUF handling as MVP. |
| **P0-06** | Clarify "slot/workstation config" in FR-001/FR-006: define schema with schema_version:1, model_id keys, slot keys, merge rules (user wins), inheritance model | Spec-001's "slot/workstation config" ambiguous. PRD Section 7(FR-013) is explicit. |

### P1 — Strongly Recommended

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| **P1-01** | Add FR-023 (Graceful shutdown): System MUST handle SIGTERM with wait-and-escalate; abnormal TUI exit MUST terminate child servers | PRD Section 7(FR-012), 10(AC-012) requires shutdown handling. |
| **P1-02** | Add FR-024 (Logging & reports): System MUST append mutating actions to rotating log with command/timestamp/exit code; redact secrets; drop report directory on failure | PRD Section 7(FR-018), 10(AC-007) requires detailed logging. |
| **P1-03** | Add FR-025 (Exit codes): doctor MUST return 0/1/2; smoke MUST return 1/2/3; document all exit codes in CLI help | PRD Section 7(FR-011), Appendix B require exit code conventions. |
| **P1-04** | Add FR-026 (VRAM heuristics): System MUST compare free VRAM against heuristic footprint; warn and require explicit confirm; not hard block except impossible headroom | PRD Section 7(FR-016), 10(AC-016) require VRAM warnings. |
| **P1-05** | Add FR-027 (Port conflict handling): System MUST refuse port-in-use unless explicit --force-bind (logged dangerous); define per-slot lockfiles; doctor --locks surfaces holders | PRD Section 7(FR-016) requires detailed port/lock handling. |
| **P1-06** | Add NFR-008 (Performance): Dry-run/config resolve MUST feel immediate (<100ms); long operations MUST show phase progress with ETA | PRD Section NFR-003 requires performance expectations. |
| **P1-07** | Expand Success Criteria: SC-004 is unverifiable; replace with "Dry-run output includes all fields specified in FR-003 for 100% of cases." | SC-004 vague; PRD Section 7(FR-003) defines exact dry-run fields. |

### P2 — Nice-to-Have

| # | Recommendation | Rationale |
| - | -------------- | --------- |
| **P2-01** | Add FR-028 (Documentation generation): System supports gendoc.py to extract <!-- readme:... --> marker sections from PRD into README | PRD Section 7(FR-019), 10(AC-015), Appendix C require gendoc.py. |
| **P2-02** | Add FR-029 (TUI monitoring): System MUST show per-slot status, logs, GPU telemetry, backend, build job state, profile warnings, UNSAVED badge | PRD Section 7(FR-010), 10(AC-011) require TUI monitoring. |
| **P2-03** | Add FR-030 (Build pipeline & provenance): System MUST allow TUI-driven serialized llama.cpp builds, record remote URL + master tip SHA + timestamps, doctor --repair clears failed staging | PRD Section 5, 7(FR-004, FR-006), 10(AC-006) require build pipeline. |

---

## Traceability Quick Map

| Type | ID | Status | Evidence |
| ---- | -- | ------ | -------- |
| **FR** | FR-001 Slot-first launch | Covered | Spec FR-001, User Story 1 |
| **FR** | FR-002 Backend selection | Missing | Spec-001 omits entirely |
| **FR** | FR-003 Dry-run | Partial | Spec FR-003 (core only) |
| **FR** | FR-004 TUI build pipeline | Missing | Deferred to M2 |
| **FR** | FR-005 Toolchain/diagnostics | Missing | Deferred to M2 |
| **FR** | FR-006 Artifacts/provenance | Missing | Deferred to M2 |
| **FR** | FR-007 Manual profiling | Missing | Deferred to M3 |
| **FR** | FR-008 Profile persistence | Missing | Deferred to M3 |
| **FR** | FR-009 Overrides | Partial | Spec FR-006 (precedence) |
| **FR** | FR-010 TUI monitoring | Missing | Deferred to M4 |
| **FR** | FR-011 CLI scripting | Missing | Deferred to M4 |
| **FR** | FR-012 Shutdown | Missing | Deferred to M4 |
| **FR** | FR-013 Config merge | Partial | "slot/workstation config" mentioned |
| **FR** | FR-014 GGUF metadata | Missing | Deferred to M4 |
| **FR** | FR-015 Smoke | Missing | Deferred to M4 |
| **FR** | FR-016 Local safety/locks | Partial | Spec FR-009 (locks), FR-002 (port) |
| **FR** | FR-017 Hardware acknowledgment | Missing | Deferred to M4 |
| **FR** | FR-018 Logging/reports | Partial | Spec FR-007 (redaction) |
| **FR** | FR-019 Docs generation | Missing | Deferred to M0 |
| **NFR** | NFR-001 Validation | Partial | Spec FR-002 (duplicate slot, port) |
| **NFR** | NFR-002 Failure attribution | Partial | Spec CA-004 (actionable diagnostics) |
| **NFR** | NFR-003 Performance | Missing | Not addressed |
| **NFR** | NFR-004 Determinism | Partial | Spec FR-006 (precedence) |
| **NFR** | NFR-005 Runtime safety | Missing | Not addressed |
| **NFR** | NFR-006 TUI responsiveness | Missing | Not addressed |
| **NFR** | NFR-007 Diagnostics | Partial | Spec FR-007 (redaction) |
| **AC** | AC-001 Distinct slots | Covered | User Story 1 Scenario 2 |
| **AC** | AC-002 Dry-run details | Missing | Spec FR-003 too vague |
| **AC** | AC-003 vllm guard | Missing | Not addressed |
| **AC** | AC-004 Hardware warning | Missing | Not addressed |
| **AC** | AC-005 Port conflict | Partial | Spec FR-002 (port conflicts) |
| **AC** | AC-006 Build artifacts | Missing | Deferred to M2 |
| **AC** | AC-007 Toolchain errors | Missing | Deferred to M2 |
| **AC** | AC-008 Profile cache | Missing | Deferred to M3 |
| **AC** | AC-009 Override precedence | Partial | Spec FR-006 (precedence) |
| **AC** | AC-010 GGUF metadata | Missing | Deferred to M4 |
| **AC** | AC-011 TUI monitoring | Missing | Deferred to M4 |
| **AC** | AC-012 Graceful shutdown | Missing | Deferred to M4 |
| **AC** | AC-013 Isolated crash | Missing | Deferred to M4 |
| **AC** | AC-014 Smoke MVP gate | Missing | Deferred to M4 |
| **AC** | AC-015 gendoc.py | Missing | Deferred to M0 |
| **AC** | AC-016 VRAM heuristic | Missing | Deferred to M4 |
| **AC** | AC-017 Smoke behavior | Missing | Deferred to M4 |
| **AC** | AC-018 GGUF fixtures | Missing | Deferred to M4 |
| **AC** | AC-019 Setup venv | Missing | Deferred to M2 |
| **AC** | AC-020 OpenAI flag bundle | Missing | Deferred to M4 |

---

## Summary

Spec-001 successfully covers the **M1 milestone** (slot-first orchestration + validation + dry-run core) but **does not** represent the full PRD MVP. Implementation teams should:

1. **Treat Spec-001 as M1-only** – Do not assume smoke, profiling, or TUI monitoring are in scope.
2. **Address P0 recommendations** – Rename title, add backend abstraction, expand dry-run, add hardware acknowledgment, add GGUF parsing.
3. **Plan M2-M4 follow-ons** – Build pipeline (M2), profiling (M3), smoke/monitoring/shutdown (M4).
4. **Update milestones** – Ensure implementation planning aligns with the PRD's M0-M4 roadmap, not Spec-001's narrow scope.

**Output Path:** `docs/spec-001-prd-coverage.md`
