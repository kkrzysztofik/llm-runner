# PRD Spec-001 Extension Patch Plan

**Purpose:** Provide implementation-ready patch plan to extend `specs/001-prd-mvp-spec/spec.md` from M1-only scope to full PRD MVP compliance or minimum viable compliance.

**Document Status:** Draft — Ready for review and implementation.

**Created:** 2026-04-09

---

## 1. Executive Summary

`specs/001-prd-mvp-spec/spec.md` (Spec-001) currently covers **only M1 milestone** (slot-first launch + dry-run core + validation) out of the full PRD MVP scope. This plan provides two tracks:

- **Track A: M1-Honest** — Keep narrow M1 scope, add clarifying documentation and P0/P1 gaps that don't require architectural changes.
- **Track B: Full PRD MVP** — Extend spec to include M2-M4 requirements (build pipeline, profiling, smoke verification, TUI monitoring, graceful shutdown, hardware acknowledgment).

**Recommendation:** Proceed with Track A immediately; plan Track B as phased follow-on work.

---

## 2. Two Tracks Clarification

### Track A: M1-Honest (Minimum Viable Compliance)

**Goal:** Make Spec-001 honest about its M1-only scope while addressing critical P0 gaps that prevent implementation ambiguity.

**Scope:**

- Rename title to "PRD M1: Slot-First Launch & Dry-Run"
- Add backend abstraction (vLLM guard)
- Expand dry-run to include OpenAI flag bundle for anchored models
- Add hardware acknowledgment (warn + allowlist + session snooze)
- Add GGUF metadata parsing (32 MiB cap, 5s timeout)
- Clarify config merge rules (schema_version:1, merge order)

**Out of Scope:** Build pipeline, profiling, smoke verification, TUI monitoring, graceful shutdown implementation details.

### Track B: Full PRD MVP Compliance

**Goal:** Extend Spec-001 to cover all PRD MVP requirements through M0-M4 milestones.

**Scope:** All Track A requirements, plus:

- **M0:** Documentation generation (gendoc.py, marker extraction)
- **M2:** Build wizard (serialized SYCL/CUDA builds, provenance, doctor --repair)
- **M3:** Manual profiling (bench integration, profile persistence, staleness warnings)
- **M4:** Operational hardening (smoke both/slot, TUI monitoring, graceful shutdown, exit codes, logging/reports)

**Trade-offs:** Significantly larger spec document, requires coordination across multiple implementation phases, higher review burden.

---

## 3. Section-by-Section Patch Plan

### Section 1: Title & Metadata

**Insertion Point:** Lines 1-6 (top of spec.md)

**Rationale:** Title misrepresents narrow M1 scope as "PRD-Aligned MVP Control Plane."

**Draft Changes:**

```markdown
- **Current Title:** Feature Specification: PRD-Aligned MVP Control Plane
- **New Title (Track A):** Feature Specification: PRD M1 — Slot-First Launch & Dry-Run
- **New Title (Track B):** Feature Specification: PRD MVP — Slot-First Launch, Dry-Run, Build, Smoke

- **Status:** Draft → Updated per PRD Spec-001 Extension Patch
- **Input:** PRD.md v0.3 (M0-M4 milestones)
- **Track:** [A: M1-Honest | B: Full PRD MVP]
```

### Section 2: Clarifications

**Insertion Point:** Lines 8-17 (after "## Clarifications")

**Rationale:** PRD requires backend selection semantics and hardware acknowledgment that are not addressed in original spec.

**Draft Changes:**

```markdown
### Session 2026-04-09 (Extension Patch)

- Q: Should the spec title reflect M1-only scope or full MVP? → A: Track A uses "M1" in title; Track B uses "MVP".
- Q: What are backend selection semantics? → A: llama_cpp is MVP runtime only; vllm MUST fail `doctor` unless experimental gate is set (Track A).
- Q: How should hardware acknowledgment work? → A: Non-anchor hardware → warn; TUI Continue or CLI `--ack-nonstandard-hardware`; persistent allowlist under `~/.config/llm-runner/`; session snooze under `$XDG_RUNTIME_DIR/llm-runner/` (Track A).
- Q: What are GGUF parsing constraints? → A: 32 MiB prefix cap, 5-second parse timeout, filename normalization, synthetic test fixtures required (Track A).
- Q: How should config merge work? → A: schema_version:1; defaults + user override file merge by model_id/slot keys; user wins on conflicts; `doctor migrate-config` creates *.bak backup (Track A).

### Session TBD (Track B Only)

- Q: What are build provenance requirements? → A: Record remote URL, master tip SHA, timestamps; doctor --repair clears failed staging (M2).
- Q: How should profiling results persist? → A: ~/.cache/llm-runner/profiles/ keyed by GPU IDs + backend + flavor; staleness warnings (M3).
- Q: What are smoke verification phases? → A: listen → /v1/models → minimal chat completion; sequential slot A then slot B; inter-slot delay 2s; distinct exit codes from doctor (M4).
- Q: How should graceful shutdown work? → A: SIGTERM, wait, escalate; abnormal TUI exit terminates child servers (M4).
```

### Section 3: User Scenarios

**Insertion Point:** Line 18 (before "### User Story 1")

**Rationale:** Original spec has three user stories for M1 only. Track B needs additional stories for build, profiling, and smoke.

**Draft Changes:**

#### Track A: No new user stories (M1 scope unchanged)

No changes to User Stories 1-3. Add **Track B Note** below:

```markdown
> **Track B Note:** User Stories 4-8 (Build Wizard, Manual Profiling, Smoke Verification, TUI Monitoring, Graceful Shutdown) defined in Track B extension.
```

#### Track B: Add new user stories (summaries for brevity)

- **User Story 4 (P1):** Build llamma.cpp artifacts — serialized SYCL/CUDA builds, preflight, SHA recording, doctor --repair
- **User Story 5 (P2):** Manual profiling — bench subprocess, profile persistence, staleness warnings
- **User Story 6 (P1):** Smoke verification — `smoke both`/`smoke slot`, listen→/v1/models→chat phases, exit codes
- **User Story 7 (P3):** TUI monitoring — per-slot status, logs, GPU telemetry, build state, UNSAVED badge
- **User Story 8 (P2):** Graceful shutdown — SIGTERM handling, orphan prevention, abnormal exit cleanup

### Section 4: Requirements

**Insertion Point:** Line 88 (before "### Functional Requirements")

**Rationale:** Original spec has FR-001 to FR-010 (M1 only). Track A needs backend abstraction, hardware acknowledgment, GGUF parsing. Track B needs M2-M4 requirements.

**Draft Changes:**

#### Track A: Add FR-011 to FR-014

```markdown
### Functional Requirements (Track A Extension)

- **FR-011**: System MUST support backend selection via config (llama_cpp or vllm); vllm MUST fail `doctor` unless experimental gate is explicitly set.
- **FR-012**: System MUST warn on non-anchor hardware topology; TUI requires explicit Continue; CLI requires `--ack-nonstandard-hardware`; persist allowlist under `~/.config/llm-runner/`; implement session snooze under `$XDG_RUNTIME_DIR/llm-runner/`.
- **FR-013**: System MUST parse GGUF metadata with 32 MiB prefix cap and 5-second parse timeout; support unsharded files; implement filename normalization; repository includes synthetic test fixtures.
- **FR-014**: System MUST merge config by schema_version:1 with defaults + user override file; merge by model_id/slot keys; user wins on conflicts; `doctor migrate-config` creates *.bak backup; fail validation on bad migrate.
```

#### Track B: Add M2-M4 requirements

- **M2 (FR-015 to FR-017):** Build pipeline with provenance, setup venv
- **M3 (FR-018 to FR-019):** Manual profiling with persistence and staleness
- **M4 (FR-020 to FR-027):** TUI monitoring, smoke, graceful shutdown, logging, VRAM heuristics
- **M0 (FR-028):** gendoc.py for README extraction

### Section 5: Success Criteria

**Insertion Point:** Line 136 (before "### Measurable Outcomes")

**Rationale:** Original criteria SC-001 to SC-004 are M1-focused. Track B needs additional criteria for build, profiling, smoke, and monitoring.

**Draft Changes:**

#### Track A: Update existing criteria

- **SC-001:** 100% of valid launch attempts complete without manual command editing.
- **SC-002:** At least 95% of launch-blocking failures return actionable correction message (vllm guard, hardware warnings, GGUF errors).
- **SC-003:** 100% of override resolution cases produce deterministic, operator-verifiable results.
- **SC-004:** Dry-run output includes all fields (binary, args, model path, slot, merged env, OpenAI flag bundle, ports, hardware notes, vLLM matrix) for 100% of cases.
- **SC-005:** GGUF metadata extraction succeeds within 32 MiB cap and 5s timeout for 100% of supported files.
- **SC-006:** Hardware acknowledgment flow works for non-anchor hardware: warn, --ack-nonstandard-hardware, allowlist, session snooze.

#### Track B: Add Track B criteria (SC-007 to SC-017)

See full plan document for M2-M4 specific acceptance criteria.

### Section 6: Assumptions

**Insertion Point:** Line 147 (before "### Assumptions")

**Rationale:** Original assumptions focus on M1 scope. Add Track B assumptions for build, profiling, smoke, monitoring.

**Draft Changes:**

#### Track A Updates

Add assumptions for backend selection, GGUF parsing, config merge.

#### Track B Only

Add assumptions for serialized builds, profiling persistence, smoke timeouts, TUI responsiveness, graceful shutdown, gendoc.py.

### Section 7: Traceability Matrix (Track B Only)

**Insertion Point:** New section after "Assumptions"

**Rationale:** Track B needs explicit traceability from spec requirements to PRD FR/NFR/AC.

**Draft Changes:** Add traceability table mapping Spec-001 requirements to PRD FR-001 to FR-019, NFR-001 to NFR-007, and AC-001 to AC-020 with status columns (Covered, Track A, Track B M0/M2/M3/M4, Partial).

---

## 4. Prioritized Changes (P0/P1/P2)

### P0 — Critical for Track A (M1-Honest Compliance)

| # | Change | Rationale |
| - | ------ | --------- |
| **P0-01** | Rename title to "PRD M1: Slot-First Launch & Dry-Run" | Current title misrepresents M1-only scope as full MVP |
| **P0-02** | Add FR-011 (Backend abstraction): llama_cpp only; vllm fails doctor | PRD Section 1, 5 requires backend abstraction; spec-001 omits |
| **P0-03** | Expand FR-003 (Dry-run): Add exact fields spec | PRD Section 7(FR-003) explicit; spec-001 too vague |
| **P0-04** | Add FR-012 (Hardware acknowledgment) | PRD Section 7(FR-017), 10(AC-004) require |
| **P0-05** | Add FR-013 (GGUF parsing): 32 MiB cap, 5s timeout | PRD Section 5, 7(FR-014), 10(AC-010, AC-018) |
| **P0-06** | Clarify "slot/workstation config" (schema_version:1) | Spec-001 ambiguous; PRD Section 7(FR-013) explicit |

### P1 — Strongly Recommended for Track A

| # | Change | Rationale |
| - | ------ | --------- |
| **P1-01** | Add FR-014 (Config merge): user override, migrate backup | PRD Section 7(FR-013) required but missing |
| **P1-02** | Replace unverifiable SC-004 with dry-run field coverage | SC-004 vague; FR-003 defines exact fields |
| **P1-03** | Add NFR-008 (Performance): dry-run <100ms | PRD Section NFR-003 requires |

### P2 — Nice-to-Have for Track A

| # | Change | Rationale |
| - | ------ | --------- |
| **P2-01** | Add FR-015 (Logging & reports) | PRD Section 7(FR-018) requires |
| **P2-02** | Add FR-016 (Exit codes) | PRD Section 7(FR-011), Appendix B require |

### P0 — Track B Only (Full MVP)

| # | Change | Milestone |
| - | ------ | --------- |
| **PB-01** | M0: gendoc.py, README markers | M0 |
| **PB-02** | M2: build pipeline, provenance | M2 |
| **PB-03** | M3: profiling, persistence | M3 |
| **PB-04** | M4: TUI monitoring, smoke, shutdown | M4 |

---

## 5. Staged Rollout Plan

### Phase 1: Immediate (Track A P0) — 2-3 days

**Goal:** Make Spec-001 honest about M1 scope and address critical PRD gaps.

**Deliverables:**

- [ ] Rename spec title to "PRD M1: Slot-First Launch & Dry-Run"
- [ ] Add P0-02 (backend abstraction FR-011)
- [ ] Add P0-04 (hardware acknowledgment FR-012)
- [ ] Add P0-05 (GGUF parsing FR-013)
- [ ] Update FR-003 (dry-run fields) to P0-03 spec
- [ ] Clarify "slot/workstation config" in FR-001/FR-006 (P0-06)
- [ ] Update Clarifications section with Track A Q&A
- [ ] Update Success Criteria SC-001 to SC-006 (Track A)
- [ ] Add Assumptions (Track A updates)

**Verification:**

- Spec-001 title reflects M1-only scope
- All P0 requirements present and testable
- PRD traceability clear (FR-001 to FR-013, AC-001 to AC-005, AC-009 to AC-010, AC-014 to AC-018)

### Phase 2: Short-Term (Track A P1/P2) — 1 week

**Goal:** Complete Track A with strongly recommended and nice-to-have changes.

**Deliverables:**

- [ ] Add P1-01 (FR-014 config merge)
- [ ] Update SC-004 to P1-02 (unverifiable → dry-run field coverage)
- [ ] Add P1-03 (NFR-008 performance)
- [ ] Add P2-01 (FR-015 logging & reports)
- [ ] Add P2-02 (FR-016 exit codes)
- [ ] Add Traceability Quick Map

**Verification:**

- All P0-P2 changes integrated
- Spec-001 self-consistent for M1 implementation
- Documentation complete for M1 handoff

### Phase 3: Medium-Term (Track B M0-M2) — 2-3 weeks

**Goal:** Add Track B M0 (docs generation) and M2 (build pipeline) to spec.

**Deliverables:**

- [ ] Add PB-01 (M0: gendoc.py, README markers)
- [ ] Add PB-02 (M2: build pipeline FR-015 to FR-017)
- [ ] Add M2 user story (User Story 4)
- [ ] Add M2 acceptance criteria (SC-007 to SC-009)
- [ ] Update Traceability Matrix (Track B)
- [ ] Add M2 assumptions

**Verification:**

- Spec-001 covers M0 and M2 requirements
- Build pipeline workflow clear
- Provenance requirements defined

### Phase 4: Medium-Term (Track B M3) — 1-2 weeks

**Goal:** Add Track B M3 (profiling & presets) to spec.

**Deliverables:**

- [ ] Add PB-03 (M3: profiling FR-018 to FR-019)
- [ ] Add M3 user story (User Story 5)
- [ ] Add M3 acceptance criteria (SC-010 to SC-011)
- [ ] Add M3 assumptions
- [ ] Update Traceability Matrix (M3 rows)

**Verification:**

- Spec-001 covers M3 requirements
- Profiling workflow clear
- Staleness mechanism defined

### Phase 5: Long-Term (Track B M4) — 3-4 weeks

**Goal:** Add Track B M4 (operational hardening + smoke) to spec.

**Deliverables:**

- [ ] Add PB-04 (M4: FR-020 to FR-027)
- [ ] Add User Stories 6-8 (smoke, TUI monitoring, graceful shutdown)
- [ ] Add M4 acceptance criteria (SC-012 to SC-017)
- [ ] Add M4 assumptions
- [ ] Complete Traceability Matrix (all rows)
- [ ] Final review of full PRD MVP coverage

**Verification:**

- Spec-001 covers M0-M4 requirements
- Full PRD MVP traceability complete
- Implementation-ready for MVP delivery

---

## 6. Quality Gates / Verification Checklist

### Pre-Implementation Gates

- [ ] PRD.md and Spec-001 both present and readable
- [ ] Extension patch plan reviewed by product owner
- [ ] Track selection documented (A or B)
- [ ] All P0 requirements identified and prioritized
- [ ] Implementation timeline approved

### During Implementation Gates

- [ ] Each requirement has corresponding test case
- [ ] Traceability matrix kept up-to-date
- [ ] User stories include independent test scenarios
- [ ] Success Criteria are measurable and verifiable
- [ ] Acceptance Criteria match PRD originals

### Post-Implementation Gates

- [ ] Spec-001 passes self-consistency review
- [ ] All PRD FR/NFR/AC either covered or explicitly deferred
- [ ] Implementation team can read spec and start coding without clarification
- [ ] Documentation includes all necessary examples and edge cases
- [ ] CI/CD gates (lint, typecheck, test) defined and passing

### Quality Metrics

- **Traceability Coverage:** 100% of PRD FR/NFR/AC mapped
- **Requirement Clarity:** All MUST statements testable
- **User Story Completeness:** Each has independent test + acceptance scenarios
- **Success Criteria:** All measurable with target percentages
- **Gap Transparency:** All missing items explicitly documented

---

## 7. Risks and Open Questions

### Risks

| Risk | Probability | Impact | Mitigation |
| ---- | ----------- | ------ | ---------- |
| Spec-001 title confusion | High | High | Rename immediately (P0-01) |
| Stakeholder expectations (MVP vs M1) | High | High | Track A/B clarification; executive briefing |
| Implementation ambiguity (backend, hardware) | Medium | High | Add P0-02, P0-04 to spec |
| GGUF parsing edge cases | Medium | Medium | P0-05 with 32 MiB cap, 5s timeout, fixtures |
| Config merge complexity | Medium | Medium | P0-06 with explicit schema_version:1 rules |
| M4 requirements scope creep | Medium | High | Phase rollout; defer non-critical M4 items |
| Spec document bloat (Track B) | Low | Medium | Consider splitting into M0, M2, M3, M4 separate specs |

### Open Questions

1. **Track B scope boundary:** Should M4 (smoke, monitoring, shutdown) be combined with M0-M3 or kept separate?
   - *Recommendation:* Keep as Track B full MVP, but implement in phases.

2. **vLLM experimental gate:** What is the env var name for vLLM experimental enablement?
   - *Recommendation:* `LLM_RUNNER_EXPERIMENTAL_VLLM=1` (document in spec).

3. **Hardware fingerprint format:** How should machine fingerprint be computed (PCI IDs, model names, etc.)?
   - *Recommendation:* Use `lspci` + `sycl-ls` output hash; document in spec.

4. **GGUF fixture generator:** Who maintains synthetic GGUF fixtures?
   - *Recommendation:* Document maintainer script; commit to tests/fixtures/; regenerate only on spec changes.

5. **Smoke model_id precedence:** Which takes precedence for smoke model_id resolution?
   - *Recommendation:* Per PRD: `general.name` (GGUF) → normalized filename → catalog `smoke.model_id` → override.

6. **Staleness threshold:** What is the profile staleness threshold (30 days recommended)?
   - *Recommendation:* 30 days default; configurable via `Config()`.

7. **Read-only vs mutating UX:** When does setup require explicit confirmation vs auto-apply?
   - *Recommendation:* All mutating steps require confirmation; read-only diagnostics auto-run.

---

## 8. Recommended Next Action

**Immediate Action:** Proceed with **Track A (M1-Honest)** following Phase 1 deliverables.

**Steps:**

1. **Day 1-2:** Complete P0 requirements (title rename, backend abstraction, hardware acknowledgment, GGUF parsing, dry-run fields, config merge clarification)
2. **Day 3:** Review spec with product owner; confirm Track A scope
3. **Day 4-5:** Complete P1/P2 requirements if time permits
4. **Day 7:** Final spec review and approval; begin implementation planning

**Out of Scope for Now:**

- Track B full MVP extension (plan for M2-M4 as separate phases)
- Implementation of actual features (build pipeline, smoke, TUI monitoring)
- Code changes (this is documentation only)

**Deliverable:** Updated `specs/001-prd-mvp-spec/spec.md` ready for M1 implementation handoff.

---

## End of Extension Patch Plan
