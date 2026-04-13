# Implementation Requirements Quality Checklist: PRD M1 — Slot-First Launch & Dry-Run

**Purpose**: Validate that the M1 specification is complete, clear, consistent, and measurable before implementation planning/review.
**Created**: 2026-04-09
**Feature**: `specs/001-prd-mvp-spec/spec.md`

**Note**: This checklist is a requirements-quality gate ("unit tests for English"), not an implementation behavior test.
**Consulted Subagents**: Architect, Python QA, Security Reviewer

## Validation Summary

- **Total items**: 32
- **Completed**: 32
- **Incomplete**: 0
- **Status**: PASS

## Requirement Completeness

- [x] CHK001 Are fallback requirements defined for lock/artifact paths when `$XDG_RUNTIME_DIR` is unset, missing, or not writable? [Completeness, Gap, Spec §FR-007, §FR-009]
- [x] CHK002 Are all warning classes explicitly documented that can be bypassed by "manual override" in risky-operation acknowledgement? [Completeness, Gap, Spec §FR-008]
- [x] CHK003 Are requirements defined for representing multiple simultaneous launch-blocking conditions in one response (single error vs error list)? [Completeness, Gap, Spec §FR-005]
- [x] CHK004 Are artifact lifecycle requirements defined (retention window, cleanup responsibility, and overwrite behavior)? [Completeness, Gap, Spec §FR-007, §SC-005]

## Requirement Clarity

- [x] CHK005 Is "slot-first orchestration" defined with explicit, testable invariants (ownership, uniqueness, and conflict semantics)? [Clarity, Ambiguity, Spec §FR-001]
- [x] CHK006 Is "launch-eligible" precisely defined as a validation-state concept, runtime-state concept, or both? [Clarity, Ambiguity, Spec §FR-011]
- [x] CHK007 Is the required format for "hardware notes" in dry-run output explicitly specified (structured fields vs free text)? [Clarity, Ambiguity, Spec §FR-003]
- [x] CHK008 Is "resolved command" clearly scoped (binary + fully quoted args + effective env context boundaries)? [Clarity, Ambiguity, Spec §FR-003, §FR-007]

## Requirement Consistency

- [x] CHK009 Do requirements avoid contradiction between showing model paths in dry-run and redacting filesystem paths by default? [Consistency, Conflict, Spec §FR-003, §FR-007]
- [x] CHK010 Are "session-only" and "current launch attempt only" terms used with one consistent canonical scope definition? [Consistency, Ambiguity, Spec §Clarifications, §Edge Cases, §FR-008]
- [x] CHK011 Are "explicit override" and "user override" treated consistently as one precedence layer rather than two different layers? [Consistency, Ambiguity, Spec §FR-006]
- [x] CHK012 Does M1 scope exclusion language remain consistent with forward-compatibility language for non-eligible backends? [Consistency, Ambiguity, Spec §FR-010, §FR-011]

## Acceptance Criteria Quality

- [x] CHK013 Is SC-002 measurable with a defined denominator/population for "launch-blocking failures"? [Acceptance Criteria, Measurability, Spec §SC-002]
- [x] CHK014 Is SC-003 backed by explicit evidence sources (which output/artifact proves operator-verifiable determinism)? [Acceptance Criteria, Measurability, Spec §SC-003, §FR-006, §FR-007]
- [x] CHK015 Can SC-004 be objectively checked via an explicit dry-run output schema (field names and required presence rules)? [Acceptance Criteria, Measurability, Spec §SC-004, §FR-003]
- [x] CHK016 Does SC-005 define the minimum artifact content contract sufficiently to determine pass/fail without interpretation drift? [Acceptance Criteria, Measurability, Spec §SC-005, §FR-007]

## Scenario Coverage

- [x] CHK017 Are "slot unavailable" scenarios enumerated by cause (port conflict, missing model source, active lock owner, backend non-eligibility) rather than one generic condition? [Coverage, Gap, Spec §User Story 1, §FR-002, §FR-004, §FR-009, §FR-011]
- [x] CHK018 Are requirements explicit for how dry-run represents non-eligible `vllm` rows while preserving M1 block semantics? [Coverage, Clarity, Spec §FR-003, §FR-011]
- [x] CHK019 Are structured error requirements explicitly aligned for both CLI and TUI surfaces to satisfy cross-interface consistency? [Coverage, Gap, Spec §FR-005, §CA-003]
- [x] CHK020 Are recovery requirements defined for failures during stale-lock cleanup (e.g., permission denied, concurrent writer, malformed lock content)? [Coverage, Recovery, Gap, Spec §FR-009]

## Edge Case Coverage

- [x] CHK021 Is malformed/truncated lockfile behavior specified (block, self-heal, or fail-safe with guidance)? [Edge Case, Gap, Spec §FR-009]
- [x] CHK022 Are PID reuse and TOCTOU race scenarios addressed for live-owner detection semantics? [Edge Case, Gap, Spec §FR-009]
- [x] CHK023 Are quoting/escaping rules specified for dry-run "exact command arguments" when arguments contain whitespace or shell-sensitive characters? [Edge Case, Ambiguity, Spec §FR-003]
- [x] CHK024 Are redaction boundary rules defined to avoid false negatives/positives (substring policy, case policy, and path pattern policy)? [Edge Case, Clarity, Spec §FR-007]

## Non-Functional Requirements

- [x] CHK025 Are performance expectations specified for dry-run resolution and lock/port checks to avoid undefined latency budgets? [Non-Functional, Gap, Spec §FR-003, §FR-009, §SC-001]
- [x] CHK026 Are security/privacy requirements defined for file permissions on lockfiles and observability artifacts? [Non-Functional, Security, Gap, Spec §FR-007, §FR-009]
- [x] CHK027 Are reliability requirements specified for artifact persistence failures (disk full, permission errors) including expected operator-facing outcomes? [Non-Functional, Reliability, Gap, Spec §FR-007, §FR-005]

## Dependencies & Assumptions

- [x] CHK028 Is the dependency boundary for deferred M0 documentation work traceable and non-conflicting with M1 acceptance scope? [Dependency, Assumption, Spec §FR-010, §Assumptions]
- [x] CHK029 Are deferred M2-M4 areas linked to named follow-on specifications or backlog artifacts to prevent implicit scope leakage? [Dependency, Gap, Spec §FR-010, §Assumptions]

## Ambiguities & Conflicts

- [x] CHK030 Is the actionable remediation content for blocked `vllm` selections explicitly defined (required correction wording/components)? [Ambiguity, Gap, Spec §FR-011, §FR-005]
- [x] CHK031 Is acknowledgement scope for risky operations explicitly stated for both dry-run and launch execution paths? [Ambiguity, Coverage, Spec §FR-008, §User Story 3]
- [x] CHK032 Is a canonical terminology mini-glossary needed to avoid interpretation drift across "warning", "blocking error", "risky operation", and "degraded launch"? [Ambiguity, Consistency, Gap, Spec §FR-004, §FR-005, §FR-008]

## Notes

- Intended depth: **Standard PR reviewer gate**.
- Scope: **M1 requirements quality** plus **dependency readiness checks** for deferred M0/M2-M4.
- Items flagged `[Gap]`, `[Ambiguity]`, or `[Conflict]` are priority clarification candidates before `/speckit.plan`.
