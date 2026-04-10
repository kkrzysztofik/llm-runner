# Implementation Readiness Checklist: PRD M1 — Slot-First Launch & Dry-Run

**Purpose**: Validate that implementation tasks (`tasks.md`) are complete, clear, consistent, and measurable against the M1 spec/plan/contracts before execution.
**Created**: 2026-04-10
**Feature**: `/home/kmk/llm-runner/specs/001-prd-mvp-spec/spec.md`

**Note**: This is a requirements-quality checklist ("unit tests for English"), not an implementation behavior test suite.
**Consulted Subagents**: Architect, Python QA, Security Reviewer, Documentation

## Validation Summary

- **Total items**: 26
- **Completed**: 26
- **Incomplete**: 0
- **Status**: PASS

## Requirement Completeness

- [x] CHK001 Are all FR-001..FR-011 represented by explicit implementation tasks without unowned requirement areas? [Completeness, Spec §FR-001..FR-011, Tasks §Phases 1-6]
- [x] CHK002 Are FR-005 required error object fields (`error_code`, `failed_check`, `why_blocked`, `how_to_fix`, optional `docs_ref`) explicitly required in task definitions and test tasks? [Completeness, Spec §FR-005, Contract §Actionable Error, Tasks T008/T023/T031]
- [x] CHK003 Are FR-003 canonical dry-run required per-slot fields fully enumerated in task language, including `warnings` and `validation_results`? [Completeness, Spec §FR-003, Contract §Dry-Run, Tasks T024/T028]

## Requirement Clarity

- [x] CHK004 Is deterministic ordering phrased unambiguously as slot configuration sequence followed by `failed_check` ascending within each slot? [Clarity, Contract §Dry-Run Determinism, Tasks T003/T008/T023/T029]
- [x] CHK005 Is redaction scope explicitly documented as key-name based while preserving filesystem paths in output surfaces? [Clarity, Spec §FR-007, Clarification Q22, Tasks T010/T025]
- [x] CHK006 Is runtime directory fallback wording unambiguous about order (`LLM_RUNNER_RUNTIME_DIR` then `$XDG_RUNTIME_DIR/llm-runner`) and failure semantics when neither is usable? [Clarity, Spec §FR-007/§FR-009, Tasks T009/T013]

## Requirement Consistency

- [x] CHK007 Do lock integrity tasks use canonical `LOCKFILE_INTEGRITY_FAILURE` and `failed_check=lockfile_integrity` consistently across implementation and tests? [Consistency, Contract §Lock Integrity Failure Modes, Tasks T011/T015/T018]
- [x] CHK008 Do vLLM non-eligibility tasks consistently use the canonical FR-011 mapping (`BACKEND_NOT_ELIGIBLE`, `vllm_launch_eligibility`, canonical remediation)? [Consistency, Spec §FR-011, Tasks T027]
- [x] CHK009 Are CLI/TUI parity requirements expressed as semantic parity (fields/meanings) rather than presentation-level coupling? [Consistency, Spec §CA-003, Tasks T031/T021/T040]

## Acceptance Criteria Quality

- [x] CHK010 Is SC-002 measurability explicit that denominator counts all FR-005 `errors[n]` entries (including multi-error responses), not just response objects? [Acceptance Criteria, Spec §SC-002, Addendum §SC-002, Tasks T023]
- [x] CHK011 Is SC-003 evidence quality explicit for deterministic comparison surfaces across dry-run and artifact fields (`resolved_command`, `validation_results`, `warnings`)? [Acceptance Criteria, Spec §SC-003, Addendum §Definitions & Measurement Notes, Tasks T034/T042]
- [x] CHK012 Are SC-006 performance budgets documented with enough context to enable consistent, operation-specific measurement (single-slot dry-run, two-slot dry-run, per-slot lock/port validation)? [Acceptance Criteria, Spec §SC-006, Tasks T041]

## Scenario Coverage

- [x] CHK013 Are tasked scenarios complete for dual-slot success, degraded one-slot launch, and full-block launch outcomes? [Coverage, Spec §User Story 1 Scenarios, Tasks T014/T016/T019]
- [x] CHK014 Are exception scenarios covered for runtime-dir unusable, artifact persistence failure, and indeterminate lock ownership? [Coverage, Spec §FR-007/§FR-009, Tasks T009/T012/T030/T015]
- [x] CHK015 Are risky-operation acknowledgement scenarios complete for privileged port, non-loopback bind, and manual warning-bypass override? [Coverage, Spec §FR-008, Tasks T033/T037/T039/T040]

## Edge Case Coverage

- [x] CHK016 Are slot-id boundary rules explicit for normalization plus strict allowed-character rejection? [Edge Case, Spec Clarification Q5, Tasks T006]
- [x] CHK017 Are lockfile edge states addressed distinctly (stale, live, indeterminate) with deterministic failure mapping? [Edge Case, Spec §FR-009, Contract §Lock Integrity Failure Modes, Tasks T011/T015/T018]
- [x] CHK018 Are dry-run edge semantics explicit for raw argv `command_args` representation and stable ordering guarantees? [Edge Case, Spec §FR-003, Contract §Dry-Run, Tasks T024/T028/T029]

## Non-Functional Requirements

- [x] CHK019 Are permission requirements explicit for both files (`0600`) and directories (`0700`) in task text where artifacts/locks are persisted? [Non-Functional, Security, Spec §FR-007/§FR-009, Tasks T012/T030]
- [x] CHK020 Are deterministic guarantees documented in task wording without introducing undocumented tie-break rules that conflict with contracts? [Non-Functional, Determinism, Contract §Dry-Run Determinism, Tasks T003/T029/T042]
- [x] CHK021 Are performance tasks scoped to spec-defined budgets rather than implementation-local surrogate metrics? [Non-Functional, Performance, Spec §SC-006, Tasks T041]

## Dependencies & Assumptions

- [x] CHK022 Are phase dependencies explicit that user stories begin only after foundational prerequisites are complete? [Dependencies, Plan §Phase Dependencies, Tasks §Phase 2 Checkpoint]
- [x] CHK023 Are user stories independently testable after shared foundation, with no hidden cross-story blockers implied by task wording? [Dependencies, Spec §User Stories, Tasks §US1/US2/US3 Independent Test]

## Ambiguities & Conflicts

- [x] CHK024 Is terminology consistent and non-conflicting for “current launch attempt,” “warning,” “launch-blocking error,” and “degraded launch”? [Ambiguity, Spec Addendum §Canonical Terminology, Tasks T033/T038/T039]
- [x] CHK025 Are any tasks over-prescriptive about implementation internals in ways that could conflict with contract-level intent? [Ambiguity, Conflict, Plan §Constitution Check, Tasks §All Phases]
- [x] CHK026 Are there any task statements that contradict source contracts on field locations (e.g., canonical dry-run per-slot `validation_results.errors`)? [Conflict, Contract §Dry-Run, Tasks T024/T028/T042]
