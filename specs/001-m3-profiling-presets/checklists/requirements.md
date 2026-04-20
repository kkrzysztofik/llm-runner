# Specification Quality Checklist: M3 Profiling + Presets

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-04-20
**Updated**: 2026-04-20 (post-Architect review additions)
**Feature**: [specs/001-m3-profiling-presets/spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) — *Fixed: removed module paths, function names, Python-specific dataclass/enum references from FRs and Key Entities*
- [x] Focused on user value and business needs — *All user stories describe actor/value flows*
- [x] Written for non-technical stakeholders — *Success criteria and FRs now use system-level language*
- [x] All mandatory sections completed — *User Scenarios, Requirements, Key Entities, Success Criteria, Assumptions present*

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous — *Each FR has a clear verb and measurable outcome*
- [x] Success criteria are measurable — *All SCs have quantifiable targets*
- [x] Success criteria are technology-agnostic (no implementation details) — *Fixed: SC-M3-003 through SC-M3-006 renamed, TUI/server references removed*
- [x] All acceptance scenarios are defined — *Gherkin-style scenarios for each user story*
- [x] Edge cases are identified and resolved — *Four edge cases with explicit behavior in Error Recovery table*
- [x] Scope is clearly bounded — *MVP warns only (no strict-profiles), llama.cpp only, per-slot/per-backend*
- [x] Dependencies and assumptions identified — *Six assumptions: benchmark tool availability, GPU ID derivation, driver version sourcing, profile scope, vLLM deferral, cache locality*
- [x] Profile cache file format/schema defined — *Added ProfileRecord JSON schema with field semantics table*
- [x] Staleness detection rules defined — *Added explicit rules: driver hash mismatch, profile age threshold (30 days default), binary version change*
- [x] Profile override scope defined — *Added whitelist table: 5 overridable fields (threads, ctx_size, ubatch_size, cache_type_k/v), 5 excluded fields*
- [x] GPU identifier format defined — *CUDA: nvidia-{name}-{index}, SYCL: intel-{name}-{index}; driver hash: first 16 hex chars*
- [x] TUI profile trigger defined — *Keybinding P, confirmation prompt, flavor submenu, progress display, stale badge*
- [x] Profile loading integration point defined — *6-step flow from launch to merge function injection*
- [x] Error recovery rules defined — *9-condition table covering corrupt files, subprocess crashes, timeouts, locks*
- [x] Cache eviction strategy defined — *MVP: manual via doctor --repair, no auto-eviction; post-MVP: automatic*

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria — *Each FR maps to PRD FR-007/FR-008/FR-009*
- [x] User scenarios cover primary flows — *TUI profiling, persistence, staleness, overrides*
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification
- [x] FR-M3-005 acknowledges existing merge_config_overrides — *spec clarifies M3 adds cache loading, not merge logic*
- [x] CA-001 clarifies subprocess boundary — *profiler.py = pure library; subprocess execution in llama_cli/*

## Notes

- Architect review identified 10 recommendations (R1-R10); all P0-P1 items applied
- Critical additions: ProfileRecord JSON schema, staleness rules table, override scope whitelist, GPU identifier format, error recovery table, TUI trigger details, profile loading flow, cache eviction strategy
- Constitution Alignment section intentionally contains implementation guidance (module names, test strategies) — appropriate for developer-facing alignment
- CA-001 explicitly separates pure library (profiler.py) from subprocess execution (llama_cli/)
- All validation iterations complete — spec is ready for `/speckit.plan`
