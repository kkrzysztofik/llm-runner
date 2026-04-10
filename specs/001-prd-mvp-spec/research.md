# Phase 0 Research — PRD M1 Slot-First Launch & Dry-Run

## 1) Runtime directory resolution behavior

- **Decision**: Resolve runtime directory in this order: `LLM_RUNNER_RUNTIME_DIR` (if set and usable), then `$XDG_RUNTIME_DIR/llm-runner`; if neither is usable, return FR-005 launch-blocking error.
- **Rationale**: Matches clarified requirement and preserves deterministic filesystem behavior for lockfiles/artifacts.
- **Alternatives considered**:
  - Fail immediately when `LLM_RUNNER_RUNTIME_DIR` is set but unusable (rejected: conflicts with clarified fallback order).
  - Silent fallback to temp dir (rejected: non-deterministic and weak observability).

## 2) Profile guidance precedence layer (FR-006)

- **Decision**: Represent profile guidance as a structured preset layer that is merged after slot/workstation config and before explicit user overrides.
- **Rationale**: Preserves the required precedence chain: defaults < slot/workstation < profile guidance < explicit override.
- **Alternatives considered**:
  - Treat profile guidance as part of slot config only (rejected: removes explicit precedence layer).
  - Treat profile guidance as top-level override (rejected: would conflict with explicit user override priority).

## 3) Canonical dry-run `openai_flag_bundle` shape

- **Decision**: Use an object/map of OpenAI-compatibility flags and effective values derived from resolved launch intent.
- **Rationale**: Machine-parseable contract supports deterministic outputs and parity across CLI/TUI presentations.
- **Alternatives considered**:
  - Flat joined string (rejected: not deterministic enough for contract tests).
  - Unstructured free-text notes (rejected: cannot reliably validate).

## 4) Canonical dry-run `vllm_eligibility` row

- **Decision**: Use structured fields: backend, eligible (bool), failed_check/error_code (when blocked), why_blocked, how_to_fix.
- **Rationale**: Aligns FR-003 matrix-row requirement with FR-011 canonical remediation fields.
- **Alternatives considered**:
  - Boolean-only eligibility (rejected: lacks actionable context).
  - Human-only text row (rejected: weak machine validation).

## 5) Slot ID normalization and validation

- **Decision**: Enforce normalized, filesystem-safe `slot_id` values (lowercase canonical form, strict allowed characters) with duplicate detection at validation time.
- **Rationale**: Lockfile and artifact naming depend on predictable slot keys; strict normalization prevents collisions and ambiguity.
- **Alternatives considered**:
  - Accept arbitrary strings (rejected: unsafe/ambiguous filenames).
  - Numeric-only slots (rejected: poor operator ergonomics and future extensibility).

## 6) FR-007 redaction boundaries

- **Decision**: Redact values for any environment key containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `AUTH` (case-insensitive); keep filesystem paths visible.
- **Rationale**: Matches clarified redaction policy and balances security with operator troubleshooting needs.
- **Alternatives considered**:
  - Redact all environment values (rejected: too opaque for debugging).
  - Redact only exact key names (rejected: misses many sensitive variants).

## 7) Deterministic dry-run verification strategy (SC-003/SC-004)

- **Decision**: Validate canonical schema completeness and deterministic equality via normalized machine-parseable output comparisons in pytest.
- **Rationale**: Contract-first comparisons reduce presentation noise and directly enforce determinism for identical inputs.
- **Alternatives considered**:
  - Raw text snapshot-only testing (rejected: brittle to formatting differences).
  - Manual-only verification (rejected: non-repeatable in CI).

## 8) SC-002 denominator for actionable-error threshold

- **Decision**: Use all FR-005 launch-blocking validation outcomes produced in launch and dry-run acceptance tests as the denominator.
- **Rationale**: Directly matches clarified success-criteria scope and avoids inflated/ambiguous measurement populations.
- **Alternatives considered**:
  - Use number of failing test cases only (rejected: undercounts per-run multi-error outcomes).
  - Use all validation checks including non-blocking warnings (rejected: violates SC-002 intent).

## 9) Performance budget verification approach

- **Decision**: Use deterministic benchmark-style test harnesses (mocked hardware/process dependencies) and reproducible percentile calculation for dry-run/lock validation.
- **Rationale**: Keeps CI reliable while still measuring against FR-003/FR-009 p95 targets.
- **Alternatives considered**:
  - Real hardware timing in CI (rejected: non-deterministic and unavailable in standard runners).
  - No automated timing checks (rejected: cannot demonstrate SC-006 evidence).
