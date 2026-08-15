# Design: Remove Snyk from llm-runner

**Date:** 2026-08-15
**Status:** Approved
**Approach:** Delete the Snyk CI job outright. Keep `pip-audit`, SonarCloud, Dependabot,
and CodeQL Default Setup (already active). Add anyka-shaped `.github/codeql/` path-ignore
config and document the security stack. No new SCA/SAST vendor.

## Problem

Snyk runs as a dedicated job in `.github/workflows/ci.yml` after lint/typecheck/test:

- **Snyk Open Source** — `uv export` → requirements.txt → `snyk test` (pip, severity ≥ high)
- **Snyk Code** — `snyk code test` (severity ≥ high)
- Soft-skips when `SNYK_TOKEN` is empty (forks / missing secret)

Drivers for removal (approved):

1. **Cost / vendor noise** — paid token, account, duplicate alerts
2. **CI friction** — export/install/npm CLI steps; fork skip path
3. **Consistency with anyka-dev** — that repo deleted Snyk and kept CodeQL + Sonar (+ cargo audit for Rust)

### Coverage already exists

| Concern | Existing coverage |
| -------- | ----------------- |
| SCA (Python deps) | `pip-audit` in CI `audit` job + pre-commit; Dependabot `pip` |
| SAST | CodeQL Default Setup (`dynamic/github-code-scanning/codeql`); SonarCloud job |
| Light security lint | ruff rule set `S` (bandit) |
| Actions deps | Dependabot `github-actions` |

Unlike anyka-dev, there is **no RustSec gap**. Removing Snyk does not require a
replacement scanner for a missing ecosystem.

## Alternatives considered

**OSV-Scanner** — overlaps `pip-audit` + Dependabot; rejected for this change.

**Semgrep OSS** — third SAST on top of CodeQL + Sonar; rejected (same as anyka).

**GitHub Dependency review action** — useful later for Dependabot PR diffs; **out of
scope** for this change (optional follow-up).

**CodeQL Advanced Setup workflow** — rejected for now; Default Setup already active.
Commit config for Default Setup; only add `codeql.yml` later if GitHub ignores the
config file.

**Tighten `pip-audit` severity / drop ignore-vulns** — out of scope unless a real CVE
forces it.

## Design

### 1. Removal

| Location | Action |
| -------- | ------ |
| `.github/workflows/ci.yml` `snyk` job | Delete entirely |
| Repo secret `SNYK_TOKEN` | Manual delete after merge (PR checklist) |
| `.snyk` policy files | None present |

Grep after change: zero `snyk` / `SNYK` hits under `.github/`, `AGENTS.md`, `docs/`,
`.opencode/`.

### 2. CodeQL config (anyka-shaped)

Create:

- `.github/codeql/codeql-config.yml` — short `paths-ignore` list for noise paths
  (`**/tests/**`, `**/.venv/**`, `**/node_modules/**`, `**/coverage/**`, common
  build/cache dirs). No vendor trees like anyka’s `anyka_reference/`.
- `.github/codeql/README.md` — Default Setup is primary; after changing the config,
  **Settings → Code security → CodeQL analysis → Edit → Save**; advanced workflow
  only as fallback if Default Setup ignores the file (do **not** commit advanced
  workflow in this change).

### 3. Docs

- Update `AGENTS.md` CI / Dependency Security sections to name **pip-audit + CodeQL
  + SonarCloud + Dependabot** (no Snyk).
- Scrub any leftover Snyk mentions elsewhere if introduced.

### 4. Gate behaviour

Unchanged aside from deleting Snyk:

- Merge gates: lint, typecheck, test, `pip-audit`
- SonarCloud as today
- CodeQL via Default Setup (Security → Code scanning), not a `ci.yml` job

### 5. Verification

1. `rg -in 'snyk|SNYK' .github/ AGENTS.md docs/ .opencode/` → empty
2. Confirm `ci.yml` still has `audit` + `sonarcloud`; no `snyk`
3. Post-merge manual: delete `SNYK_TOKEN`; re-save CodeQL Default Setup

## Consequences

- Less CI work and one fewer vendor/token in the supply chain
- SCA/SAST ownership is clearer and matches anyka’s pattern (Python-appropriate tools)
- Operator must perform two one-time GitHub UI steps after merge (secret + CodeQL save)
