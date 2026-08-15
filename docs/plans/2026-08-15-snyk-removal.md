# Snyk Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the Snyk CI job and document the remaining security stack (pip-audit + CodeQL Default Setup + SonarCloud + Dependabot), matching anyka-dev’s approach without adding a Rust-only replacement.

**Architecture:** Delete the `snyk` job from `ci.yml`. Add `.github/codeql/codeql-config.yml` + README for Default Setup path ignores. Update `AGENTS.md`. Manual post-merge: delete `SNYK_TOKEN` and re-save CodeQL Default Setup. Design: `docs/plans/2026-08-15-snyk-removal-design.md`.

**Tech Stack:** GitHub Actions, CodeQL Default Setup, pip-audit, SonarCloud, Dependabot, Markdown docs.

---

### Task 1: Delete the Snyk job from CI

**Files:**
- Modify: `.github/workflows/ci.yml:124-174` (entire `snyk:` job through the skip-notice step; leave `sonarcloud:` intact)

**Step 1: Remove the job**

Delete from the blank line before `snyk:` through the end of the Skip notice step (inclusive), so `audit:` is immediately followed by `sonarcloud:`.

Do **not** change the `audit` or `sonarcloud` jobs.

**Step 2: Confirm structure**

Run:

```bash
rg -n '^(  )?[a-z].*:' .github/workflows/ci.yml | head -40
rg -in 'snyk|SNYK' .github/workflows/ci.yml
```

Expected: job names include `lint`, `typecheck`, `test`, `audit`, `sonarcloud` only; second command prints nothing.

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: remove Snyk security scan job

SCA/SAST remain covered by pip-audit, CodeQL Default Setup, and SonarCloud.
EOF
)"
```

---

### Task 2: Add CodeQL path-ignore config (anyka-shaped)

**Files:**
- Create: `.github/codeql/codeql-config.yml`
- Create: `.github/codeql/README.md`

**Step 1: Create config**

Write `.github/codeql/codeql-config.yml`:

```yaml
name: "CodeQL Configuration"

# Used by CodeQL Default Setup when present under .github/codeql/.
# Reference: https://docs.github.com/en/code-security/code-scanning/creating-an-advanced-setup-for-code-scanning/customizing-advanced-setup-for-code-scanning#specifying-directories-to-scan

paths-ignore:
  - "**/tests/**"
  - "**/.venv/**"
  - "**/node_modules/**"
  - "**/coverage/**"
  - "**/dist/**"
  - "**/build/**"
  - "**/__pycache__/**"
  - "**/.pytest_cache/**"
  - "**/.ruff_cache/**"
  - "**/.mypy_cache/**"
```

**Step 2: Create README**

Write `.github/codeql/README.md` (adapt from anyka-dev; llm-runner has no vendor trees):

```markdown
# CodeQL configuration

This directory holds the CodeQL config used to exclude tests, venvs, and
cache/build paths from analysis.

## Default Setup (GitHub UI)

This repo uses **CodeQL Default Setup** (Settings → Code security and analysis).

1. After changing `codeql-config.yml`, refresh the configuration:
   - **Settings** → **Code security and analysis** → **CodeQL analysis** (Default) → **Edit**
   - Click **Save changes** (or disable then re-enable Default Setup).
2. Re-run CodeQL on a PR (re-run jobs or push) to confirm ignored paths drop out.

## Advanced Setup (fallback)

If Default Setup does not honor this config, switch to Advanced Setup with a
workflow that passes `config-file: ./.github/codeql/codeql-config.yml` to
`github/codeql-action/init`, and **disable Default Setup** to avoid duplicate runs.
Do not enable both at once.
```

**Step 3: Commit**

```bash
git add .github/codeql/codeql-config.yml .github/codeql/README.md
git commit -m "$(cat <<'EOF'
ci: add CodeQL path-ignore config for Default Setup

Exclude tests, venvs, and cache dirs from CodeQL noise, following anyka-dev.
EOF
)"
```

---

### Task 3: Update AGENTS.md security / CI docs

**Files:**
- Modify: `AGENTS.md` (sections `## CI / Pre-commit` and `## Dependency Security Policy`)

**Step 1: Extend CI / Pre-commit**

After the bullet list that ends with the audit job mention (~lines 200–207), ensure the text states:

- Merge checks: lint, typecheck, test
- Additionally: `pip-audit` (CVE scan), SonarCloud (quality / SAST gate), CodeQL Default Setup (SAST in GitHub code scanning)
- Do **not** mention Snyk

Suggested replacement for the “Additionally…” paragraph:

```markdown
Additionally:

- **audit** — `uv run pip-audit` for known CVEs in dependencies
- **SonarCloud** — quality gate / SAST on PRs and pushes (when `SONAR_TOKEN` is set)
- **CodeQL** — GitHub Default Setup code scanning (configured under `.github/codeql/`)
```

**Step 2: Extend Dependency Security Policy**

Under `### CI Dependency Scan`, clarify the full stack:

```markdown
### CI Dependency Scan

CI runs `uv run pip-audit` on every push and pull request to detect known CVEs
in dependencies. Dependabot opens weekly update PRs for `pip` and
`github-actions`. SAST is covered by CodeQL Default Setup and SonarCloud — not
by a third-party SCA/SAST vendor CLI in this workflow.
```

Keep vulnerability cadence and routine refresh sections as-is (still `pip-audit`).

**Step 3: Grep scrub**

```bash
rg -in 'snyk|SNYK' AGENTS.md docs/ .opencode/ .github/
```

Expected: empty (design doc may mention Snyk historically — that is fine; if the design path matches, allow `docs/plans/2026-08-15-snyk-removal*.md` only).

Stricter check excluding plans:

```bash
rg -in 'snyk|SNYK' AGENTS.md .opencode/ .github/ --glob '!docs/plans/**'
```

Expected: empty.

**Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "$(cat <<'EOF'
docs: document pip-audit, CodeQL, and Sonar as the security stack

Replace any implication of a Snyk CI job with the tools that remain.
EOF
)"
```

---

### Task 4: Final verification + PR notes

**Files:** none (verification only)

**Step 1: Repo grep**

```bash
rg -in 'snyk|SNYK' .github/workflows/ AGENTS.md .opencode/
```

Expected: no matches.

**Step 2: Optional actionlint**

If installed:

```bash
actionlint .github/workflows/ci.yml
```

Expected: no errors.

**Step 3: Open / update PR body checklist**

Include:

```markdown
## Post-merge (manual)
- [ ] Delete repo secret `SNYK_TOKEN` (Settings → Secrets and variables → Actions)
- [ ] CodeQL Default Setup → Edit → Save so `.github/codeql/codeql-config.yml` is applied
- [ ] Confirm next CodeQL run respects `paths-ignore` (tests/venv noise reduced)
```

**Step 4: Push** when the user asks (do not push unless requested).

---

### Out of scope (do not implement in this plan)

- GitHub Dependency review action
- OSV-Scanner / Semgrep
- Changing `pip-audit` ignore-vuln list
- Committing an Advanced Setup `codeql.yml` workflow
- Deleting the GitHub secret via API (manual only)
