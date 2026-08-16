# CodeQL configuration

This directory holds the CodeQL config used to exclude tests, venvs, and
cache/build paths from analysis.

## Default Setup (GitHub UI)

This repo uses **CodeQL Default Setup** (Settings → Code security and analysis).

Default Setup does **not** auto-load this file from disk alone. Point the
repository property `github-codeql-config-file` at
`.github/codeql/codeql-config.yml` (Settings → Custom properties, or org
defaults), then refresh Default Setup so the merge takes effect.

1. Set `github-codeql-config-file` = `.github/codeql/codeql-config.yml`
2. After changing `codeql-config.yml` (or the property), refresh Default Setup:
   - **Settings** → **Code security and analysis** → **CodeQL analysis** (Default) → **Edit**
   - Click **Save changes** (or disable then re-enable Default Setup).
3. Re-run CodeQL on a PR (re-run jobs or push) to confirm ignored paths drop out.

## Advanced Setup (fallback)

If Default Setup does not honor this config, switch to Advanced Setup with a
workflow that passes `config-file: ./.github/codeql/codeql-config.yml` to
`github/codeql-action/init`, and **disable Default Setup** to avoid duplicate runs.
Do not enable both at once.
