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
