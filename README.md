# llm-runner

A terminal-based user interface for managing multiple llama-server instances
with live logs, configuration display, and GPU monitoring.

**Current Branch Scope:** The `001-prd-mvp-spec` branch implements **PRD Milestone M1 only**
(slot-first orchestration, dry-run, validation), not the full PRD MVP. See
[PRD Spec-001 Compliance Review](docs/reviews/prd-spec-001-compliance-review.md) for
detailed scope and deferred items.

## Setup

```bash
cd /path/to/llm-runner
source .venv/bin/activate
```

## Usage

```bash
# Run both models side-by-side
python src/run_models_tui.py both

# Run with custom ports
python src/run_models_tui.py both --port 8080 --port2 8081

# Run single model
python src/run_models_tui.py summary-balanced --port 8080
python src/run_models_tui.py qwen35 --port 8081
python src/run_models_tui.py summary-fast

# Get help
python src/run_models_tui.py --help

# Or use the llm-runner CLI script (for CLI mode only)
uv run llm-runner both
```

## Features

- **2-column layout**: View two models side-by-side (auto-switches to single
column on small terminals)
- **Live logs**: Real-time stdout/stderr from each server process
- **GPU stats**: Monitor GPU utilization, memory, temperature, and power draw
via nvtop
- **Config display**: Shows port, device, context size, threads, and batch size
- **Auto-scroll**: Logs automatically scroll to show newest output
- **Resize support**: Layout adapts to terminal size changes

## GPU Device Mapping

- **NVIDIA (CUDA)**: GPU 0 (RTX 3090) - used by qwen35-coding
- **Intel (SYCL)**: GPU 1 (Arc B580) - used by summary-balanced, summary-fast

## Security

### Dependency Security

We take dependency security seriously. All CI runs include automated dependency
auditing via `pip-audit`.

#### CI Dependency Scan

CI automatically runs `uv run pip-audit` on every push and pull request to detect
known CVEs in dependencies. The audit job is part of the CI workflow but does not
block merging — it provides visibility into potential vulnerabilities.

#### Local Pre-release Check

Before merging or releasing, run:

```bash
uv run pip-audit
```

#### Vulnerability Response Cadence

| Severity | Response Target |
| -------- | --------------- |
| Critical | Immediately — patch or pin within 24h |
| High     | Within 1 week |
| Medium   | Within 1 month |
| Low      | Included in routine dependency refresh |

#### Routine Dependency Refresh

Quarterly (or before major releases), update all dependencies:

```bash
uv pip compile pyproject.toml --upgrade
uv sync
uv run pip-audit
```

Review `pip-audit` output and update dependencies via `uv add --upgrade-package <pkg>`.

### Snyk CI Integration

Snyk provides continuous security scanning in CI via two checks:

- **Snyk Open Source** — scans Python dependencies for known CVEs
- **Snyk Code** — scans source code for security vulnerabilities (SAST)

Both scans run on every push to `main` and on every pull request targeting `main`.
CI **fails** on findings at severity level **high** or above.

The GitHub Actions secret `SNYK_TOKEN` must be configured in repository settings.
Pull requests from forks may skip Snyk checks because fork workflows cannot access
repository secrets.

---

The inference servers bind to `127.0.0.1:8080` and `127.0.0.1:8081` by default,
making them accessible only from localhost. Do not expose these ports to external
networks without configuring proper authentication and firewall rules.

## Exit

Press `Ctrl+C` to gracefully stop all servers and exit.
