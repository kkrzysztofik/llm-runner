# Feature Specification: M4 — Operational Hardening and Smoke Verification

**Feature Branch**: `003-m4-op-hardening`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "M4 from PRD"

## Clarifications

### Session 2026-04-23

- **Q**: What are the valid operational states for a slot, given the TUI must show per-slot status and handle crashes/OOMs? → **A**: Six-state model: idle, launching, running, degraded, crashed, offline.
- **Q**: When running `smoke both`, if one slot fails, should probing continue to remaining slots or stop immediately? → **A**: Continue probing remaining slots in declaration order, then exit non-zero with a composite report.
- **Q**: What should `smoke` do if a slot is still in the `launching` state when probed? → **A**: Wait up to `tui_launch_timeout_s` (default 120 s), then treat as not-ready (exit code 10) if still not accepting connections.
- **Q**: Which OpenAI API version should the smoke probe target? → **A**: OpenAI API v1 (current stable).
- **Q**: How should concurrent TUI and CLI access be handled to prevent race conditions? → **A**: Shared lockfiles prevent concurrent mutating commands (launch, shutdown); read-only commands (smoke, doctor) run concurrently.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Launch, Monitor, and Shutdown Two Models Safely (Priority: P1)

As a solo operator with a mixed-GPU workstation, I want to launch one LLM serving process per GPU slot, see real-time health, logs, and GPU telemetry in a single terminal view, and shut down gracefully without leaving orphan GPU processes on the system.

**Why this priority**: This is the core operational loop. Without reliable launch, monitoring, and shutdown, the tool cannot be trusted for daily use, and orphan processes waste VRAM and destabilize the workstation.

**Independent Test**: Start both models from the TUI, confirm per-slot status and logs are visible, then trigger shutdown and verify no `llama-server` processes remain.

**Acceptance Scenarios**:

1. **Given** both GPU slots are healthy and configured, **When** the user launches both models, **Then** the TUI shows each slot’s status, live logs, GPU telemetry, backend label, and any profile or build warnings.
2. **Given** one model crashes or is killed externally, **When** the TUI updates, **Then** the failed slot is marked offline while the sibling continues running; the user receives a clear hint about the failed slot.
3. **Given** the user initiates shutdown or the TUI exits abnormally, **When** the shutdown sequence runs, **Then** each child server receives SIGTERM, waits for exit, escalates if needed, and no orphan processes remain.

---

### User Story 2 — Verify Serving with OpenAI-Compatible Smoke Tests (Priority: P1)

As a systems user, I want to run a sequential verification that each slot responds to OpenAI-compatible endpoints, so I can confirm the serving stack is healthy before relying on it for downstream tools.

**Why this priority**: `smoke both` is the MVP completion gate. It provides confidence that launch produced working endpoints and catches flag drift or model misconfiguration early.

**Independent Test**: Run `smoke both` against running servers and confirm it exits zero after probing each slot sequentially; run `smoke slot <id>` for partial probing.

**Acceptance Scenarios**:

1. **Given** all configured slots are serving, **When** the user runs `smoke both`, **Then** the first slot is probed (listen/accept, `/v1/models` when present, minimal chat completion), then after a pause the next slot is probed, continuing until all configured slots have been processed; the command exits zero only if all pass.
2. **Given** only one slot is operable, **When** the user runs `smoke slot <slot_id>`, **Then** the command probes that single slot and exits zero on success without faking a second model.
3. **Given** a chat probe is performed, **When** the server responds, **Then** the probe uses a minimal prompt, temperature 0, and a small token limit, and the resolved model ID and provenance are printed clearly.

---

### User Story 3 — Inspect Model Metadata Without Loading Weights (Priority: P2)

As a model experimenter, I want to inspect GGUF metadata (architecture, context length, attention heads, attention heads KV, embedding length, block count, tokenizer type, and name) quickly and safely without loading the full weight file into memory, so I can validate model compatibility and infer smoke identifiers.

**Why this priority**: Metadata inspection supports pre-flight validation, filename normalization for smoke IDs, and VRAM estimation without the cost of full model load.

**Independent Test**: Point the tool at a GGUF file and confirm metadata fields are extracted within a bounded time and memory budget; confirm timeout/corrupt handling produces clear errors.

**Acceptance Scenarios**:

1. **Given** a valid unsharded GGUF file, **When** metadata is requested, **Then** the tool reads only a limited prefix from the file start, parses the header, and returns the required fields within a few seconds.
2. **Given** a file that exceeds the parse timeout or appears corrupt, **When** the tool attempts extraction, **Then** it reports a clear error distinguishing parser mismatch from corruption, without hanging or loading weights.
3. **Given** a file missing `general.name`, **When** metadata is displayed or used for smoke, **Then** the normalized filename stem is used as a fallback, and the raw path and normalized name are visible in diagnostics.

---

### User Story 4 — Acknowledge Hardware and VRAM Risks Before Launch (Priority: P2)

As an operator on a non-anchor or constrained workstation, I want to be warned when hardware differs from the expected profile or when free VRAM looks insufficient for the chosen model, with explicit confirmation paths, so I can proceed knowingly or abort safely.

**Why this priority**: Prevents accidental launches on wrong hardware or out-of-memory crashes that destabilize the system. Supports both interactive TUI and scripted CLI use.

**Independent Test**: Simulate a non-standard GPU topology and verify warnings appear; simulate low free VRAM and verify launch/smoke blocks without explicit confirmation.

**Acceptance Scenarios**:

1. **Given** the detected GPU topology does not match the anchored workstation profile, **When** the user attempts launch from the TUI, **Then** a warning is shown with details, and the user must explicitly choose Continue before proceeding.
2. **Given** the same mismatch from the CLI, **When** the user provides the documented acknowledgment flag, **Then** the warning is logged but launch proceeds for this invocation.
3. **Given** free VRAM (per best-effort query) minus a safety margin appears insufficient for the model, **When** the user attempts launch or full-model smoke, **Then** a warning requires explicit confirmation; without confirmation, the operation does not proceed.

---

## Requirements *(mandatory)*

### Functional Requirements

_Spec FR numbers are independent of PRD FR numbers. For traceability, PRD FR-010→spec FR-001; FR-011→spec FR-019; FR-012→spec FR-010; FR-014→spec FR-007–009; FR-015→spec FR-002–006; FR-016→spec FR-013–014; FR-017→spec FR-011–012; FR-018→spec FR-016._

- **FR-001**: The system MUST display per-slot status, live logs, GPU telemetry, active backend label, and profile staleness warnings.
- **FR-002**: The system MUST provide `smoke both` and `smoke slot <slot_id>` commands that probe OpenAI-compatible endpoints sequentially, with a configurable pause applied after each successful slot completion (not before the first slot). `smoke both` runs slots in declaration order from config, starting with the first slot and proceeding sequentially. Smoke probes operate independently of TUI slot state — they probe by port directly (attempting TCP connection). The `launching` state wait (up to `tui_launch_timeout_s`) is relevant ONLY when smoke is invoked from the TUI monitor context. For CLI smoke, the probe simply attempts the TCP connection and times out if the server isn't ready. If a slot is in the `launching` state when probed (TUI context), the probe waits up to `tui_launch_timeout_s` (default 120 s) before treating the slot as not-ready. If the slot transitions to `running` during the wait, the probe proceeds to the next phase. If it transitions to `crashed`, the probe fails with exit code 19. Each smoke probe phase is attempted exactly once — no retries. Failure at any phase is immediate and non-recoverable for that probe session. If a slot fails, probing continues to the next slot in order; the command exits non-zero if any slot fails, producing a composite report. The default inter-slot delay is 2 seconds (`smoke.inter_slot_delay_s`).
- **FR-003**: Smoke probes MUST target the OpenAI API v1 (current stable) interface. Probes progress through listen/accept (using `smoke.listen_timeout_s`, default 120 s), optional `/v1/models` discovery, and a minimal chat completion phase, recording which phase fails when a probe does not pass. All HTTP requests within smoke probes use `smoke.http_request_timeout_s` (default 10 s) as the request timeout to prevent indefinite hangs. The `/v1/models` discovery phase is optional: it is skipped if the listen/accept probe fails, or if `smoke.skip_models_discovery` is set to `true` in config. If `/v1/models` returns 404, the phase is marked as 'skipped — endpoint not supported' and the probe proceeds to chat completion. If it returns 5xx or connection refused, the phase is marked as 'failed'. If `/v1/models` returns HTTP 200 with an empty `models` array, the phase is marked as 'failed — no models available' and the probe exits with code 13 (model not found). If `/v1/models` returns HTTP 200 with a non-empty `models` array, the probe compares the first model's `id` field with the expected model ID. A mismatch results in exit code 13 with failure reason 'model ID mismatch'. Network-layer errors (DNS failure, SSL error, connection reset) are mapped to the exit code of the phase in which they occurred: listen/accept network errors → exit 10; HTTP/API network errors → exit 11. Smoke probes connect by port directly (not dependent on TUI slot state), as clarified in FR-002.
- **FR-004**: Smoke chat probes MUST use temperature 0 and a configurable token limit within an allowed range, with a minimal single-user prompt. The default `max_tokens` is 16 with an allowed range of 8–32. The smoke chat probe sends a single-turn user message with the text `Respond with exactly one word.` (configurable via `smoke.prompt`, default as shown). No system prompt is included.
- **FR-005**: The model ID used in smoke chat probes MUST resolve via the following precedence chain (matching PRD FR-015): (1) GGUF `general.name` (highest), (2) normalized filename stem, (3) catalog `smoke.model_id` / override, (4) optional `/v1/models` match. If both `general.name` and the normalized filename stem are empty or invalid after normalization, use the raw file path as the smoke ID, prefixed with `path:` to distinguish it. The smoke header prints the resolved id and provenance (binary tip SHA, tool version). The smoke header is a single line printed to stdout (CLI mode) or displayed in the smoke result panel (TUI mode) in the format: `smoke: slot={slot_id} model={model_id} provenance={sha},{version}`. Tool version is obtained via `importlib.metadata.version('llm_runner')` or equivalent package metadata lookup. If the package metadata is unavailable, falls back to `'dev'`. Binary tip SHA is obtained by: (1) resolving `Path(Config().llama_cpp_root) / '.git' / 'HEAD'`, (2) if the path exists, running `git rev-parse HEAD` with a 5-second timeout, (3) on any failure (missing git, timeout, non-zero exit), using `'unknown'`. A future iteration may add a user-provided catalog override; for MVP, the resolution chain stops at `/v1/models` match.
- **FR-006**: If two consecutive smokes for the same slot fail with model-not-found attributable to a wrong ID, the system MUST require an explicit `smoke.model_id` override before further attempts. The 'two consecutive failures' counter is scoped to a single `llm-runner` process lifetime (in-memory only). A process restart resets the counter. This prevents requiring an override after a transient failure followed by a restart. The counter increments only when the smoke probe exits with code 13 (model not found). Other exit codes (timeout, auth failure, connection error) do not increment the counter. If the user provides a `smoke.model_id` override that fails to match on the next attempt, the counter is NOT reset. The user must provide a correct override; otherwise the counter continues incrementing and the override requirement remains.
- **FR-007**: The system MUST extract GGUF metadata for `general.name`, `general.architecture`, `tokenizer.type`, `llama.embedding_length`, `llama.block_count`, `llama.context_length`, `llama.attention.head_count`, and `llama.attention.head_count_kv` without loading full weights. Metadata extraction uses the pinned `gguf` PyPI dependency (per PRD FR-014) to read only the first N bytes of the file. The parser reads the GGUF header format: magic bytes `GGUF` (4 bytes), version as little-endian uint32 (4 bytes), key-value count as little-endian uint64 (8 bytes), followed by key-value pairs. Each key is a string (length prefix + UTF-8 bytes). Values are typed (uint32, uint64, float32, string, array). The parser reads only the header region up to the prefix cap. It MUST support standard unsharded GGUF v3 files. Files containing only header metadata (no tensor data) are also valid for metadata-only inspection. The parser MUST support GGUF v3 files (current stable as of llama.cpp release). GGUF v4 or later files MUST be reported as 'unsupported format version' with a clear error. `doctor` MUST distinguish a corrupt file from a parser/spec mismatch (e.g. GGUF version).
- **FR-008**: GGUF metadata extraction MUST enforce a prefix read cap and a wall-clock parse timeout; on timeout or parse failure, the system MUST treat the file as unsupported/corrupt with a clear, specific error. The default prefix read cap is 32 MiB (tunable in `Config`). The default wall-clock parse timeout is 5 seconds (tunable). Unicode normalization uses NFKC. Optional strip quant suffix patterns are supported via a workstation-tuned list, off by default. `doctor` dry-run/metadata view shows the raw path, normalized stem, and resolved smoke name. The test fixtures policy requires synthetic GGUF files under `src/tests/fixtures/`, produced once by a documented maintainer script; CI consumes committed bytes only. Synthetic fixtures are generated by `src/scripts/generate_gguf_fixtures.py`, which creates minimal valid GGUF files containing only header metadata (no tensor data) using the pinned `gguf` PyPI dependency. CI consumes committed fixture files only; the generator script is documented in `src/tests/fixtures/README.md`. Minimum fixtures: (1) valid GGUF v3 with all required keys, (2) valid GGUF v3 missing `general.name`, (3) corrupt file (bad magic bytes), (4) truncated file (valid header, no key-value data), (5) valid GGUF v4 file with all required keys, expected to produce 'unsupported format version' error. All fixtures must be under 10 KiB.
- **FR-009**: The system MUST normalize GGUF filenames for display and smoke ID inference using a deterministic pipeline including Unicode normalization (NFKC), whitespace replacement, and optional lowercase/quant-stripping. After NFKC normalization, invalid filename characters (null bytes, path separators) are replaced with underscores.
- **FR-010**: The system MUST perform graceful shutdown via SIGTERM with a wait-and-escalate policy; abnormal TUI exit MUST terminate child servers to avoid orphan GPU processes. After sending SIGTERM, the system waits for process exit (normal exit code 143 for SIGTERM). If the process does not exit, SIGKILL is sent. After sending SIGKILL, the system polls for process termination. If the process is still alive after a 2-second poll, the system logs a CRITICAL error, marks the slot as `offline`, and records the failure. The system continues shutting down remaining slots and reports all failures in the audit log. After all slots have been processed, the tool exits with code 130 if any slot required SIGKILL escalation (process stuck in D state). Note: exit code 130 is also the standard shell exit code for SIGINT (128 + 2); this SIGKILL-escalation 130 is distinct from a normal SIGINT-caused exit. Exit code 130 for SIGKILL escalation is reserved for process-manager failures and is outside the doctor (1-9) and smoke (10-19) families.
- **FR-011**: The system MUST warn (not hard-fail) when hardware topology differs from the anchored workstation profile, requiring explicit user acknowledgment in the TUI or a CLI flag. In the TUI, an explicit **Continue** button is shown after the user reads the warning details. The CLI provides `--ack-nonstandard-hardware`. If the user selects Cancel or presses Escape, the launch aborts and the slot remains in the `idle` state. The warning can be re-triggered in the same session by attempting launch again. The TUI displays the warning in the status panel and waits for a keypress via the TUI's key handler: `y` to continue, `n` or `q` to abort. If the user presses Enter without `y`, `n`, or `q`, the prompt is re-displayed. If no input is received within 30 seconds, the prompt times out and the safe default (`n` / abort) is applied. The `Live` context remains active throughout; the prompt is rendered as a layout update. During the warning prompt, the TUI layout keeps all existing panels visible. The warning text replaces the content of the slot's status panel, and a prompt line (`[y] Continue  [n] Abort`) is appended at the bottom of the layout. Log panels continue to update if the slot was already running. **Implementation note**: Rich Live does not natively support blocking keypresses while keeping the live display. The implementation uses a non-blocking input check pattern: the TUI reads from stdin via `select` (on POSIX) or `msvcrt` (on Windows) with a 0-second timeout, checking for keypresses on each render cycle. The key handler runs as part of the Rich `Live` render callback, so the display continues updating while polling for input. Alternatively, the prompt can be rendered as a layout panel update with key polling driven by the TUI refresh interval (`tui_refresh_interval_ms`), ensuring the render loop is never blocked.
- **FR-012**: The system MUST maintain a persistent hardware allowlist invalidated automatically when the machine fingerprint or PCI device set changes, and a session snooze file to suppress duplicate warnings during a single runtime session. The persistent allowlist lives at `~/.config/llm-runner/hardware-allowlist.json`. The session snooze file lives under `$XDG_RUNTIME_DIR/llm-runner/` (fallback `/tmp/llm-runner/`). The machine fingerprint is computed as `SHA256(lspci_gpu_output + "|" + sycl_ls_output)`, where `lspci_gpu_output` is the output of `lspci -d ::0300` (VGA/3D controllers) and `sycl_ls_output` is the output of `sycl-ls`. If `lspci` is unavailable, the VGA/3D controller component is omitted from the fingerprint. If `sycl-ls` is unavailable, the SYCL component is omitted. If both are unavailable, the fingerprint is empty and hardware warnings are always triggered (since the allowlist cannot match). Both `lspci` and `sycl-ls` outputs are normalized before hashing: (1) strip leading/trailing whitespace per line, (2) remove empty lines, (3) sort lines lexicographically, (4) join with newlines.
- **FR-013**: The system MUST warn when free VRAM appears insufficient for a model load, requiring explicit confirmation before launch or full-model smoke proceeds. The VRAM heuristic formula is: warn if `free_vram × 0.85 < gguf_file_size × 1.2`, where `free_vram` is best-effort free memory from `nvidia-smi` (NVIDIA) or the Intel Arc equivalent, and `gguf_file_size` is the on-disk file size of the model weights. If GPU telemetry is unavailable, the check is skipped with a logged warning. The CLI provides `--confirm-vram-risk` for scripted confirmation. The heuristic is intentionally conservative: the combined multiplier (0.85 × 1.2 ≈ 1.02) means the check triggers when free VRAM is approximately equal to or less than the model file size. This accounts for runtime overhead but does not guarantee sufficient memory for all workloads.
- **FR-014**: The system MUST use per-slot lockfiles under a runtime directory and refuse launch if a slot lock is already held, with a diagnostic command to surface lock holders. Lockfiles prevent concurrent mutating commands (launch, shutdown) on the same slot; read-only commands (smoke, doctor) may run concurrently with each other and with an active slot. The default bind address is `127.0.0.1` with no API key unless configured otherwise. Port-in-use refusal can be overridden with `--force-bind`, which is logged as dangerous. `doctor --locks` surfaces lock holders. Lockfiles contain the owning PID and acquisition timestamp. Before rejecting a launch due to an existing lock, the tool checks if the owning PID is still alive (via `psutil`). If the PID is dead or the lock is older than `lock_stale_threshold_s` (default 300 s), the lock is considered stale and is removed.
- **FR-015**: The system MUST document non-overlapping exit code families for `doctor` and `smoke`, with `smoke` using a distinct band from `doctor`. `doctor` uses exit codes 1–9; `smoke` uses exit codes 10–19. Specific codes are defined in Appendix B.
- **FR-016**: The system MUST append mutating actions to a rotating log that includes command, timestamp, exit code, and truncated output, redacting obvious secret patterns from logs and reports. When server auth is enabled, `smoke` MUST accept an API key via CLI, config, or environment without emitting the key into logs. Auth accepts `--api-key`, config `smoke.api_key`, or the environment variable `LLM_RUNNER_SMOKE_API_KEY` (exact name documented in CLI help). Precedence (highest to lowest): `--api-key` CLI flag > `smoke.api_key` config > `LLM_RUNNER_SMOKE_API_KEY` environment variable. Failure drops a report directory under `Config().reports_dir / '<timestamp>'` (default: `~/.local/share/llm-runner/reports/<timestamp>/`) containing plain files.
- _(FR-017 and FR-018 consolidated into FR-011, FR-012, and FR-016 per PRD alignment.)_
- **FR-019**: The system MUST support stable `--json` output for `smoke` (schema defined in FR-020) and `doctor` (schema defined below). The `doctor --json` output is: `{ 'checks': [{ 'name': str, 'status': 'pass' | 'warn' | 'fail', 'message': str | null }], 'config': { 'effective_values': dict }, 'hardware': { 'fingerprint': str, 'gpus': list } }`.
- **FR-020**: The `smoke` command MUST support two output modes: (1) human-readable text printed to stdout with a per-slot summary and composite report footer, and (2) structured JSON via `--json` flag. The JSON format is: `{ 'slots': [{ 'slot_id': str, 'status': str, 'phase_reached': str, 'failure_phase': str | null, 'model_id': str | null, 'latency_ms': int | null, 'provenance': { 'sha': str, 'version': str } }], 'overall_exit_code': int }`. Enum definitions for typed fields:
  - `status`: `"pass" | "fail" | "timeout" | "crashed" | "model_not_found" | "auth_failure"`
  - `phase_reached`: `"listen" | "models" | "chat" | "complete"`
  - `failure_phase`: null when `status` is `"pass"`, otherwise the phase where failure occurred (`"listen"`, `"models"`, or `"chat"`).
Per-slot exit codes are included in the composite report.

### Smoke Output Formats

The `smoke` command produces output in two modes:

- **Human-readable (default)**: Per-slot summary printed to stdout, followed by a composite report footer listing all slot results and the overall exit code.
- **Structured JSON (`--json` flag)**: Single JSON document matching the FR-020 schema, suitable for programmatic consumption.

### CLI Smoke Command

`smoke` is a new top-level command alongside `both`, `summary-balanced`, `doctor`, `dry-run`, `build`, and `setup`. The CLI parser adds `smoke` to `VALID_MODES` and handles it as a special case with subcommands, following the same pattern as `doctor` and `setup`.

**Subcommands:**

- `smoke both` — Probe all configured slots sequentially, in declaration order from config. Exits zero only if all slots pass.
- `smoke slot <slot_id>` — Probe a single slot by ID (e.g., `smoke slot arc_b580`). Exits zero if the slot passes.
- `smoke --json` — Output JSON instead of human-readable (applies to both `both` and `slot` subcommands).
- `smoke --api-key <key>` — Override the API key for smoke probes (takes precedence over config and environment).
- `smoke --model-id <id>` — Override the model ID for smoke chat probes. Useful when the resolved model ID from GGUF metadata is incorrect.

**Argument structure (pseudo-argparse):**

```python
# Top-level smoke handler (handled before normal mode parsing)
if args[0] == "smoke":
    if len(args) < 2:
        error("smoke requires a subcommand: both | slot <slot_id>")
    subcommand = args[1]
    if subcommand == "both":
        parse_smoke_both_args(args[2:])
    elif subcommand == "slot":
        parse_smoke_slot_args(args[2:])
    else:
        error(f"unknown smoke subcommand '{subcommand}'. Valid: both, slot")
```

**`smoke both` arguments:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--json` | flag | `false` | Output structured JSON instead of human-readable text |
| `--api-key <key>` | string | (from config/env) | Override API key for smoke probes |
| `--model-id <id>` | string | (from GGUF metadata) | Override model ID for smoke chat probes |
| `--max-tokens <n>` | int | 16 (`smoke.max_tokens`) | Max tokens in smoke chat probe (range 8–32) |
| `--prompt <text>` | string | `"Respond with exactly one word."` (`smoke.prompt`) | Single-turn user message for chat probe |
| `--delay <seconds>` | int | 2 (`smoke.inter_slot_delay_s`) | Pause between slot probes |
| `--timeout <seconds>` | int | 120 (`smoke.listen_timeout_s`) | TCP ready-check timeout per slot |

**`smoke slot <slot_id>` arguments:**

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--json` | flag | `false` | Output structured JSON instead of human-readable text |
| `--api-key <key>` | string | (from config/env) | Override API key for smoke probes |
| `--model-id <id>` | string | (from GGUF metadata) | Override model ID for smoke chat probes |
| `--max-tokens <n>` | int | 16 (`smoke.max_tokens`) | Max tokens in smoke chat probe (range 8–32) |
| `--prompt <text>` | string | `"Respond with exactly one word."` (`smoke.prompt`) | Single-turn user message for chat probe |
| `--timeout <seconds>` | int | 120 (`smoke.listen_timeout_s`) | TCP ready-check timeout |

**Exit codes:** Follow the smoke exit code family (10–19) defined in Appendix B. `smoke both` exits with the highest-severity (lowest-numbered) failure among all slots.

### Non-Functional Requirements

- **NFR-002**: Failures MUST attribute to slot → model → process → phase, and MUST include remediation hints.
- **NFR-003**: Dry-run and config resolve MUST feel immediate; long operations MUST show phase progress.
- **NFR-006**: Polling and log buffer defaults MUST keep the TUI responsive; caps MUST be documented; `doctor` MUST print effective values.
- **NFR-007**: Diagnostics MUST support an optional `--redact-paths` flag for share bundles; default logs MUST favor local debuggability.

### Constitution Alignment *(mandatory)*

- **CA-001 Code Quality**: All changes in monitoring, smoke, metadata, and shutdown paths must preserve `llama_manager/` as a pure library (no argparse, no Rich, no subprocess at module level) and pass `ruff` + `pyright`.
- **CA-002 Testing**: Unit tests must cover GGUF metadata extraction with committed synthetic fixtures, smoke phase logic with mocked HTTP, shutdown signal flow, and VRAM/heuristic decision trees. No subprocess spawning of real servers in CI.
- **CA-003 UX Consistency**: TUI and CLI must behave parity for smoke results, dry-run must show OpenAI flag bundles and compatibility matrix rows, and diagnostics must attribute failures to slot → model → process → phase with remediation hints.
- **CA-004 Safety and Observability**: Log entries for mutating actions must include command, timestamp, exit code, and truncated output; secret patterns must be redacted; report directories must be plain files under a predictable path.

### Key Entities *(include if feature involves data)*

- **Slot**: A GPU slot owning bind host, port, backend flavor, environment overlays, and at most one active model process. Identified by a slot ID (e.g., `arc_b580`, `rtx3090`). A slot transitions through the following operational states:
  - `idle` — no process assigned;
  - `launching` — process start initiated, not yet accepting connections;
  - `running` — process healthy and accepting connections;
  - `degraded` — process running but telemetry or probe indicates reduced health;
  A slot transitions to `degraded` when it is running but exhibits reduced health: probe latency exceeds a configurable threshold (default 10s), GPU telemetry reports errors but the process still responds, or the log buffer shows repeated error patterns (defined as 3 or more error-level log lines within a 60-second sliding window).
  - `crashed` — process exited unexpectedly (non-zero exit code or signal);
  - `offline` — process is unresponsive (not accepting connections but PID may still exist). An offline slot may be reached from any active state (`running`, `degraded`, `crashed`) when the process becomes unresponsive (e.g., zombie, GPU hang, TCP listen failure).

  State transitions:

  | From | To | Trigger |
  | --- | --- | --- |
  | idle | launching | launch initiated |
  | launching | running | process accepts connections |
  | launching | crashed | process exits before accepting |
  | running | degraded | probe latency > threshold OR GPU errors OR repeated log errors |
  | running | crashed | process exits unexpectedly |
  | running | offline | health probe (TCP listen check) fails repeatedly |
  | running | idle | shutdown completes successfully |
  | degraded | running | health recovers (probe latency < threshold) |
  | degraded | crashed | process exits |
  | degraded | offline | health probe fails while in degraded state |
  | degraded | idle | user-initiated reset |
  | crashed | offline | post-exit probe confirms unresponsiveness |
  | crashed | idle | crash cleanup completes |
  | offline | idle | user initiates reset or cleanup completes |
  | offline | launching | user initiates relaunch |

  Note: Offline slots may transition directly to launching when the user initiates a relaunch, or may first be reset to idle.
- **Smoke Probe Result**: Per-slot outcome containing phase reached, failure phase if any, response latency in milliseconds (single observed value for the probe session), resolved model ID, and provenance (binary SHA, tool version).
- **GGUF Metadata Record**: Extracted header fields plus normalized filename stem, raw path, parse timestamp, and timeout/cap settings used during extraction.
- **Hardware Allowlist Entry**: Machine fingerprint combined with PCI device IDs, stored persistently; invalidated when fingerprint or PCI set changes.
- **VRAM Risk Assessment**: Best-effort free memory from GPU telemetry, heuristic model footprint, safety margin, and a recommended action (proceed, warn, or confirm-required).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001a**: Full launch + monitor + shutdown cycle completes in under 120 seconds.
- **SC-001b**: Shutdown initiates within 1 second of user request and completes without orphan processes within 30 seconds. Verification scans for running `llama-server` processes owned by the current user. A process is an orphan if it was started by llm-runner and is still running after the shutdown timeout.
- **SC-002**: `smoke both` completes sequential probing of two slots and exits zero when both respond correctly; `smoke slot` completes for a single slot in the same manner. All probe phases (listen/accept, `/v1/models` when not skipped, chat completion) complete with HTTP 200 and valid JSON response.
- **SC-003**: GGUF metadata extraction returns required fields for a valid file within 5 seconds and under a 32 MiB read cap; invalid files produce a clear error within the same timeout.
- **SC-004**: Hardware mismatch warnings appear for non-anchor topologies and require explicit acknowledgment; once acknowledged per session or allowlisted, the warning does not block the operator.
- **SC-005**: VRAM-risk warnings surface when `free_vram × 0.85 < gguf_file_size × 1.2`, and launch or smoke does not proceed without explicit operator confirmation.
- **SC-006**: Smoke and doctor use non-overlapping exit code families, with each code documented in CLI help and mapping to a specific, actionable outcome.

## Acceptance Criteria

- **AC-010**: GGUF metadata is extracted for MVP fields; timeout and read cap are honored; fixture tests run in CI.
- **AC-011**: The TUI shows per-slot health, logs, GPU stats, `llama_cpp` backend, and build state.
- **AC-012**: All child processes receive SIGTERM. If any process does not exit within 5 seconds, SIGKILL is attempted. Post-shutdown, no child processes owned by llm-runner remain.
- **AC-013**: An isolated crash or OOM highlights the failed slot without killing the sibling.
- **AC-014**: `smoke both` is the authoritative MVP gate: sequential probes pass; it uses a distinct exit code family from `doctor`.
- **AC-016**: The VRAM heuristic warns when `free_vram × 0.85 < gguf_file_size × 1.2`; launch and smoke do not proceed without an explicit confirmation path.
- **AC-017**: `smoke both` and `smoke slot` behave per FR-002, FR-003, FR-004, FR-005, and Appendix D (delays, timeouts, `max_tokens`). Probe ordering is defined in FR-003.
- **AC-018**: Committed synthetic GGUF fixtures and a maintainer generator script are documented; CI uses static fixtures only.


## Assumptions

- The anchored workstation profile defines two slots (`arc_b580` on Intel SYCL, `rtx3090` on NVIDIA CUDA) with default binds and build flavors.

### Constraints

- **Constraint C-001**: Only `llama.cpp` is enabled at runtime. If a slot config specifies `backend: vllm`, the tool MUST reject it with a clear error: 'vLLM backend is not supported. Use `llama_cpp` or remove the backend override'.
- **Constraint C-002**: Smoke probes require the `httpx` HTTP client library. This dependency is declared in `pyproject.toml` under `[project.dependencies]`.
- **Constraint C-003**: GGUF metadata extraction requires the pinned `gguf` PyPI dependency (per PRD FR-014). This dependency is declared in `pyproject.toml` under `[project.dependencies]`. The `gguf` library is used for reading GGUF file headers only; full weight loading is not performed.
- Synthetic GGUF test fixtures are committed to the repository and generated by a documented maintainer script; CI uses static fixtures only.
- GPU telemetry is best-effort: `nvidia-smi` for NVIDIA, equivalent for Intel Arc; missing telemetry does not hard-block launch but may disable VRAM heuristics.
- Secret redaction uses a conservative pattern set (e.g., `api_key`, `token`, `password`); operators may still leak secrets in custom arguments.
- The default bind address is `127.0.0.1` with no API key unless configured otherwise.
- Per-slot lockfiles live under the resolved runtime directory (`LLM_RUNNER_RUNTIME_DIR` else `$XDG_RUNTIME_DIR/llm-runner`).
- Each smoke or diagnostic run creates a timestamped report subdirectory under `~/.local/share/llm-runner/reports/<timestamp>/`.
- `gendoc.py` is an ad-hoc release-time tool, not part of the MVP runtime.

## Appendix B — Exit Code Families

| Code | Command | Meaning |
| --- | --- | --- |
| 0 | Any | Success |
| 1 | `doctor` | Generic / unspecified error |
| 2 | `doctor` | Config validation failure |
| 3 | `doctor` | Missing dependency or binary |
| 4 | `doctor` | Hardware mismatch (non-anchor topology) |
| 5 | `doctor` | Lockfile conflict (slot already held) |
| 6 | `doctor` | Port-in-use conflict |
| 7 | `doctor` | GGUF parse failure / corrupt file |
| 8 | `doctor` | VRAM heuristic indicates impossible headroom |
| 9 | `doctor` | Reserved for future `doctor` codes |
| 10 | `smoke` | Server not ready (listen/accept timeout) |
| 11 | `smoke` | HTTP / API / network error (non-2xx response, DNS failure, SSL error, or connection reset) |
| 12 | `smoke` | Config validation failure (smoke-specific) |
| 13 | `smoke` | Model not found (wrong `model` ID) |
| 14 | `smoke` | Chat completion timeout (first-token or total) |
| 15 | `smoke` | Auth failure (API key rejected) |
| 16–18 | `smoke` | Reserved for future `smoke` codes |
| 19 | `smoke` | Slot crashed during probe (process exited unexpectedly) |
| 20+ | — | Reserved for future commands |
| 130 | `Any` | Process manager SIGKILL escalation failure (process stuck in D state) — outside doctor/smoke families |

When multiple slots are probed and one or more fail, the exit code reflects the highest-severity (lowest-numbered) failure among the slots.

## Appendix D — Default Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `smoke.inter_slot_delay_s` | 2 | Pause applied after successful completion of slot *n*, before starting slot *n+1* (not applied before the first slot) |
| `smoke.listen_timeout_s` | 120 | Timeout for smoke probing TCP ready check (matches PRD Appendix D "Listen / TCP ready timeout") |
| `smoke.http_request_timeout_s` | 10 | HTTP request timeout for `/v1/models` and chat completion endpoints |
| `tui_launch_timeout_s` | 120 | Timeout for TUI launch monitoring (process readiness wait) |
| First-token timeout | 1200 s (20 m) | Wall clock from chat request to first token |
| Total chat completion timeout | 1500 s (25 m) | Hard cap per slot smoke including generation |
| Smoke `max_tokens` | 16 | Allowed range in config 8–32 |
| GGUF metadata prefix cap | 32 MiB | Bytes read from file start for KV scan |
| GGUF parse wall timeout | 5 s | Per file metadata read path |
| `probe_latency_threshold_s` | 10 | Threshold in seconds for degraded state transition (range: 1–300) |
| `gpu_poll_interval_ms` | 1000 | GPU telemetry polling interval |
| `log_buffer_max_lines` | 500 | Maximum lines retained per slot log buffer |
| `lock_stale_threshold_s` | 300 | Lockfile staleness threshold in seconds (5 minutes) |
| `tui_refresh_interval_ms` | 200 | TUI panel refresh interval |
