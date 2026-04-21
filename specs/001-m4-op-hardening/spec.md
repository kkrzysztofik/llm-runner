# Feature Specification: M4 — Operational Hardening and Smoke Verification

**Feature Branch**: `003-m4-op-hardening`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "M4 from PRD"

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

1. **Given** both slots are serving, **When** the user runs `smoke both`, **Then** slot A is probed first (listen/accept, `/v1/models` when present, minimal chat completion), then after a short pause slot B is probed in the same way, and the command exits zero only if both pass.
2. **Given** only one slot is operable, **When** the user runs `smoke slot <slot_id>`, **Then** the command probes that single slot and exits zero on success without faking a second model.
3. **Given** a chat probe is performed, **When** the server responds, **Then** the probe uses a minimal prompt, temperature 0, and a small token limit, and the resolved model ID and provenance are printed clearly.

---

### User Story 3 — Inspect Model Metadata Without Loading Weights (Priority: P2)

As a model experimenter, I want to inspect GGUF metadata (architecture, context length, attention heads, and name) quickly and safely without loading the full weight file into memory, so I can validate model compatibility and infer smoke identifiers.

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

- **FR-001**: The system MUST display per-slot status, live logs, GPU telemetry, active backend label, build job state, profile staleness warnings, and an UNSAVED indicator when a model edit has been applied but not saved.
- **FR-002**: The system MUST provide `smoke both` and `smoke slot <slot_id>` commands that probe OpenAI-compatible endpoints sequentially, with a configurable pause between slots. `smoke both` runs slot A then slot B in declaration order from config. It fails non-zero if either slot cannot be probed. The default inter-slot delay is 2 seconds (`smoke.inter_slot_delay_s`).
- **FR-003**: Smoke probes MUST progress through listen/accept, optional `/v1/models` discovery, and a minimal chat completion phase, recording which phase fails when a probe does not pass.
- **FR-004**: Smoke chat probes MUST use temperature 0 and a configurable token limit within an allowed range, with a minimal single-user prompt. The default `max_tokens` is 16 with an allowed range of 8–32.
- **FR-005**: The model ID used in smoke chat probes MUST resolve from GGUF `general.name`, falling back to normalized filename stem, then catalog override, then optional `/v1/models` match. The smoke header prints the resolved id and provenance (binary tip SHA, tool version).
- **FR-006**: If two consecutive smokes for the same slot fail with model-not-found attributable to a wrong ID, the system MUST require an explicit `smoke.model_id` override before further attempts.
- **FR-007**: The system MUST extract GGUF metadata for `general.name`, `general.architecture`, `tokenizer.type`, `llama.embedding_length`, `llama.block_count`, `llama.context_length`, `llama.attention.head_count`, and `llama.attention.head_count_kv` without loading full weights. It MUST support unsharded files or a user-indicated header shard. `doctor` MUST distinguish a corrupt file from a parser/spec mismatch (e.g. GGUF version).
- **FR-008**: GGUF metadata extraction MUST enforce a prefix read cap and a wall-clock parse timeout; on timeout or parse failure, the system MUST treat the file as unsupported/corrupt with a clear, specific error. The default prefix read cap is 32 MiB (tunable in `Config`). The default wall-clock parse timeout is 5 seconds (tunable). Unicode normalization uses NFKC. Optional strip quant suffix patterns are supported via a workstation-tuned list, off by default. `doctor` dry-run/metadata view shows the raw path, normalized stem, and resolved smoke name. The test fixtures policy requires synthetic GGUF files under `tests/fixtures/` (or equivalent), produced once by a documented maintainer script; CI consumes committed bytes only.
- **FR-009**: The system MUST normalize GGUF filenames for display and smoke ID inference using a deterministic pipeline including Unicode normalization (NFKC), whitespace replacement, and optional lowercase/quant-stripping.
- **FR-010**: The system MUST perform graceful shutdown via SIGTERM with a wait-and-escalate policy; abnormal TUI exit MUST terminate child servers to avoid orphan GPU processes.
- **FR-011**: The system MUST warn (not hard-fail) when hardware topology differs from the anchored workstation profile, requiring explicit user acknowledgment in the TUI or a CLI flag. In the TUI, an explicit **Continue** button is shown after the user reads the warning details. The CLI provides `--ack-nonstandard-hardware`.
- **FR-012**: The system MUST maintain a persistent hardware allowlist invalidated automatically when the machine fingerprint or PCI device set changes, and a session snooze file to suppress duplicate warnings during a single runtime session. The persistent allowlist lives at `~/.config/llm-runner/hardware-allowlist.json`. The session snooze file lives under `$XDG_RUNTIME_DIR/llm-runner/` (fallback `/tmp/llm-runner/`). The machine fingerprint is computed as `SHA256(lspci_gpu_output + "|" + sycl_ls_output)`, where `lspci_gpu_output` is the output of `lspci -d ::0300` (VGA/3D controllers) and `sycl_ls_output` is the output of `sycl-ls`.
- **FR-013**: The system MUST warn when free VRAM appears insufficient for a model load, requiring explicit confirmation before launch or full-model smoke proceeds. The VRAM heuristic formula is: warn if `free_vram × 0.85 < gguf_file_size × 1.2`, where `free_vram` is best-effort free memory from `nvidia-smi` (NVIDIA) or the Intel Arc equivalent, and `gguf_file_size` is the on-disk file size of the model weights. If GPU telemetry is unavailable, the check is skipped with a logged warning. The CLI provides `--confirm-vram-risk` for scripted confirmation.
- **FR-014**: The system MUST use per-slot lockfiles under a runtime directory and refuse launch if a slot lock is already held, with a diagnostic command to surface lock holders. The default bind address is `127.0.0.1` with no API key unless configured otherwise. Port-in-use refusal can be overridden with `--force-bind`, which is logged as dangerous. `doctor --locks` surfaces lock holders.
- **FR-015**: The system MUST document non-overlapping exit code families for `doctor` and `smoke`, with `smoke` using a distinct band from `doctor`. `doctor` uses exit codes 1–9; `smoke` uses exit codes 10–19. Specific codes are defined in Appendix B.
- **FR-016**: The system MUST append mutating actions to a rotating log that includes command, timestamp, exit code, and truncated output, redacting obvious secret patterns from logs and reports. When server auth is enabled, `smoke` MUST accept an API key via CLI, config, or environment without emitting the key into logs. Auth accepts `--api-key`, config `smoke.api_key`, or an environment variable (exact name documented in CLI help). Failure drops a report directory under `~/.local/share/llm-runner/reports/<timestamp>/` containing plain files.
- _(FR-017 and FR-018 consolidated into FR-011, FR-012, and FR-016 per PRD alignment.)_
- **FR-019**: The system MUST support stable `--json` output for health and status commands where applicable, including `doctor` and `smoke`, with documented exit codes.

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

- **Slot**: A GPU slot owning bind host, port, backend flavor, environment overlays, and at most one active model process. Identified by a slot ID (e.g., `arc_b580`, `rtx3090`).
- **Smoke Probe Result**: Per-slot outcome containing phase reached, failure phase if any, response latency bounds, resolved model ID, and provenance (binary SHA, tool version).
- **GGUF Metadata Record**: Extracted header fields plus normalized filename stem, raw path, parse timestamp, and timeout/cap settings used during extraction.
- **Hardware Allowlist Entry**: Machine fingerprint combined with PCI device IDs, stored persistently; invalidated when fingerprint or PCI set changes.
- **VRAM Risk Assessment**: Best-effort free memory from GPU telemetry, heuristic model footprint, safety margin, and a recommended action (proceed, warn, or confirm-required).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can launch both models, observe per-slot health and logs, and shut down without orphan processes in under 30 seconds from shutdown initiation.
- **SC-002**: `smoke both` completes sequential probing of two slots and exits zero when both respond correctly; `smoke slot` completes for a single slot in the same manner.
- **SC-003**: GGUF metadata extraction returns required fields for a valid file within 5 seconds and under a 32 MiB read cap; invalid files produce a clear error within the same timeout.
- **SC-004**: Hardware mismatch warnings appear for non-anchor topologies and require explicit acknowledgment; once acknowledged per session or allowlisted, the warning does not block the operator.
- **SC-005**: VRAM-risk warnings surface when `free_vram × 0.85 < gguf_file_size × 1.2`, and launch or smoke does not proceed without explicit operator confirmation.
- **SC-006**: Smoke and doctor use non-overlapping exit code families, with each code documented in CLI help and mapping to a specific, actionable outcome.

## Acceptance Criteria

- **AC-010**: GGUF metadata is extracted for MVP fields; timeout and read cap are honored; fixture tests run in CI.
- **AC-011**: The TUI shows per-slot health, logs, GPU stats, `llama_cpp` backend, and build state.
- **AC-012**: Graceful shutdown and abnormal exit both end without orphan servers (best-effort SIGTERM escalation).
- **AC-013**: An isolated crash or OOM highlights the failed slot without killing the sibling.
- **AC-014**: `smoke both` is the authoritative MVP gate: sequential probes pass; it uses a distinct exit code family from `doctor`.
- **AC-015**: `gendoc.py` extracts marker sections from the PRD into the README when run (release-time expectation).
- **AC-016**: The VRAM heuristic warns when `free_vram × 0.85 < gguf_file_size × 1.2`; launch and smoke do not proceed without an explicit confirmation path.
- **AC-017**: `smoke both` and `smoke slot` behave per FR-002, FR-003, FR-004, FR-005, and Appendix D (ordering, delays, timeouts, `max_tokens`).
- **AC-018**: Committed synthetic GGUF fixtures and a maintainer generator script are documented; CI uses static fixtures only.
- **AC-019**: The `setup` venv path is created and verified; no serve-time pip drift occurs.
- **AC-020**: `dry-run` for anchored models includes Qwen-class template / jinja (or successor) flags in the printed OpenAI flag bundle.

## Assumptions

- The anchored workstation profile defines two slots (`arc_b580` on Intel SYCL, `rtx3090` on NVIDIA CUDA) with default binds and build flavors.
- Only `llama.cpp` is enabled at runtime; `vLLM` may appear in config but is blocked from launching in MVP.
- Synthetic GGUF test fixtures are committed to the repository and generated by a documented maintainer script; CI uses static fixtures only.
- GPU telemetry is best-effort: `nvidia-smi` for NVIDIA, equivalent for Intel Arc; missing telemetry does not hard-block launch but may disable VRAM heuristics.
- Secret redaction uses a conservative pattern set (e.g., `api_key`, `token`, `password`); operators may still leak secrets in custom arguments.
- The default bind address is `127.0.0.1` with no API key unless configured otherwise.
- Per-slot lockfiles live under the resolved runtime directory (`LLM_RUNNER_RUNTIME_DIR` else `$XDG_RUNTIME_DIR/llm-runner`).
- Report directories are plain files under `~/.local/share/llm-runner/reports/<timestamp>/`.
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
| 11 | `smoke` | HTTP / API error (non-2xx response) |
| 12 | `smoke` | Config validation failure (smoke-specific) |
| 13 | `smoke` | Model not found (wrong `model` ID) |
| 14 | `smoke` | Chat completion timeout (first-token or total) |
| 15 | `smoke` | Auth failure (API key rejected) |
| 16–19 | `smoke` | Reserved for future `smoke` codes |
| 20+ | — | Reserved for future commands |

## Appendix D — Default Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `smoke.inter_slot_delay_s` | 2 | Pause between successful completion of slot *n* and start of slot *n+1* |
| Listen / TCP ready timeout | 120 s | Per slot; server must accept connections |
| First-token timeout | 1200 s (20 m) | Wall clock from chat request to first token |
| Total chat completion timeout | 1500 s (25 m) | Hard cap per slot smoke including generation |
| Smoke `max_tokens` | 16 | Allowed range in config 8–32 |
| GGUF metadata prefix cap | 32 MiB | Bytes read from file start for KV scan |
| GGUF parse wall timeout | 5 s | Per file metadata read path |
