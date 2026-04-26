# M4 Operational Hardening — Quickstart

## What is M4?

M4 adds operational hardening and smoke verification to llm-runner. It introduces:

- **Smoke verification** — `smoke both` / `smoke slot <id>` probes OpenAI-compatible endpoints sequentially
- **GGUF metadata extraction** — Header-only parsing with prefix cap and timeout
- **Graceful shutdown** — SIGTERM → SIGKILL escalation with orphan prevention
- **Hardware/VRAM warnings** — Non-standard hardware acknowledgment, VRAM heuristic
- **Slot state machine** — Six operational states: `idle`, `launching`, `running`, `degraded`, `crashed`, `offline`
- **Lockfiles** — Per-slot lockfiles with stale detection (5 min)
- **Extended exit codes** — Doctor (1–9), smoke (10–19), process manager (130)
- **Rotating log + redaction** — Command audit trail with secret redaction

---

## Prerequisites

- Python 3.12+
- NVIDIA drivers + `nvidia-smi` (for CUDA GPU)
- Intel Arc drivers + `sycl-ls` (for SYCL GPU)
- A built `llama.cpp` server binary (unless `LLAMA_CPP_ROOT` points at an explicit checkout):
  - SYCL: `$XDG_CACHE_HOME/llm-runner/llama.cpp/build/bin/llama-server` (or `~/.cache/llm-runner/llama.cpp/build/bin/llama-server` if XDG_CACHE_HOME is unset)
  - CUDA: `$XDG_CACHE_HOME/llm-runner/llama.cpp/build_cuda/bin/llama-server` (or `~/.cache/llm-runner/llama.cpp/build_cuda/bin/llama-server` if XDG_CACHE_HOME is unset)

---

## Installation

```bash
uv sync --extra dev
```

M4 adds two new dependencies declared in `pyproject.toml`:

| Dependency | Purpose |
| --- | --- |
| `httpx` | Smoke probe HTTP client |
| `gguf` | GGUF metadata header parser |

---

## Smoke Verification

Smoke probes verify that each slot responds to OpenAI API v1 endpoints. Probing is sequential — if one slot fails, the next is attempted, and a composite report is produced.

### Smoke all slots

```bash
uv run llm-runner smoke both
```

```
smoke: slot=arc_b580 model=Qwen3.5-2B provenance=abc1234,0.1.0
  listen    ... OK        (23ms)
  models    ... OK        (45ms)
  chat      ... OK        (312ms)  model_id=Qwen3.5-2B

smoke: slot=rtx3090 model=Qwen3.5-35B-A3B provenance=abc1234,0.1.0
  listen    ... OK        (31ms)
  models    ... OK        (52ms)
  chat      ... OK        (1204ms)  model_id=Qwen3.5-35B-A3B

──────────────────────────────────────────
  arc_b580  pass  models  complete
  rtx3090   pass  models  complete
  Overall:  PASS  exit=0
```

### Smoke a single slot

```bash
uv run llm-runner smoke slot arc_b580
```

### JSON output

```bash
uv run llm-runner smoke both --json
```

```json
{
  "results": [
    {
      "slot_id": "arc_b580",
      "status": "pass",
      "phase_reached": "complete",
      "failure_phase": null,
      "model_id": "Qwen3.5-2B",
      "latency_ms": 312,
      "provenance": {"sha": "abc1234", "version": "0.1.0"}
    }
  ],
  "overall_exit_code": 0
}
```

### Smoke flags

| Flag | Default | Description |
| --- | --- | --- |
| `--api-key <key>` | (from config/env) | Override API key for smoke probes |
| `--model-id <id>` | (from GGUF) | Override resolved model ID |
| `--max-tokens <n>` | 16 (range 8–32) | Max tokens in chat probe |
| `--prompt <text>` | "Respond with exactly one word." | Single-turn user message for chat probe |
| `--delay <s>` | 2 | Pause between slot probes |
| `--timeout <s>` | 120 (`smoke.listen_timeout_s`) | TCP ready-check timeout per slot |

### Smoke probe phases

1. **listen** — TCP connection to slot host:port (timeout: `smoke.listen_timeout_s`)
2. **models** — GET `/v1/models` (optional; skipped if `smoke.skip_models_discovery: true`)
3. **chat** — POST `/v1/chat/completions` with `temperature=0`, 16 tokens, prompt `Respond with exactly one word.`

Each phase is attempted exactly once — no retries. Failure at any phase is immediate.

---

## GGUF Metadata

Inspect model metadata without loading weights. The `doctor` command surfaces metadata for all configured models.

```bash
uv run llm-runner doctor
```

Metadata fields extracted from the GGUF header (first 32 MiB by default):

| Field | Source |
| --- | --- |
| `general.name` | GGUF key |
| `general.architecture` | GGUF key |
| `tokenizer.type` | GGUF key |
| `llama.embedding_length` | GGUF key |
| `llama.block_count` | GGUF key |
| `llama.context_length` | GGUF key |
| `llama.attention.head_count` | GGUF key |
| `llama.attention.head_count_kv` | GGUF key |

Parse timeout defaults to 5 seconds. Corrupt or unsupported files produce a clear error distinguishing parser mismatch from corruption.

---

## Hardware Warnings

On launch, llm-runner detects GPU topology and compares it against the anchored workstation profile (Intel Arc B580 + NVIDIA RTX 3090).

### TUI workflow

The TUI displays a warning panel when hardware differs from the expected profile. The layout keeps all existing panels visible while the warning replaces the slot status panel:

```
╔══════════════════════════════════════════╗
║  ⚠  NON-STANDARD HARDWARE DETECTED       ║
║  Expected: Intel Arc B580 + NVIDIA RTX   ║
║  Detected: NVIDIA RTX 3090 only          ║
║                                          ║
║  [y] Continue  [n] Abort                 ║
╚══════════════════════════════════════════╝
```

Press `y` to continue, `n` or `q` to abort. If no input within 30 seconds, the safe default (`n` / abort) is applied.

### CLI workflow

```bash
uv run llm-runner both --ack-nonstandard-hardware
```

### Hardware allowlist

Acknowledged hardware is persisted at `~/.config/llm-runner/hardware-allowlist.json`. The allowlist is invalidated when the machine fingerprint changes (computed from `lspci -d ::0300` + `sycl-ls` output, SHA256-hashed).

### VRAM heuristic

A warning triggers when:

```
free_vram × 0.85 < gguf_file_size × 1.2
```

CLI override: `--confirm-vram-risk`.

---

## Shutdown

Shutdown initiates within 1 second of user request. The escalation policy:

1. Send **SIGTERM** to each child `llama-server` process
2. Wait up to **5 seconds** for graceful exit
3. If still alive, send **SIGKILL**
4. Poll for termination; if the process remains after 2 seconds, log CRITICAL and mark slot `offline`
5. If any slot required SIGKILL escalation, exit with code **130**

After shutdown, the tool scans for orphan `llama-server` processes owned by the current user. No orphans should remain.

---

## Exit Codes

| Code | Command | Meaning |
| --- | --- | --- |
| **0** | Any | Success |
| **1** | `doctor` | Generic / unspecified error |
| **2** | `doctor` | Config validation failure |
| **3** | `doctor` | Missing dependency or binary |
| **4** | `doctor` | Hardware mismatch (non-anchor topology) |
| **5** | `doctor` | Lockfile conflict (slot already held) |
| **6** | `doctor` | Port-in-use conflict |
| **7** | `doctor` | GGUF parse failure / corrupt file |
| **8** | `doctor` | VRAM heuristic indicates impossible headroom |
| **9** | `doctor` | Reserved for future |
| **10** | `smoke` | Server not ready (listen/accept timeout) |
| **11** | `smoke` | HTTP / API / network error |
| **12** | `smoke` | Config validation failure (smoke-specific) |
| **13** | `smoke` | Model not found (wrong model ID) |
| **14** | `smoke` | Chat completion timeout |
| **15** | `smoke` | Auth failure (API key rejected) |
| **16–18** | `smoke` | Reserved for future |
| **19** | `smoke` | Slot crashed during probe |
| **130** | `Any` | Process manager SIGKILL escalation failure (process stuck in D state) — implementation-level process-manager code, outside doctor/smoke families |

When multiple slots are probed, the exit code reflects the maximum (worst) numeric exit code among all slots.

---

## Troubleshooting

### Smoke fails with exit 10 (listen timeout)

The server hasn't accepted connections within `smoke.listen_timeout_s` (default 120 s). Check that the server process is running and the port matches.

### Smoke fails with exit 13 (model not found)

The resolved model ID from GGUF metadata doesn't match the server's `/v1/models` response. Override with `--model-id <expected_id>`. After two consecutive failures for the same slot, the tool requires an explicit override.

### GGUF metadata extraction hangs

The file may be corrupted or exceed the 32 MiB prefix cap. The parser enforces a 5-second wall-clock timeout. Check with `doctor` — it distinguishes corrupt files from unsupported format versions.

### Lockfile prevents launch

A stale lockfile may exist from a prior crash. Run `doctor --locks` to see lock holders. Stale locks (PID dead or older than 300 s) are auto-removed.

### VRAM warning blocks launch

The heuristic flags `free_vram × 0.85 < gguf_file_size × 1.2`. Confirm with `--confirm-vram-risk` or reduce the model's context size.

---

## Configuration Fields Added by M4

### Config (global defaults)

| Field | Default | Description |
| --- | --- | --- |
| `smoke.inter_slot_delay_s` | 2 | Pause applied after each successful slot probe, before starting the next (not before the first slot) |
| `smoke.listen_timeout_s` | 120 | TCP ready-check timeout per slot |
| `smoke.http_request_timeout_s` | 10 | HTTP request timeout for `/v1/models` and chat |
| `smoke.max_tokens` | 16 | Max tokens in smoke chat probe (range 8–32) |
| `smoke.prompt` | `"Respond with exactly one word."` | Smoke chat prompt |
| `smoke.skip_models_discovery` | `false` | Skip `/v1/models` phase |
| `gguf_metadata_prefix_cap_bytes` | 33554432 (32 MiB) | Bytes read for GGUF header parsing |
| `gguf_metadata_parse_timeout_s` | 5 | Wall-clock timeout for GGUF parse |
| `probe_latency_threshold_s` | 10 | Degraded state transition threshold |
| `gpu_poll_interval_ms` | 1000 | GPU telemetry polling interval |
| `log_buffer_max_lines` | 500 | Max lines per slot log buffer |
| `lock_stale_threshold_s` | 300 | Lockfile staleness threshold (5 min) |
| `tui_refresh_interval_ms` | 200 | TUI panel refresh interval |
| `tui_launch_timeout_s` | 120 | TUI launch monitoring timeout |

### ServerConfig (per-slot)

| Field | Default | Description |
| --- | --- | --- |
| `api_key` | `""` | API key for the server |
| `smoke_api_key` | `""` | API key for smoke probes (overrides config/env) |

### Runtime directories

| Path | Purpose |
| --- | --- |
| `$XDG_RUNTIME_DIR/llm-runner/*.lock` | Per-slot lockfiles |
| `~/.config/llm-runner/hardware-allowlist.json` | Persistent hardware allowlist |
| `$XDG_RUNTIME_DIR/llm-runner/` (fallback `/tmp/llm-runner/`) | Session snooze file |
| `~/.local/share/llm-runner/reports/<timestamp>/` | Smoke/doctor report files |
