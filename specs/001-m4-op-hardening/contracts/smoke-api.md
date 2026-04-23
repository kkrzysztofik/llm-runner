# Smoke Probe API Contract

**Feature**: M4 — Operational Hardening and Smoke Verification  
**Spec**: [spec.md](../spec.md) | **Plan**: [plan.md](../plan.md) | **Data Model**: [data-model.md](../data-model.md)  
**Branch**: `003-m4-op-hardening`

---

## 1. API Overview

### Purpose

The smoke probe API provides a programmatic verification that each llama.cpp inference server is responding correctly to OpenAI-compatible endpoints. It is the **MVP completion gate** — `smoke both` is the authoritative pass/fail signal that launch produced working endpoints.

### Who Calls It

| Caller | Context | Mode |
| --- | --- | --- |
| `llm-runner smoke both` | CLI — full verification of all slots | Human-readable or JSON |
| `llm-runner smoke slot <id>` | CLI — single-slot verification | Human-readable or JSON |
| TUI monitor (future) | TUI — periodic health check | Internal state update |
| CI scripts | Automated — pre-flight or post-deploy check | JSON (`--json`) |

### Design Principles

- **Sequential probing**: Slots are probed one at a time in declaration order from config. Failure at one slot does NOT stop probing remaining slots.
- **No retries**: Each phase is attempted exactly once. Failure is immediate and non-recoverable.
- **Pure library**: `llama_manager/smoke.py` owns probe logic — no argparse, no Rich, no subprocess at module level.
- **httpx-based**: All HTTP requests use the `httpx` client library (declared in `pyproject.toml`).
- **Composite reporting**: `smoke both` produces a composite report with per-slot results and an overall exit code.

---

## 2. Probe Phases

Each smoke probe for a single slot progresses through three phases. Each phase is attempted exactly once; failure at any phase is terminal for that probe session.

### Phase 1: Listen / Accept

**Goal**: Verify the server process is accepting TCP connections on the configured host:port.

**Method**: Non-blocking socket connect with timeout.

**Parameters**:
| Parameter | Config Key | Default | Description |
| --- | --- | --- | --- |
| `listen_timeout_s` | `smoke_listen_timeout_s` | 30 | Maximum seconds to wait for TCP connection |

**Flow**:
1. Create a `socket.socket(socket.AF_INET, socket.SOCK_STREAM)`
2. Set `socket.settimeout(smoke_cfg.listen_timeout_s)`
3. Attempt `sock.connect((host, port))`
4. On success: proceed to Phase 2
5. On timeout: record `status="timeout"`, `phase_reached="listen"`, `failure_phase="listen"`, exit code 10
6. On connection refused / network error: record `status="fail"`, `phase_reached="listen"`, `failure_phase="listen"`, exit code 10

**Exit codes**:
| Code | Condition |
| --- | --- |
| 0 | Connection accepted — proceed to Phase 2 |
| 10 | Timeout or network error — probe fails |

**Request/Response**: N/A (TCP-level only, no HTTP).

**Edge cases**:
- If `smoke_listen_timeout_s` is less than 1, raise `ValueError` (validated in `__post_init__`).
- If the slot's process crashed between launch and probe, `connect()` returns `ECONNREFUSED` → exit 10.

---

### Phase 2: Models Discovery (`GET /v1/models`)

**Goal**: Verify the server responds to the OpenAI `/v1/models` endpoint and returns a non-empty model list matching the expected model ID.

**Method**: `httpx.get(url, timeout=http_request_timeout_s)`

**Parameters**:
| Parameter | Config Key | Default | Description |
| --- | --- | --- | --- |
| `http_request_timeout_s` | `smoke_http_request_timeout_s` | 10 | HTTP request timeout |
| `skip_models_discovery` | `smoke.skip_models_discovery` | `False` | Skip this phase entirely |

**Flow**:
1. If `skip_models_discovery=True`, proceed directly to Phase 3 (chat).
2. Construct URL: `http://{bind_address}:{port}/v1/models`
3. Send `GET /v1/models` with no authorization header (or with API key if configured)
4. Parse response JSON

**Response handling**:

| Response | Action |
| --- | --- |
| HTTP 200, `models` array non-empty | Compare first model's `id` with expected model ID. Match → proceed to Phase 3. Mismatch → exit 13 (`model_not_found`). |
| HTTP 200, `models` array empty | Exit 13 (`model_not_found`). |
| HTTP 404 | Phase **skipped** — endpoint not supported. Proceed to Phase 3. |
| HTTP 401 / 403 | Exit 15 (`auth_failure`). |
| HTTP 5xx | Exit 11 (`fail` — HTTP/API error). |
| Network error (DNS, SSL, connection reset) | Exit 11 (`fail`). |

**Model ID resolution chain** (FR-005):
1. `--model-id` CLI flag (highest precedence)
2. GGUF `general.name` from metadata extraction
3. First model's `id` from `/v1/models` response
4. Fallback: normalized filename stem prefixed with `path:`

**Exit codes**:
| Code | Condition |
| --- | --- |
| 0 | Model found and ID matches — proceed to Phase 3 |
| 11 | HTTP error or network error |
| 13 | No models returned or model ID mismatch |
| 15 | Auth failure (401/403) |

---

### Phase 3: Chat Completion (`POST /v1/chat/completions`)

**Goal**: Verify the server can produce a completion via the OpenAI chat completions API.

**Method**: `httpx.post(url, json=payload, timeout=http_request_timeout_s)`

**Parameters**:
| Parameter | Config Key | Default | Range |
| --- | --- | --- | --- |
| `max_tokens` | `smoke.max_tokens` | 16 | 8–32 |
| `temperature` | — | 0 | Fixed |
| `prompt` | `smoke.prompt` | `"Respond with exactly one word."` | Non-empty |
| `http_request_timeout_s` | `smoke_http_request_timeout_s` | 10 | ≥ 1 |

**Request payload** (minimal, no system prompt):

```json
{
  "model": "<resolved_model_id>",
  "messages": [
    {
      "role": "user",
      "content": "Respond with exactly one word."
    }
  ],
  "max_tokens": 16,
  "temperature": 0,
  "stream": false
}
```

**Response handling**:

| Response | Action |
| --- | --- |
| HTTP 200, valid JSON with `choices[0].message` | Probe **passes** — record `status="pass"`, `phase_reached="complete"`, exit 0 |
| HTTP 200, but `choices` is empty | Exit 11 (`fail`) |
| HTTP 401 / 403 | Exit 15 (`auth_failure`) |
| HTTP 4xx (other) | Exit 11 (`fail`) |
| HTTP 5xx | Exit 11 (`fail`) |
| Timeout | Exit 14 (`timeout`) |
| Network error | Exit 11 (`fail`) |

**Timeout strategy**:
- The `httpx` request timeout (`smoke_http_request_timeout_s`, default 10s) applies to the full response.
- A separate first-token timeout (1200s / 20 min) and total chat timeout (1500s / 25 min) are configurable but default to very generous values to handle large models on constrained hardware.
- If streaming is enabled, first-token timeout applies; for non-streaming (default), the full-request timeout applies.

**Exit codes**:
| Code | Condition |
| --- | --- |
| 0 | Chat completion succeeded — probe passes |
| 11 | HTTP error or network error |
| 14 | Timeout (first-token or total) |
| 15 | Auth failure |

---

### Phase 3b: Slot Crash Detection (interleaved)

**Goal**: If the server process exits unexpectedly during Phase 2 or Phase 3, detect this and report a crash.

**Method**: Check process liveness via `psutil.pid_exists()` or by examining the process return code.

**Flow**:
1. After each phase, check if the server process is still alive.
2. If the process has exited with a non-zero code: record `status="crashed"`, `phase_reached` = last completed phase, `failure_phase` = next phase, exit code 19.
3. If the process is still alive, proceed to the next phase.

**Exit codes**:
| Code | Condition |
| --- | --- |
| 19 | Process exited unexpectedly during probe |

---

## 3. Error Classification

### httpx Exception Mapping

| Exception | Exit Code | SmokeProbeStatus | Failure Phase |
| --- | --- | --- | --- |
| `httpx.ConnectTimeout` | 10 | `TIMEOUT` | `LISTEN` or `MODELS` or `CHAT` |
| `httpx.ReadTimeout` | 14 | `TIMEOUT` | `CHAT` |
| `httpx.ConnectError` | 10 | `TIMEOUT` | `LISTEN` |
| `httpx.NetworkError` | 11 | `FAIL` | `MODELS` or `CHAT` |
| `httpx.RemoteProtocolError` | 11 | `FAIL` | `MODELS` or `CHAT` |
| `httpx.PoolTimeout` | 14 | `TIMEOUT` | `CHAT` |
| `httpx.ReadError` | 11 | `FAIL` | `MODELS` or `CHAT` |

### HTTP Status Code Mapping

| Status Code | Exit Code | SmokeProbeStatus |
| --- | --- | --- |
| 200 with valid response | 0 | `PASS` |
| 200 with empty `models` array | 13 | `MODEL_NOT_FOUND` |
| 200 with model ID mismatch | 13 | `MODEL_NOT_FOUND` |
| 401 / 403 | 15 | `AUTH_FAILURE` |
| 404 (only for `/v1/models`) | — | Skipped (proceed to chat) |
| 4xx (other) | 11 | `FAIL` |
| 5xx | 11 | `FAIL` |

### Socket-Level Errors

| Error | Exit Code | SmokeProbeStatus |
| --- | --- | --- |
| `socket.timeout` | 10 | `TIMEOUT` |
| `ConnectionRefusedError` | 10 | `TIMEOUT` |
| `ConnectionResetError` | 11 | `FAIL` |
| `OSError` (other network) | 10 | `TIMEOUT` |

---

## 4. JSON Output Schema

### 4.1 Smoke Probe Result (Per Slot)

```json
{
  "slot_id": "string",
  "status": "pass" | "fail" | "timeout" | "crashed" | "model_not_found" | "auth_failure",
  "phase_reached": "listen" | "models" | "chat" | "complete",
  "failure_phase": "listen" | "models" | "chat" | null,
  "model_id": "string" | null,
  "latency_ms": "integer" | null,
  "provenance": {
    "sha": "string",
    "version": "string"
  }
}
```

**Field descriptions**:
- `slot_id`: Normalized slot identifier (e.g., `"arc_b580"`, `"rtx3090"`)
- `status`: Outcome of the probe. `pass` means all phases completed successfully.
- `phase_reached`: The last phase that was attempted. `complete` means all phases passed.
- `failure_phase`: The phase where failure occurred. `null` when status is `pass`.
- `model_id`: Resolved model ID from GGUF metadata or `/v1/models`. `null` if unavailable.
- `latency_ms`: Single observed latency in milliseconds for the entire probe session. `null` if not measured (e.g., on crash or listen failure).
- `provenance.sha`: Binary tip SHA (40-char hex) or `"unknown"`.
- `provenance.version`: Package version from `importlib.metadata.version('llm_runner')` or `"dev"`.

### 4.2 Smoke Composite Report (smoke both)

```json
{
  "slots": [
    {
      "slot_id": "string",
      "status": "pass" | "fail" | "timeout" | "crashed" | "model_not_found" | "auth_failure",
      "phase_reached": "listen" | "models" | "chat" | "complete",
      "failure_phase": "listen" | "models" | "chat" | null,
      "model_id": "string" | null,
      "latency_ms": "integer" | null,
      "provenance": {
        "sha": "string",
        "version": "string"
      }
    }
  ],
  "overall_exit_code": "integer"
}
```

**Field descriptions**:
- `slots`: Per-slot results in declaration order from config.
- `overall_exit_code`: Highest-severity (lowest-numbered) failure code among all slots. 0 if all pass.

### 4.3 JSON Schema (Draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SmokeCompositeReport",
  "type": "object",
  "required": ["slots", "overall_exit_code"],
  "properties": {
    "slots": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["slot_id", "status", "phase_reached", "provenance"],
        "properties": {
          "slot_id": { "type": "string" },
          "status": {
            "type": "string",
            "enum": ["pass", "fail", "timeout", "crashed", "model_not_found", "auth_failure"]
          },
          "phase_reached": {
            "type": "string",
            "enum": ["listen", "models", "chat", "complete"]
          },
          "failure_phase": {
            "oneOf": [
              { "type": "null" },
              { "type": "string", "enum": ["listen", "models", "chat"] }
            ]
          },
          "model_id": {
            "oneOf": [
              { "type": "string" },
              { "type": "null" }
            ]
          },
          "latency_ms": {
            "oneOf": [
              { "type": "integer" },
              { "type": "null" }
            ]
          },
          "provenance": {
            "type": "object",
            "required": ["sha", "version"],
            "properties": {
              "sha": { "type": "string" },
              "version": { "type": "string" }
            }
          }
        }
      }
    },
    "overall_exit_code": {
      "type": "integer",
      "minimum": 0,
      "maximum": 19
    }
  }
}
```

---

## 5. CLI Contract

### 5.1 Command Structure

```
llm-runner smoke <subcommand> [flags]

Subcommands:
  both              Probe all configured slots sequentially
  slot <slot_id>    Probe a single slot by ID
```

**Argument handling** (pseudo-argparse, following the pattern in `cli_parser.py`):

```python
# Pseudo-code: handled in cli_parser.py before normal mode parsing
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

### 5.2 `smoke both` Arguments

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--json` | flag | `false` | Output structured JSON instead of human-readable text |
| `--api-key <key>` | string | (from config/env) | Override API key for smoke probes |
| `--model-id <id>` | string | (from GGUF metadata) | Override model ID for smoke chat probes |
| `--max-tokens <n>` | int | 16 (`smoke.max_tokens`) | Max tokens in smoke chat probe (range 8–32) |
| `--delay <seconds>` | int | 2 (`smoke_inter_slot_delay_s`) | Pause between slot probes |
| `--timeout <seconds>` | int | 30 (`smoke_listen_timeout_s`) | TCP ready-check timeout per slot |

### 5.3 `smoke slot <slot_id>` Arguments

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--json` | flag | `false` | Output structured JSON instead of human-readable text |
| `--api-key <key>` | string | (from config/env) | Override API key for smoke probes |
| `--model-id <id>` | string | (from GGUF metadata) | Override model ID for smoke chat probes |
| `--max-tokens <n>` | int | 16 (`smoke.max_tokens`) | Max tokens in smoke chat probe (range 8–32) |
| `--timeout <seconds>` | int | 30 (`smoke_listen_timeout_s`) | TCP ready-check timeout |

### 5.4 Exit Codes

Smoke exit codes (10–19) are distinct from doctor exit codes (1–9):

| Code | Meaning |
| --- | --- |
| 0 | All probed slots passed |
| 10 | Server not ready (listen/accept timeout) |
| 11 | HTTP / API / network error |
| 12 | Config validation failure (smoke-specific) |
| 13 | Model not found (wrong model ID) |
| 14 | Chat completion timeout |
| 15 | Auth failure |
| 19 | Slot crashed during probe |
| 16–18 | Reserved for future smoke codes |

**Composite exit code rule**: When multiple slots are probed (`smoke both`), the exit code reflects the **highest-severity (lowest-numbered) failure** among all slots. If all slots pass, exit 0.

### 5.5 Output Formats

**Human-readable (default)**:
```
smoke: slot=arc_b580 model=qwen3.5-2b provenance=abc123,dev
  [PASS] listen=12ms  models=45ms  chat=230ms  model_id=qwen3.5-2b

smoke: slot=rtx3090 model=qwen3.5-35b provenance=abc123,dev
  [PASS] listen=8ms   models=32ms  chat=145ms  model_id=qwen3.5-35b

--- Smoke Report ---
  arc_b580: PASS (287ms)
  rtx3090:  PASS (185ms)
  Overall:  PASS (exit 0)
```

**JSON (`--json`)**: Single JSON document matching the FR-020 schema.

---

## 6. Configuration Contract

### 6.1 SmokeProbeConfiguration

```python
@dataclass
class SmokeProbeConfiguration:
    """Runtime settings for smoke probes.

    Attributes:
        inter_slot_delay_s: Pause between slot probes.
        listen_timeout_s: TCP ready-check timeout per slot.
        http_request_timeout_s: HTTP request timeout.
        max_tokens: Token limit for chat probe (8-32).
        prompt: Single-turn user message text.
        skip_models_discovery: Skip /v1/models phase.
        api_key: API key from CLI flag, config, or env.
        model_id_override: User-provided model ID override.
    """

    inter_slot_delay_s: int = 2
    listen_timeout_s: int = 30
    http_request_timeout_s: int = 10
    max_tokens: int = 16
    prompt: str = "Respond with exactly one word."
    skip_models_discovery: bool = False
    api_key: str = ""
    model_id_override: str | None = None

    def __post_init__(self) -> None:
        if not (8 <= self.max_tokens <= 32):
            raise ValueError("max_tokens must be between 8 and 32")
```

### 6.2 API Key Precedence

Highest to lowest:
1. `--api-key` CLI flag
2. `smoke.api_key` config field
3. `LLM_RUNNER_SMOKE_API_KEY` environment variable

**Implementation**: The CLI layer (`llama_cli/smoke_cli.py`) resolves the API key using this precedence chain and passes the resolved value to `llama_manager/smoke.py`. The library never reads environment variables directly — it receives the resolved key as a parameter.

### 6.3 Config Field Mapping

| Config Field | Default | Source |
| --- | --- | --- |
| `smoke_listen_timeout_s` | 30 | `Config` dataclass |
| `smoke_http_request_timeout_s` | 10 | `Config` dataclass |
| `smoke_inter_slot_delay_s` | 2 | `Config` dataclass |
| `smoke.max_tokens` | 16 | `Config` dataclass |
| `smoke.prompt` | `"Respond with exactly one word."` | `Config` dataclass |
| `smoke.skip_models_discovery` | `False` | `Config` dataclass |
| `smoke.api_key` | `""` (empty = use env) | `Config` dataclass |

---

## 7. Module Boundaries

### 7.1 Responsibility Matrix

| Responsibility | Module | Rationale |
| --- | --- | --- |
| Smoke probe orchestration (phase sequencing, composite reporting) | `llama_manager/smoke.py` | Pure library — no I/O, no argparse |
| GGUF metadata extraction (model ID resolution) | `llama_manager/metadata.py` | Pure library — reads file headers only |
| Provenance resolution (SHA, version) | `llama_manager/smoke.py` | Pure library — calls `subprocess` only within a function, not at module level |
| CLI argument parsing for `smoke` command | `llama_cli/cli_parser.py` | I/O layer — owns argparse |
| Smoke CLI orchestrator (entry point, JSON output, human-readable output) | `llama_cli/smoke_cli.py` | I/O layer — owns stdout/stderr |
| TUI smoke integration (future) | `llama_cli/tui_app.py` | I/O layer — owns Rich Live rendering |
| Config dataclasses (SmokeProbeConfiguration) | `llama_manager/config.py` | Pure library — data definitions |
| Smoke config factory functions | `llama_manager/config_builder.py` | Pure library — factory pattern |

### 7.2 Dependency Graph

```text
llama_cli/smoke_cli.py
  ├── llama_manager/smoke.py          # probe orchestration
  │     ├── llama_manager/metadata.py  # GGUF metadata extraction
  │     ├── llama_manager/config.py    # SmokeProbeConfiguration, enums
  │     └── llama_manager/config_builder.py  # Smoke config factories
  ├── llama_manager/config.py          # Config (smoke fields)
  └── llama_cli/cli_parser.py          # parse_args() (smoke subcommands)

llama_manager/smoke.py
  └── httpx (external dependency)
llama_manager/metadata.py
  └── gguf (external dependency)
```

### 7.3 One-Way Dependencies

```
llama_cli/ ──► llama_manager/   (I/O layer depends on pure library)
tests/     ──► llama_manager/   (unit tests depend on pure library)
llama_cli/ ──► llama_cli/       (internal imports only)
```

**Critical rule**: `llama_manager/smoke.py` MUST NOT import from `llama_cli`. The dependency is one-way.

### 7.4 Interface Contracts

#### `llama_manager/smoke.py` Public API

```python
def probe_slot(
    host: str,
    port: int,
    smoke_cfg: SmokeProbeConfiguration,
    model_id: str | None = None,
    expected_model_id: str | None = None,
) -> SmokeProbeResult:
    """Run a complete smoke probe against a single slot.

    Args:
        host: Bind host of the server.
        port: Port of the server.
        smoke_cfg: Smoke probe configuration.
        model_id: Model ID to use in chat completion request.
        expected_model_id: Expected model ID for /v1/models comparison.

    Returns:
        SmokeProbeResult with probe outcome.
    """

def resolve_provenance() -> ProvenanceRecord:
    """Resolve build provenance (SHA + version).

    Returns:
        ProvenanceRecord with sha and version.
    """

def compute_overall_exit_code(results: list[SmokeProbeResult]) -> int:
    """Compute the highest-severity (lowest-numbered) exit code.

    Args:
        results: List of SmokeProbeResult objects.

    Returns:
        Overall exit code (0 if all pass).
    """
```

#### `llama_manager/metadata.py` Public API

```python
def extract_gguf_metadata(
    model_path: str,
    prefix_cap_bytes: int = 32 * 1024 * 1024,
    parse_timeout_s: float = 5.0,
) -> GGUFMetadataRecord:
    """Extract GGUF header metadata without loading weights.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap_bytes: Maximum bytes to read from file start.
        parse_timeout_s: Wall-clock timeout for extraction.

    Returns:
        GGUFMetadataRecord with extracted fields.

    Raises:
        GgufParseError: On parse failure (corrupt, timeout, unsupported version).
    """
```

#### `llama_cli/smoke_cli.py` Public API

```python
def main() -> int:
    """Entry point for 'llm-runner smoke' command.

    Returns:
        Exit code (0-19).
    """

def run_smoke(args: argparse.Namespace) -> int:
    """Run smoke probes based on parsed CLI arguments.

    Args:
        args: Parsed arguments from cli_parser.

    Returns:
        Exit code (0-19).
    """

def format_human_readable(results: list[SmokeProbeResult], overall_code: int) -> str:
    """Format smoke results as human-readable text.

    Args:
        results: Per-slot probe results.
        overall_code: Overall exit code.

    Returns:
        Formatted string for stdout.
    """
```

---

## 8. Test Contract

### 8.1 Test Organization

```
src/tests/
├── test_smoke.py          # Smoke probe logic tests (mocked HTTP)
├── test_smoke_cli.py      # CLI parsing and output format tests
├── test_metadata.py       # GGUF metadata extraction tests
└── fixtures/
    ├── gguf_v3_valid.bin      # Valid GGUF v3 with all required keys
    ├── gguf_v3_no_name.bin    # Valid GGUF v3 missing general.name
    ├── gguf_corrupt.bin       # Corrupt file (bad magic bytes)
    ├── gguf_truncated.bin     # Truncated file (valid header, no KV data)
    └── gguf_v4_unsupported.bin # Valid GGUF v4 (expected error)
```

### 8.2 What Must Be Mocked

| Dependency | Mock Strategy | Rationale |
| --- | --- | --- |
| `httpx.Client` | `unittest.mock.patch` or `pytest-mock` | No real HTTP calls in CI |
| `httpx.get()` / `httpx.post()` | Return `httpx.Response` objects with controlled status codes | Test each HTTP status code path |
| `socket.socket` | `unittest.mock.patch` | No real TCP connections in CI |
| `httpx.ConnectTimeout` | Raise directly | Test timeout handling |
| `httpx.ReadTimeout` | Raise directly | Test read timeout handling |
| `gguf.open()` or file reads | `unittest.mock.patch` with fixture bytes | Test GGUF parsing without real files |
| `subprocess.run` (for git SHA) | `unittest.mock.patch` | Test provenance resolution |
| `importlib.metadata.version` | `unittest.mock.patch` | Test version fallback to `"dev"` |

### 8.3 Test Fixtures

#### Smoke Fixtures (HTTP Response Mocks)

```python
# test_smoke.py — fixture examples

# Successful /v1/models response
MOCK_MODELS_RESPONSE = httpx.Response(
    200,
    json={
        "data": [
            {"id": "qwen3.5-2b", "object": "model", "owned_by": "organization"}
        ]
    },
)

# Successful chat completion response
MOCK_CHAT_RESPONSE = httpx.Response(
    200,
    json={
        "id": "cmpl-abc123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3.5-2b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Yes."},
                "finish_reason": "stop",
            }
        ],
    },
)

# Auth failure response
MOCK_AUTH_FAILURE = httpx.Response(401, json={"error": {"message": "Invalid API key"}})

# Empty models response
MOCK_EMPTY_MODELS = httpx.Response(200, json={"data": []})

# Model ID mismatch response
MOCK_MODEL_MISMATCH = httpx.Response(
    200,
    json={
        "data": [
            {"id": "wrong-model-id", "object": "model", "owned_by": "organization"}
        ]
    },
)
```

#### GGUF Fixtures

Generated by `scripts/generate_gguf_fixtures.py`. Committed as binary files under `tests/fixtures/`. CI consumes static fixture bytes only.

### 8.4 Test Categories

#### Unit Tests (no subprocess, no GPU, no network)

| Test | What It Verifies |
| --- | --- |
| `test_probe_slot_listens_success` | Phase 1 succeeds, proceeds to Phase 2 |
| `test_probe_slot_listen_timeout` | Phase 1 timeout → exit 10 |
| `test_probe_slot_listen_connection_refused` | Phase 1 ECONNREFUSED → exit 10 |
| `test_probe_slot_models_success` | Phase 2 returns matching model ID |
| `test_probe_slot_models_empty` | Phase 2 empty models → exit 13 |
| `test_probe_slot_models_mismatch` | Phase 2 model ID mismatch → exit 13 |
| `test_probe_slot_models_404` | Phase 2 404 → skip to Phase 3 |
| `test_probe_slot_models_auth_failure` | Phase 2 401/403 → exit 15 |
| `test_probe_slot_models_network_error` | Phase 2 network error → exit 11 |
| `test_probe_slot_chat_success` | Phase 3 returns valid completion |
| `test_probe_slot_chat_timeout` | Phase 3 timeout → exit 14 |
| `test_probe_slot_chat_auth_failure` | Phase 3 401/403 → exit 15 |
| `test_probe_slot_chat_network_error` | Phase 3 network error → exit 11 |
| `test_probe_slot_process_crashed` | Process exits during probe → exit 19 |
| `test_probe_slot_skip_models_discovery` | `skip_models_discovery=True` skips Phase 2 |
| `test_probe_slot_api_key_header` | API key sent in `Authorization` header |
| `test_resolve_provenance_sha_from_git` | SHA resolved from git HEAD |
| `test_resolve_provenance_sha_unknown` | SHA falls back to `"unknown"` |
| `test_resolve_provenance_version_from_package` | Version from `importlib.metadata` |
| `test_resolve_provenance_version_dev` | Version falls back to `"dev"` |
| `test_compute_overall_exit_code_all_pass` | All pass → exit 0 |
| `test_compute_overall_exit_code_mixed` | Mixed results → lowest failure code |
| `test_smoke_config_validation_max_tokens` | `max_tokens` outside 8-32 raises `ValueError` |
| `test_smoke_config_validation_listen_timeout` | `listen_timeout_s` < 1 raises `ValueError` |
| `test_api_key_precedence_cli_override` | CLI flag wins over config and env |
| `test_api_key_precedence_config` | Config wins over env |
| `test_api_key_precedence_env` | Env is lowest |

#### CLI Tests (no subprocess, no GPU)

| Test | What It Verifies |
| --- | --- |
| `test_parse_smoke_both_args` | `smoke both` parses correctly |
| `test_parse_smoke_slot_args` | `smoke slot arc_b580` parses correctly |
| `test_parse_smoke_both_with_flags` | `smoke both --json --max-tokens 8` parses correctly |
| `test_parse_smoke_missing_subcommand` | Missing subcommand → `SystemExit(1)` |
| `test_parse_smoke_invalid_subcommand` | Invalid subcommand → `SystemExit(1)` |
| `test_format_human_readable_single_pass` | Single slot pass formatted correctly |
| `test_format_human_readable_composite` | Multiple slots with mixed results formatted correctly |
| `test_format_json_single_pass` | Single slot JSON output matches schema |
| `test_format_json_composite` | Composite JSON output matches schema |

#### GGUF Metadata Tests

| Test | What It Verifies |
| --- | --- |
| `test_extract_metadata_valid_gguf_v3` | Valid GGUF v3 extracts all fields |
| `test_extract_metadata_missing_general_name` | Missing `general.name` → `None` |
| `test_extract_metadata_corrupt_file` | Bad magic bytes → `GgufParseError.CORRUPT_FILE` |
| `test_extract_metadata_truncated` | Valid header, no KV data → `GgufParseError.PARSE_TIMEOUT` or `READ_ERROR` |
| `test_extract_metadata_gguf_v4` | GGUF v4 → `GgufParseError.UNSUPPORTED_VERSION` |
| `test_extract_metadata_timeout` | Timeout → `GgufParseError.PARSE_TIMEOUT` |
| `test_normalize_filename_nfkc` | Unicode NFKC normalization applied |
| `test_normalize_filename_whitespace` | Whitespace replaced |

### 8.5 CI Quality Gates

All tests must pass these CI gates:

1. **lint** — `uv run ruff check .`
2. **format** — `uv run ruff format --check .`
3. **typecheck** — `uv run pyright`
4. **test** — `uv run pytest`
5. **coverage** — `uv run pytest --cov --cov-report=term-missing`

**Additional gates**:
- `uv run pip-audit` — check for known CVEs in dependencies (including new `httpx` and `gguf` deps)

### 8.6 Test Isolation Requirements

- **No subprocess spawning**: Tests MUST NOT spawn real `llama-server` processes.
- **No GPU hardware**: Tests MUST NOT require GPU hardware or drivers.
- **No filesystem side effects**: Tests MUST use `tmp_path` fixture for any file operations.
- **No network calls**: All HTTP calls MUST be mocked.
- **No environment mutation**: Tests MUST NOT modify `os.environ` without restoration.

---

## Appendix A: Consecutive Failure Counter (FR-006)

### In-Memory Tracking

```python
@dataclass
class ConsecutiveFailureCounter:
    """In-memory tracking of consecutive model-not-found failures."""

    slot_id: str
    count: int = 0
    model_id_override: str | None = None

    def reset(self) -> None:
        """Reset the counter to zero."""
        self.count = 0
        self.model_id_override = None
```

### Rules

1. Counter is **in-memory only** (not persisted). Resets on process restart.
2. Counter increments **only** when the smoke probe exits with code **13** (model not found).
3. Other exit codes (timeout, auth failure, connection error) do **not** increment the counter.
4. After **2 consecutive failures** with wrong model ID, the system requires an explicit `--model-id` override.
5. If the user provides a `--model-id` override that **fails to match** on the next attempt, the counter is **NOT reset**. The user must provide a correct override.
6. If the override **succeeds**, the counter resets.

### Test Implications

| Scenario | Counter After |
| --- | --- |
| Exit 13 (model not found) | +1 |
| Exit 13 again (model not found) | +1 (now 2) → require override |
| Exit 10 (timeout) | No change |
| Exit 15 (auth failure) | No change |
| Exit 0 (pass) | Reset to 0 |
| Process restart | Reset to 0 |

---

## Appendix B: Provenance Resolution

### SHA Resolution

```python
def resolve_git_sha(llama_cpp_root: str) -> str:
    """Resolve binary tip SHA from llama.cpp git repo.

    Resolution chain:
    1. Resolve Path(llama_cpp_root) / '.git' / 'HEAD'
    2. If path exists, run `git rev-parse HEAD` with 5-second timeout
    3. On any failure (missing git, timeout, non-zero exit), return 'unknown'

    Args:
        llama_cpp_root: Path to llama.cpp source tree.

    Returns:
        40-char hex SHA or 'unknown'.
    """
```

### Version Resolution

```python
def resolve_version() -> str:
    """Resolve package version.

    Resolution chain:
    1. Try `importlib.metadata.version('llm_runner')`
    2. On any failure (DistributionNotFound, PackageNotFoundError), return 'dev'

    Returns:
        Package version string or 'dev'.
    """
```

---

## Appendix C: Inter-Slot Delay

When running `smoke both`, after successfully completing a slot probe (either pass or fail), the probe pauses for `smoke_inter_slot_delay_s` (default 2 seconds) before starting the next slot. This prevents rapid-fire probing from overwhelming the servers.

```python
import time

for i, slot in enumerate(slots):
    result = probe_slot(slot.host, slot.port, smoke_cfg, ...)
    results.append(result)
    if i < len(slots) - 1:  # Don't delay after the last slot
        time.sleep(smoke_cfg.inter_slot_delay_s)
```

---

## Appendix D: Dry-Run Integration

The `dry-run` mode must show smoke-relevant flag bundles for each slot. This is handled in `llama_cli/dry_run.py`, which calls `build_dry_run_slot_payload()` from `llama_manager/server.py`. The payload includes an `openai_flag_bundle` that indicates OpenAI API compatibility flags.

When `smoke` is integrated, the dry-run output should also show:
- Whether `/v1/models` will be probed
- The chat completion prompt that will be sent
- The expected model ID (from GGUF metadata)
- The API key source (CLI flag, config, or env)

This provides operators with visibility into what the smoke probe will do before executing it.
