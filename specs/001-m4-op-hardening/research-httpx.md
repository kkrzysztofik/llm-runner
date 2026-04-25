# Research: httpx Best Practices for Smoke Probes

**Date**: 2026-04-23
**Feature**: M4 — Operational Hardening and Smoke Verification
**Status**: Implemented

---

## Decision 1: Per-Request Timeout via `httpx.Timeout` Object

### Research Finding
The M4 spec requires two distinct timeout values (`smoke_listen_timeout_s` for TCP ready-check, `smoke_http_request_timeout_s` for HTTP requests). An `httpx.Timeout` object with separate `connect` and `read` parameters cleanly maps to this two-phase model.

### Proposed Approach
Use a fine-grained `httpx.Timeout` object with separate `connect` and `read` timeouts for each smoke probe request. Set `connect` to the listen timeout (120s default) and `read` to the HTTP request timeout (10s default).

```python
import httpx

# For smoke probes: connect=listen_timeout, read=http_request_timeout
timeout = httpx.Timeout(
    connect=smoke_listen_timeout_s,   # 120s default (per final spec)
    read=smoke_http_request_timeout_s,  # 10s default
    write=5.0,
    pool=5.0,
)
```

### Rationale
- The M4 spec requires two distinct timeout values: `smoke_listen_timeout_s` (120s, for TCP ready-check) and `smoke_http_request_timeout_s` (10s, for HTTP requests). A single scalar timeout cannot express this distinction.
- `httpx.Timeout(connect=..., read=..., write=..., pool=...)` with four distinct per-operation values cleanly maps to the spec's timeout model.
- Per-request timeout (passed to `client.get()`, `client.post()`, etc.) is preferred over client-level in this case because each smoke probe targets a different port and may have different timeout requirements.
- The spec explicitly states: "Each phase attempted exactly once — no retries." Per-request timeouts align with this no-retry semantics.

### Note on Implementation Status
**Implemented**: The per-request `httpx.Timeout` pattern is implemented in `llama_manager/smoke.py`. Both `_probe_models()` (lines 423-428) and `_probe_chat()` (lines 593-598) create `httpx.Timeout` objects with separate `connect`, `read`, `write`, and `pool` parameters passed to `httpx.Client(timeout=timeout)`.

### Alternatives Considered
- **Client-level timeout**: Would require creating a new `httpx.Client` per probe to apply different timeouts. This adds unnecessary object lifecycle management. Per-request timeout is simpler and equally correct.
- **`httpx.Timeout(None)` (infinite)**: Rejected — the spec mandates bounded timeouts at all times.

---

## Decision 2: Exception Hierarchy for Error Classification

### Decision
Use the httpx exception hierarchy to classify errors into exit codes per Appendix B:

```python
import httpx
from enum import IntEnum

class SmokeExitCode(IntEnum):
    SERVER_NOT_READY = 10       # listen/accept timeout
    HTTP_API_ERROR = 11         # network error during HTTP phase
    CONFIG_ERROR = 12           # smoke config validation failure
    MODEL_NOT_FOUND = 13        # wrong model ID or empty models array
    CHAT_TIMEOUT = 14           # chat completion timeout
    AUTH_FAILURE = 15           # 401/403
    SLOT_CRASHED = 19           # process exited during probe

def classify_smoke_error(
    exc: BaseException,
    phase: str,
) -> SmokeExitCode:
    """Classify an httpx exception into a smoke exit code."""
    if isinstance(exc, httpx.ConnectTimeout):
        return SmokeExitCode.SERVER_NOT_READY if phase == "listen" else SmokeExitCode.HTTP_API_ERROR

    if isinstance(exc, httpx.ConnectError):
        # DNS failure, connection refused, connection reset
        return SmokeExitCode.HTTP_API_ERROR

    if isinstance(exc, httpx.ReadTimeout):
        if phase == "chat":
            return SmokeExitCode.CHAT_TIMEOUT
        return SmokeExitCode.HTTP_API_ERROR

    if isinstance(exc, httpx.NetworkError):
        return SmokeExitCode.HTTP_API_ERROR

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return SmokeExitCode.AUTH_FAILURE
        # 404, 5xx handled at the caller level (HTTP response is available)
        return SmokeExitCode.HTTP_API_ERROR

    # Fallback
    return SmokeExitCode.HTTP_API_ERROR
```

### Rationale
- `httpx.RequestError` is the base class for all network-layer failures (connect timeout, read timeout, connect error, network error). Subclassing allows precise classification.
- `httpx.HTTPStatusError` is raised only when `response.raise_for_status()` is called, meaning an HTTP response was received. This cleanly separates network-layer errors from HTTP-level errors.
- The spec maps:
  - **Exit 10** (SERVER_NOT_READY): `ConnectTimeout` during listen phase
  - **Exit 11** (HTTP_API_ERROR): `ConnectError`, `NetworkError`, `ReadTimeout` during HTTP phases, plus non-2xx HTTP responses
  - **Exit 13** (MODEL_NOT_FOUND): HTTP 200 with empty `models` array, or model ID mismatch — these are business logic errors, not exceptions
  - **Exit 15** (AUTH_FAILURE): HTTP 401/403 — caught as `HTTPStatusError` with specific status code
- `ReadTimeout` during chat phase should map to exit 14 (CHAT_TIMEOUT), not 11, per the spec's exit code table.

### Exception Hierarchy (httpx)
```
BaseException
 └── httpx.HTTPError
      ├── httpx.RequestError          ← network-layer failures
      │    ├── httpx.ConnectTimeout   ← TCP connect timed out
      │    ├── httpx.ReadTimeout      ← no data within read timeout
      │    ├── httpx.WriteTimeout     ← send timed out (unlikely for smoke)
      │    ├── httpx.PoolTimeout      ← connection pool exhausted
      │    ├── httpx.ConnectError     ← DNS failure, connection refused, reset
      │    └── httpx.NetworkError     ← other network failures
      └── httpx.HTTPStatusError       ← HTTP response with error status (4xx/5xx)
```

### Alternatives Considered
- **Catching `Exception` broadly**: Too coarse — would swallow unexpected errors and lose the ability to distinguish between network and HTTP errors.
- **Checking `exc.__class__.__name__` strings**: Fragile and breaks on httpx version changes. Use isinstance checks against the actual exception hierarchy.

---

## Decision 3: Per-Probe httpx.Client with Context Manager

### Decision
Create a new `httpx.Client` per smoke probe (per slot), using it as a context manager for automatic resource cleanup. Do NOT share a single client across slots.

```python
from contextlib import contextmanager
import httpx

@contextmanager
def smoke_client(
    host: str,
    port: int,
    timeout: httpx.Timeout,
    api_key: str | None = None,
):
    """Create an httpx.Client scoped to a single smoke probe."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(
        base_url=f"http://{host}:{port}",
        headers=headers,
        timeout=timeout,
        follow_redirects=False,  # smoke probes should not follow redirects
    ) as client:
        yield client
```

### Rationale
- **Different hosts/ports**: Each smoke probe targets a different port (e.g., 8080 for summary, 8081 for qwen35). While httpx.Client can handle multiple hosts, connection pooling benefits only apply to requests to the *same* host. Since smoke probes are sequential (not concurrent), pooling across slots provides no benefit.
- **No retry semantics**: The spec requires exactly-once probing. Creating a fresh client per probe is cleaner than managing a shared client's state.
- **Resource cleanup**: The context manager (`with httpx.Client() as client:`) ensures connection pool resources are released even if the probe fails with an exception.
- **Memory/performance**: Creating a client per probe is negligible overhead — each client uses ~100-200 bytes for the pool structure. With only 2 slots max, total memory is <1 KB.
- **API key isolation**: Each probe may use a different API key (resolved from CLI > config > env). Per-client header injection avoids mutating shared state.

### Alternatives Considered
- **Shared client with `headers` override**: Could pass `headers` per-request to override the Authorization header. However, this introduces subtle mutation risks if the client's default headers are modified. Per-client isolation is safer.
- **`httpx.Client()` without context manager**: Rejected — the spec's no-retry, single-phase model means exceptions are common, and the context manager ensures cleanup.

### Thread Safety Note
The `llama_manager` library is pure and does not spawn threads for smoke probes. The smoke probe runs synchronously in the CLI process. No thread-safety concerns exist for the smoke probe client. However, if a future TUI integration needs concurrent smoke probes (e.g., `smoke both` from the TUI monitor), each thread should have its own client instance — httpx.Client is NOT thread-safe for concurrent use.

---

## Decision 4: API Key Injection via Client Headers

### Research Finding
The M4 spec defines an API key precedence chain: CLI flag > config field > environment variable. The `Authorization: Bearer <token>` header is the standard pattern for OpenAI-compatible APIs.

### Proposed Approach
Inject the API key as a `Bearer` token in the `Authorization` header at the client level (not per-request). This is the standard OpenAI-compatible API authentication pattern.

```python
def resolve_smoke_api_key(
    cli_key: str | None = None,
    config_key: str | None = None,
) -> str | None:
    """Resolve API key with precedence: CLI > config > env.

    Precedence (highest to lowest):
    1. --api-key CLI flag
    2. smoke.api_key from config
    3. LLM_RUNNER_SMOKE_API_KEY environment variable
    """
    if cli_key:
        return cli_key
    if config_key:
        return config_key
    return os.environ.get("LLM_RUNNER_SMOKE_API_KEY")
```

### Rationale
- **OpenAI-compatible convention**: The llama.cpp server (and OpenAI's API) expects `Authorization: Bearer <token>`. This is the standard pattern.
- **Client-level injection**: Setting the header on the client means all requests (`/v1/models`, `/v1/chat/completions`) automatically include the key. No per-request header management needed.
- **Precedence chain**: The spec requires CLI > config > env precedence. The `resolve_smoke_api_key` function implements this as a simple short-circuit chain.
- **Security**: The API key should NOT be logged. The `llama_manager` library's `sys.stderr` logging should redact the key (e.g., replace with `***`).

### Note on Implementation Status
**Implemented**: `resolve_api_key()` in `llama_manager/smoke.py` implements the precedence: returns the explicit key if non-empty, otherwise falls back to `LLM_RUNNER_API_KEY` env var. The CLI layer (`llama_cli/smoke_cli.py`) resolves the API key using this precedence chain and passes the resolved value to the library.

### Alternatives Considered
- **`httpx.BasicAuth`**: Designed for username/password pairs, not bearer tokens. Would produce `Authorization: Basic <base64>` header, which llama.cpp does not expect.
- **Custom `httpx.Auth` class**: Overkill for a static Bearer token. The header injection approach is simpler and equally correct.
- **Per-request headers**: Would work but requires passing headers to every request call, increasing code duplication and error risk.

### Thread Safety
Since smoke probes are synchronous and single-threaded, no thread-safety concerns exist. If concurrent probes are added in the future, each probe's client should be created independently (as per Decision 3).

---

## Decision 5: JSON Response Parsing for OpenAI-Compatible Endpoints

### Decision
Use `response.json()` for non-streaming JSON responses. For `/v1/models`, parse the `models` array. For chat completion, parse the `choices[0].message.content` field and validate the response model ID.

```python
def probe_models_endpoint(
    client: httpx.Client,
    expected_model_id: str,
) -> tuple[bool, str | None]:
    """Probe /v1/models endpoint.

    Returns:
        (success, failure_reason)
    """
    response = client.get("/v1/models")

    if response.status_code == 404:
        # Endpoint not supported — skip to chat probe
        return True, None  # "skipped — endpoint not supported"

    if response.status_code >= 500:
        return False, "server_error"

    data = response.json()

    if not isinstance(data, dict) or "models" not in data:
        return False, "invalid_response_format"

    models = data["models"]
    if not isinstance(models, list) or len(models) == 0:
        return False, "no_models_available"  # exit code 13

    first_model = models[0]
    if not isinstance(first_model, dict) or "id" not in first_model:
        return False, "invalid_model_entry"

    if first_model["id"] != expected_model_id:
        return False, f"model_id_mismatch: expected={expected_model_id}, got={first_model['id']}"

    return True, None
```

```python
def probe_chat_completion(
    client: httpx.Client,
    model_id: str,
    max_tokens: int = 16,
    prompt: str = "Respond with exactly one word.",
) -> tuple[bool, str | None, str | None]:
    """Probe /v1/chat/completions with minimal request.

    Returns:
        (success, response_model_id, response_content)
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=payload)

    if response.status_code in (401, 403):
        return False, None, "auth_failure"

    if response.status_code >= 500:
        return False, None, "server_error"

    data = response.json()

    # Validate response structure
    if not isinstance(data, dict) or "choices" not in data:
        return False, None, "invalid_response_format"

    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return False, None, "no_choices"

    choice = choices[0]
    if not isinstance(choice, dict) or "message" not in choice:
        return False, None, "invalid_choice_format"

    message = choice["message"]
    content = message.get("content", "")
    response_model = data.get("model", "")

    return True, response_model, content
```

### Rationale
- **Non-streaming for smoke**: The spec requires a minimal chat completion probe. Using `stream: False` simplifies response handling — the entire response is available via `response.json()`.
- **OpenAI-compatible response format**: llama.cpp's OpenAI-compatible API returns responses in the OpenAI format: `{"choices": [{"message": {"content": "...", "role": "assistant"}, "finish_reason": "...", "index": 0}], "model": "...", ...}`.
- **Model ID validation**: The spec requires comparing the resolved model ID from GGUF metadata with the server's response. The chat probe's `model` field in the response provides the authoritative server-side model ID.
- **Error handling**: HTTP status codes are checked before parsing JSON. This prevents `json.JSONDecodeError` on non-JSON error responses (e.g., HTML error pages from llama.cpp).

### Alternatives Considered
- **Streaming responses (`stream: True`)**: Would allow first-token timing measurement but adds complexity (iterating `response.iter_lines()`, handling JSONL fragments). For MVP, non-streaming is sufficient and simpler to test.
- **`response.text` + `json.loads()`**: Equivalent to `response.json()` but more verbose. `response.json()` handles content-type validation automatically.

### Streaming JSONL Note
If streaming is added in a future iteration, use:
```python
with client.stream("POST", "/v1/chat/completions", json=payload) as response:
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            # Handle delta content
```
This uses `response.iter_lines()` to parse JSONL line-by-line.

---

## File Structure Recommendations

Based on the M4 spec and existing architecture:

```text
src/llama_manager/
  smoke_probe.py         # Pure library: probe logic, error classification
  # (no imports from llama_cli)

src/llama_cli/
  smoke.py               # I/O layer: CLI smoke command, output formatting
  # (imports from llama_manager.smoke_probe)

src/tests/
  test_smoke_probe.py    # Unit tests: mocked httpx, no real servers
```

### Module Boundaries

**`llama_manager/smoke_probe.py`** (pure library):
- `SmokeProbeResult` dataclass (per-slot outcome)
- `classify_smoke_error()` — maps httpx exceptions to exit codes
- `probe_tcp_connect()` — uses `httpx.Client` with connect timeout
- `probe_models_endpoint()` — GET `/v1/models`
- `probe_chat_completion()` — POST `/v1/chat/completions`
- `run_smoke_probe()` — orchestrates the three-phase probe flow
- `resolve_smoke_api_key()` — CLI > config > env precedence
- No argparse, no Rich, no subprocess

**`llama_cli/smoke.py`** (I/O layer):
- `smoke_cli()` — handles `smoke both` / `smoke slot <id>` subcommands
- `format_smoke_result()` — human-readable or JSON output (FR-020)
- `print_smoke_header()` — prints `smoke: slot=... model=... provenance=...`

---

## Testing Strategy

### Unit Tests (in `tests/test_smoke_probe.py`)

```python
import pytest
import httpx
from llama_manager.smoke_probe import (
    SmokeExitCode,
    classify_smoke_error,
    probe_models_endpoint,
    probe_chat_completion,
    resolve_smoke_api_key,
)

class TestClassifySmokeError:
    def test_connect_timeout_during_listen(self) -> None:
        exc = httpx.ConnectTimeout("connection timed out")
        assert classify_smoke_error(exc, "listen") == SmokeExitCode.SERVER_NOT_READY

    def test_connect_error_during_http(self) -> None:
        exc = httpx.ConnectError("connection refused")
        assert classify_smoke_error(exc, "models") == SmokeExitCode.HTTP_API_ERROR

    def test_auth_failure_401(self) -> None:
        response = httpx.Response(401, request=httpx.Request("GET", "http://localhost:8080/v1/models"))
        exc = httpx.HTTPStatusError("401", request=response.request, response=response)
        assert classify_smoke_error(exc, "models") == SmokeExitCode.AUTH_FAILURE

    def test_read_timeout_during_chat(self) -> None:
        exc = httpx.ReadTimeout("read timed out")
        assert classify_smoke_error(exc, "chat") == SmokeExitCode.CHAT_TIMEOUT

class TestResolveSmokeApiKey:
    def test_cli_key_takes_precedence(self) -> None:
        assert resolve_smoke_api_key(cli_key="cli_key", config_key="config_key") == "cli_key"

    def test_config_key_used_when_no_cli(self) -> None:
        assert resolve_smoke_api_key(cli_key=None, config_key="config_key") == "config_key"

    def test_env_used_when_no_cli_or_config(self) -> None:
        with patch.dict(os.environ, {"LLM_RUNNER_SMOKE_API_KEY": "env_key"}):
            assert resolve_smoke_api_key(cli_key=None, config_key=None) == "env_key"
```

### Mocking httpx in Tests

Use `pytest-httpx` fixture (add to `[project.optional-dependencies]`):

```bash
uv add --dev pytest-httpx
```

```python
import pytest
import httpx
from pytest_httpx import HTTPXMock

def test_probe_models_endpoint_success(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="/v1/models",
        json={"models": [{"id": "Qwen3.5-2B", "name": "Qwen3.5-2B", "root": "/"}]},
        status_code=200,
    )
    with httpx.Client(base_url="http://127.0.0.1:8080") as client:
        success, reason = probe_models_endpoint(client, "Qwen3.5-2B")
        assert success is True
        assert reason is None
```

---

## CI Quality Gates

All plans must note the CI gates:

1. **lint** — `uv run ruff check .` + `uv run ruff format --check`
2. **typecheck** — `uv run pyright`
3. **test** — `uv run pytest` with coverage
4. **security** — `uv run pip-audit` (for httpx and pytest-httpx dependencies)

### Dependencies to Add

```toml
[project.dependencies]
httpx>=0.28.0  # For smoke probes (Constraint C-002)

[project.optional-dependencies]
dev = [
    ...
    "pytest-httpx>=0.30.0",  # For mocking httpx in tests
]
```

### Ruff Security Considerations

The `.s` (flake8-bandit) rules in `pyproject.toml` may flag:
- `S104` (hardcoded bindings): Not applicable — smoke probes connect to localhost
- `S113` (try-except without specific exception): Should use specific httpx exception types
- `S310` (URL validation): Ensure URLs are validated before use (use `httpx.URL` constructor)

---

## Implementation Sequence

1. **Phase 1**: Add `httpx` dependency to `pyproject.toml`
2. **Phase 2**: Implement `llama_manager/smoke_probe.py` (pure library)
3. **Phase 3**: Implement `llama_cli/smoke.py` (CLI layer)
4. **Phase 4**: Add `pytest-httpx` to dev deps and write unit tests
5. **Phase 5**: Update `cli_parser.py` to handle `smoke` subcommand
6. **Phase 6**: Integration with existing `Config` for smoke parameters

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `httpx` adds a new runtime dependency | It's a single lightweight dependency (~200 KB). The spec marks it as a constraint (C-002). |
| `httpx` version compatibility | Pin to `>=0.28.0` (stable, well-tested). Test against latest release in CI. |
| Network errors in CI | All smoke tests use `pytest-httpx` mocking — no real network calls. |
| Thread safety in future TUI integration | Document that `httpx.Client` is not thread-safe. Each probe creates its own client. |
| API key leakage in logs | Implement redaction in `llama_manager`'s stderr logging (replace with `***`). |
