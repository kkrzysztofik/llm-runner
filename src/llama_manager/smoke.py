"""Smoke probe for verifying llama.cpp inference server health.

Performs sequential phase-based probing (TCP connect → /v1/models →
/v1/chat/completions) and produces structured results suitable for
CLI output, JSON export, or TUI integration.

Pure library — no argparse, no Rich, no subprocess at module level.
"""

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import (
    SmokeFailurePhase,
    SmokePhase,
    SmokeProbeConfiguration,
    SmokeProbeStatus,
)
from .metadata import extract_gguf_metadata, normalize_filename

# ---------------------------------------------------------------------------
# API key resolution (env fallback — pure library)
# ---------------------------------------------------------------------------

_API_KEY_ENV_VAR: str = "LLM_RUNNER_API_KEY"


def resolve_api_key(explicit_key: str = "") -> str:
    """Resolve API key with env-fallback.

    Strips ``explicit_key`` first; if the stripped result is empty,
    falls back to the ``LLM_RUNNER_API_KEY`` environment variable
    (stripped).  This provides a consistent resolution order (explicit
    → env) without importing CLI modules.

    Args:
        explicit_key: API key passed explicitly by the caller.

    Returns:
        The resolved API key (stripped, may be empty string).
    """
    stripped = explicit_key.strip()
    if stripped:
        return stripped
    env_key = os.environ.get(_API_KEY_ENV_VAR, "")
    if env_key:
        return env_key.strip()
    return ""


# ---------------------------------------------------------------------------
# GGUF model ID resolution
# ---------------------------------------------------------------------------


def resolve_model_id_from_gguf(model_path: str) -> str | None:
    """Resolve model ID from GGUF metadata.

    Extracts the ``general.name`` from GGUF metadata via the ``gguf``
    library.  Returns the normalized filename stem as a fallback when
    ``general.name`` is missing.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        The model ID string, or ``None`` if extraction fails.

    """
    try:
        record = extract_gguf_metadata(model_path)
    except (OSError, ValueError, TimeoutError):
        return None

    # Prefer general.name from GGUF metadata
    if record.general_name:
        return normalize_filename(record.general_name)

    # Fallback to normalized filename stem
    return record.normalized_stem


# ---------------------------------------------------------------------------
# Exit code mapping (per spec Appendix B)
# ---------------------------------------------------------------------------

_EXIT_CODE_MAP: dict[SmokeProbeStatus, int] = {
    SmokeProbeStatus.PASS: 0,
    SmokeProbeStatus.FAIL: 10,
    SmokeProbeStatus.TIMEOUT: 13,
    SmokeProbeStatus.CRASHED: 19,
    SmokeProbeStatus.MODEL_NOT_FOUND: 14,
    SmokeProbeStatus.AUTH_FAILURE: 15,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    """Git provenance for the running server binary.

    Attributes:
        sha: Full git SHA of the llama.cpp HEAD at build time.
        version: Package version from ``importlib.metadata``.
    """

    sha: str
    version: str


@dataclass
class SmokeProbeResult:
    """Result of probing a single slot.

    Attributes:
        slot_id: Normalized slot identifier.
        status: Overall probe status.
        phase_reached: Last phase that completed successfully.
        failure_phase: Phase at which the probe failed (None if passed).
        model_id: Resolved model ID from the server (None if unavailable).
        latency_ms: Time to complete the full probe in milliseconds
            (None if probe did not complete).
        provenance: Git provenance of the server binary.
    """

    slot_id: str
    status: SmokeProbeStatus
    phase_reached: SmokePhase
    failure_phase: SmokeFailurePhase | None = None
    model_id: str | None = None
    latency_ms: int | None = None
    provenance: ProvenanceRecord = field(
        default_factory=lambda: ProvenanceRecord(sha="unknown", version="dev")
    )

    @property
    def exit_code(self) -> int:
        """Return the exit code corresponding to this result's status."""
        return _EXIT_CODE_MAP.get(self.status, 10)


@dataclass
class SmokeCompositeReport:
    """Composite report from probing multiple slots.

    Attributes:
        results: Per-slot smoke probe results.
        overall_status: Aggregate status (PASS if all pass, else worst).
        overall_exit_code: Exit code for the smoke command.
    """

    results: list[SmokeProbeResult]

    @property
    def overall_status(self) -> SmokeProbeStatus:
        """Return the worst status across all results."""
        if not self.results:
            return SmokeProbeStatus.PASS
        # Priority: CRASHED > AUTH_FAILURE > MODEL_NOT_FOUND > TIMEOUT > FAIL > PASS
        _PRIORITY = [
            SmokeProbeStatus.CRASHED,
            SmokeProbeStatus.AUTH_FAILURE,
            SmokeProbeStatus.MODEL_NOT_FOUND,
            SmokeProbeStatus.TIMEOUT,
            SmokeProbeStatus.FAIL,
            SmokeProbeStatus.PASS,
        ]
        for status in _PRIORITY:
            if any(r.status == status for r in self.results):
                return status
        return SmokeProbeStatus.PASS

    @property
    def overall_exit_code(self) -> int:
        """Return the highest/worst exit code across all results."""
        return compute_overall_exit_code(self.results)

    @property
    def pass_count(self) -> int:
        """Number of slots that passed."""
        return sum(1 for r in self.results if r.status == SmokeProbeStatus.PASS)

    @property
    def fail_count(self) -> int:
        """Number of slots that failed."""
        return len(self.results) - self.pass_count


@dataclass
class ConsecutiveFailureCounter:
    """Tracks consecutive smoke failures per slot.

    Used to implement exponential backoff or auto-restart logic.

    Attributes:
        slot_id: Normalized slot identifier.
        count: Number of consecutive failures.
        model_id_override: Model ID that was last probed.
    """

    slot_id: str
    count: int = 0
    model_id_override: str | None = None

    def record_failure(self, model_id: str | None = None) -> None:
        """Record a failure for this slot.

        Args:
            model_id: The model ID that was being probed.
        """
        self.count += 1
        if model_id is not None:
            self.model_id_override = model_id

    def record_success(self) -> None:
        """Record a success and reset the counter."""
        self.reset()

    def reset(self) -> None:
        """Reset the failure counter."""
        self.count = 0
        self.model_id_override = None


# ---------------------------------------------------------------------------
# Report directory management (T072)
# ---------------------------------------------------------------------------


def _ensure_report_dir(report_dir: Path) -> Path:
    """Create report directory if it doesn't exist.

    Creates the directory with owner-only permissions (0o700).
    Returns the path to the report directory regardless of whether
    it already existed or was newly created.

    Args:
        report_dir: Path to the report directory.

    Returns:
        The report directory path (absolute).
    """
    report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    return report_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_slot(
    host: str,
    port: int,
    smoke_cfg: SmokeProbeConfiguration,
    model_id: str | None = None,
    expected_model_id: str | None = None,
    model_path: str | None = None,
) -> SmokeProbeResult:
    """Probe a single slot through sequential smoke test phases.

    Phases are attempted in order:
    1. TCP connect (listen/accept)
    2. GET /v1/models (unless skipped)
    3. POST /v1/chat/completions

    No retries — each phase is attempted exactly once.

    Args:
        host: Server hostname or IP address.
        port: Server port number.
        smoke_cfg: Smoke probe configuration.
        model_id: Resolved model ID (from GGUF metadata or config).
        expected_model_id: Expected model ID for /v1/models comparison.
        model_path: Optional path to GGUF file for model ID resolution
            via ``general.name`` metadata.

    Returns:
        A SmokeProbeResult with the probe outcome.

    """
    start_time = time.monotonic()
    provenance = resolve_provenance()

    # Determine model ID for probe — precedence: explicit → override → expected → GGUF
    resolved_model_id = model_id or smoke_cfg.model_id_override or expected_model_id or ""
    if not resolved_model_id and model_path:
        resolved_model_id = resolve_model_id_from_gguf(model_path) or ""

    # Phase 1: TCP connect
    phase = SmokePhase.LISTEN
    try:
        _tcp_connect(host, port, smoke_cfg.listen_timeout_s)
    except TimeoutError:
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.TIMEOUT,
            phase_reached=phase,
            failure_phase=SmokeFailurePhase.LISTEN,
            model_id=None,
            latency_ms=int((time.monotonic() - start_time) * 1000),
            provenance=provenance,
        )
    except OSError:
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=phase,
            failure_phase=SmokeFailurePhase.LISTEN,
            model_id=None,
            latency_ms=int((time.monotonic() - start_time) * 1000),
            provenance=provenance,
        )

    # Resolve API key once — consistent stripping for all phases (explicit → env)
    resolved_api_key = resolve_api_key(smoke_cfg.api_key)

    # Phase 2: Models discovery
    phase = SmokePhase.MODELS
    if not smoke_cfg.skip_models_discovery:
        result, discovered_model_id = _probe_models(
            host,
            port,
            smoke_cfg.http_request_timeout_s,
            resolved_api_key,
            smoke_cfg.model_id_override or expected_model_id or resolved_model_id,
            smoke_cfg.listen_timeout_s,
        )
        if result is not None:
            return result
        # Phase 2 passed or was skipped (404 → proceed to Phase 3)
        # Use discovered model ID if no model ID was resolved yet
        if not resolved_model_id and discovered_model_id:
            resolved_model_id = discovered_model_id

    # Phase 3: Chat completion
    phase = SmokePhase.CHAT
    result = _probe_chat(
        host,
        port,
        smoke_cfg,
        resolved_model_id,
        resolved_api_key,
    )
    if result is not None:
        return result

    # All phases passed
    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    return SmokeProbeResult(
        slot_id=f"{host}:{port}",
        status=SmokeProbeStatus.PASS,
        phase_reached=SmokePhase.COMPLETE,
        model_id=resolved_model_id,
        latency_ms=elapsed_ms,
        provenance=provenance,
    )


def _tcp_connect(host: str, port: int, timeout_s: int) -> None:
    """Attempt a TCP connection with timeout.

    Args:
        host: Target hostname.
        port: Target port.
        timeout_s: Connection timeout in seconds.

    Raises:
        socket.timeout: If connection times out.
        OSError: If connection is refused or fails.

    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        sock.connect((host, port))
    finally:
        sock.close()


def _models_failure_result(
    host: str,
    port: int,
    status: SmokeProbeStatus,
    model_id: str | None = None,
) -> SmokeProbeResult:
    """Construct a failure SmokeProbeResult for the models phase."""
    return SmokeProbeResult(
        slot_id=f"{host}:{port}",
        status=status,
        phase_reached=SmokePhase.MODELS,
        failure_phase=SmokeFailurePhase.MODELS,
        model_id=model_id,
        provenance=resolve_provenance(),
    )


def _probe_models(
    host: str,
    port: int,
    timeout_s: int,
    api_key: str,
    expected_model_id: str,
    listen_timeout_s: int = 5,
) -> tuple[SmokeProbeResult | None, str | None]:
    """Probe /v1/models endpoint.

    Returns a tuple of (SmokeProbeResult | None, discovered_model_id | None).
    On success, returns (None, discovered_model_id). On failure, returns
    (SmokeProbeResult, None).

    Args:
        host: Server hostname.
        port: Server port.
        timeout_s: HTTP request timeout.
        api_key: API key for authentication.
        expected_model_id: Expected model ID from the response.
        listen_timeout_s: TCP connect timeout (used for httpx connect timeout).

    Returns:
        Tuple of (failure result or None, discovered model id or None).

    """
    url = f"http://{host}:{port}/v1/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Use explicit httpx.Timeout with distinct connect/read/write/pool values
    timeout = httpx.Timeout(
        connect=listen_timeout_s,
        read=timeout_s,
        write=timeout_s,
        pool=timeout_s,
    )

    # Attempt HTTP request
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
    except httpx.TimeoutException:
        return (_models_failure_result(host, port, SmokeProbeStatus.TIMEOUT), None)
    except (httpx.ConnectError, httpx.NetworkError, httpx.UnsupportedProtocol):
        return (_models_failure_result(host, port, SmokeProbeStatus.FAIL), None)

    # Handle HTTP status codes
    if response.status_code == 404:
        # Endpoint not supported — proceed to Phase 3
        return (None, None)

    result = _handle_models_status(host, port, response, expected_model_id)
    if result is not None:
        return result

    # Parse response
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return (_models_failure_result(host, port, SmokeProbeStatus.FAIL), None)

    models = data.get("data", [])
    if not models:
        return (
            _models_failure_result(host, port, SmokeProbeStatus.MODEL_NOT_FOUND),
            None,
        )

    # Check model ID match — accept if expected_model_id appears anywhere
    discovered_model_id: str | None = None
    for model_entry in models:
        model_id_val = model_entry.get("id", "")
        if expected_model_id and model_id_val == expected_model_id:
            discovered_model_id = model_id_val
            break
        if discovered_model_id is None:
            discovered_model_id = model_id_val

    if expected_model_id and discovered_model_id != expected_model_id:
        return (
            _models_failure_result(
                host,
                port,
                SmokeProbeStatus.MODEL_NOT_FOUND,
                model_id=discovered_model_id,
            ),
            None,
        )

    return (None, discovered_model_id)  # Phase 2 passed


def _handle_models_status(
    host: str,
    port: int,
    response: httpx.Response,
    expected_model_id: str,
) -> tuple[SmokeProbeResult, str | None] | None:
    """Handle HTTP status codes for the models endpoint.

    Returns (SmokeProbeResult, None) for terminal statuses, or None
    to indicate the response should continue to JSON parsing.

    Args:
        host: Server hostname.
        port: Server port.
        response: The HTTP response.
        expected_model_id: Expected model ID for validation.

    Returns:
        Tuple of (failure result, None) for terminal cases, or None.
    """
    if response.status_code in (401, 403):
        return (_models_failure_result(host, port, SmokeProbeStatus.AUTH_FAILURE), None)

    if response.status_code >= 500:
        return (_models_failure_result(host, port, SmokeProbeStatus.FAIL), None)

    if response.status_code != 200:
        return (_models_failure_result(host, port, SmokeProbeStatus.FAIL), None)

    return None


def _probe_chat(
    host: str,
    port: int,
    smoke_cfg: SmokeProbeConfiguration,
    model_id: str,
    api_key: str = "",
) -> SmokeProbeResult | None:
    """Probe /v1/chat/completions endpoint.

    Returns a SmokeProbeResult on failure (terminal), or None on success.

    Args:
        host: Server hostname.
        port: Server port.
        smoke_cfg: Smoke probe configuration.
        model_id: Model ID to use for chat completion.
        api_key: Resolved API key for authentication.

    Returns:
        SmokeProbeResult on failure, None on success.

    """
    url = f"http://{host}:{port}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": smoke_cfg.prompt}],
        "max_tokens": smoke_cfg.max_tokens,
        "temperature": 0,
        "stream": False,
    }

    # Use explicit httpx.Timeout with distinct connect/read/write/pool values
    timeout = httpx.Timeout(
        connect=smoke_cfg.listen_timeout_s,
        read=smoke_cfg.first_token_timeout_s,
        write=smoke_cfg.http_request_timeout_s,
        pool=smoke_cfg.total_chat_timeout_s,
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.TIMEOUT,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )
    except (httpx.ConnectError, httpx.NetworkError, httpx.UnsupportedProtocol):
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    # Handle HTTP status codes
    if response.status_code in (401, 403):
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.AUTH_FAILURE,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    if response.status_code != 200:
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    # Parse response and check choices
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    choices = data.get("choices", [])
    if not choices:
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    # Validate choices[0] contains a non-empty message with non-empty content
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    if not isinstance(message, dict):
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )
    content = message.get("content")
    if not content or (isinstance(content, str) and not content.strip()):
        return SmokeProbeResult(
            slot_id=f"{host}:{port}",
            status=SmokeProbeStatus.FAIL,
            phase_reached=SmokePhase.CHAT,
            failure_phase=SmokeFailurePhase.CHAT,
            model_id=model_id,
            provenance=resolve_provenance(),
        )

    return None  # Phase 3 passed


def resolve_provenance() -> ProvenanceRecord:
    """Resolve git provenance for the running server binary.

    Reads the SHA from ``.git/HEAD`` in the llama.cpp root directory
    and the package version from ``importlib.metadata``.

    Returns:
        A ProvenanceRecord with sha and version.

    """
    sha = _resolve_sha()
    version = _resolve_version()
    return ProvenanceRecord(sha=sha, version=version)


def _resolve_sha() -> str:
    """Resolve the git SHA from the llama.cpp repository.

    Reads ``.git/HEAD`` and runs ``git rev-parse`` to get the full SHA.

    Returns:
        Full git SHA, or 'unknown' if resolution fails.

    """
    from subprocess import CalledProcessError, run

    git_head = Path("src/llama.cpp/.git/HEAD")
    if not git_head.exists():
        return "unknown"

    try:
        head_content = git_head.read_text().strip()
        # .git/HEAD can contain a ref (ref: refs/heads/main) or a direct SHA
        if head_content.startswith("ref: "):
            ref_path = git_head.parent / head_content[5:]
            if ref_path.exists():
                sha = ref_path.read_text().strip()
                return sha[:7] if len(sha) > 7 else sha
        else:
            # Direct SHA reference
            return head_content[:7] if len(head_content) > 7 else head_content
    except OSError:
        pass

    # Fallback: try git rev-parse
    try:
        result = run(
            ["git", "-C", "src/llama.cpp", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha[:7] if len(sha) > 7 else sha
    except (FileNotFoundError, CalledProcessError, TimeoutError):
        pass

    return "unknown"


def _resolve_version() -> str:
    """Resolve the package version from importlib.metadata.

    Returns:
        Package version string, or 'dev' if unavailable.

    """
    try:
        from importlib.metadata import version as _version

        return _version("llm_runner")
    except Exception:
        return "dev"


def compute_overall_exit_code(results: list[SmokeProbeResult]) -> int:
    """Compute the overall exit code from a list of smoke probe results.

    Returns the highest exit code among all results (i.e., the worst
    failure).  If all results pass, returns 0.

    Args:
        results: List of SmokeProbeResult objects.

    Returns:
        Overall exit code (0 for all pass, otherwise worst failure).

    """
    if not results:
        return 0

    worst_code = 0
    for result in results:
        code = result.exit_code
        if code > worst_code:
            worst_code = code

    return worst_code
