"""probe package — smoke testing for llama.cpp inference servers."""

from .provenance import ProvenanceRecord, resolve_provenance
from .smoke import (
    _EXIT_CODE_MAP,
    ConsecutiveFailureCounter,
    SmokeCompositeReport,
    SmokeProbeResult,
    _handle_models_status,
    _models_failure_result,
    _probe_chat,
    _probe_models,
    _tcp_connect,
    compute_overall_exit_code,
    probe_slot,
    resolve_api_key,
    resolve_model_id_from_gguf,
)

__all__ = [
    # Public probe API
    "probe_slot",
    "resolve_api_key",
    "resolve_model_id_from_gguf",
    # Result types
    "SmokeProbeResult",
    "SmokeCompositeReport",
    "ConsecutiveFailureCounter",
    "ProvenanceRecord",
    # Computation
    "compute_overall_exit_code",
    # Provenance
    "resolve_provenance",
    # Internal (exported for tests)
    "_EXIT_CODE_MAP",
    "_tcp_connect",
    "_probe_models",
    "_probe_chat",
    "_handle_models_status",
    "_models_failure_result",
]
