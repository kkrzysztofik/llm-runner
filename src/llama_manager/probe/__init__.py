"""probe package — smoke testing for llama.cpp inference servers."""

from .provenance import ProvenanceRecord, resolve_provenance
from .smoke import (
    ConsecutiveFailureCounter,
    SmokeCompositeReport,
    SmokeProbeResult,
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
]
