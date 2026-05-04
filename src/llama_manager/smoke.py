"""Backward compatibility shim. Import from llama_manager.probe instead."""

# Re-export stdlib/3rd-party modules that tests mock on this module
import socket  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401

import httpx  # noqa: F401

from .config import SmokePhase, SmokeProbeStatus  # noqa: F401
from .probe import *  # noqa: F401, F403
from .probe.provenance import (  # noqa: F401
    ProvenanceRecord,
    _resolve_sha,
    _resolve_version,
    resolve_provenance,
)
from .probe.smoke import (  # noqa: F401
    _EXIT_CODE_MAP,
    ConsecutiveFailureCounter,
    SmokeCompositeReport,
    SmokeProbeResult,
    _ensure_report_dir,
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
