"""Server command building and dry-run payload construction."""

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from ...common.security import redact_env_value
from ...config import (
    Config,
    ErrorDetail,
    ServerConfig,
    VRamRecommendation,
)

# ---------------------------------------------------------------------------
# Doctor diagnostics (T069)
# ---------------------------------------------------------------------------

_SPEC_TYPE_FLAG: Final = "--spec-type"
_SPEC_TYPE_DFLASH: Final = "dflash"
_SPEC_TYPE_DRAFT_MTP: Final = "draft-mtp"
_SPEC_TYPE_NGRAM_MOD: Final = "ngram-mod"


@dataclass
class DoctorCheckResult:
    """Result of a single doctor diagnostic check."""

    name: str
    status: Literal["pass", "warn", "fail"]
    message: str = ""


@dataclass
class DoctorReport:
    """Aggregated doctor diagnostic report."""

    checks: list[DoctorCheckResult]
    config: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Return JSON string representation."""
        return json.dumps(
            {
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "message": c.message,
                    }
                    for c in self.checks
                ],
                "config": self.config,
                "hardware": self.hardware,
            },
            indent=2,
        )

    def to_text(self) -> str:
        """Return human-readable text representation."""
        lines: list[str] = ["=== DOCTOR DIAGNOSTIC REPORT ==="]
        for check in self.checks:
            icon = {"pass": "\u2713", "warn": "\u26a0", "fail": "\u2717"}.get(check.status, "?")
            lines.append(f"  [{icon}] {check.name}: {check.message}")
        return "\n".join(lines)


# FR-003: Canonical dry-run payload types
@dataclass
class VllmEligibility:
    """FR-003: vLLM eligibility status for a slot."""

    eligible: bool
    reason: str


@dataclass
class DryRunValidationSummary:
    """FR-003: Aggregated validation results for a slot."""

    passed: bool
    checks: list[dict[str, Any]]


@dataclass
class DryRunSlotPayload:
    """FR-003: Canonical dry-run slot payload with deterministic field ordering."""

    slot_id: str
    binary_path: str
    command_args: list[str]
    model_path: str
    bind_address: str
    port: int
    environment_redacted: dict[str, str]
    openai_flag_bundle: dict[str, str | int | bool | None]
    hardware_notes: dict[str, str | None]
    vllm_eligibility: VllmEligibility
    warnings: list[str]
    validation_results: DryRunValidationSummary
    server_config: ServerConfig | None = None


def build_server_cmd(cfg: ServerConfig, default_bin: str | None = None) -> list[str]:
    """Build llama-server command arguments."""
    if cfg.server_bin:
        server_bin = cfg.server_bin
    elif default_bin:
        server_bin = default_bin
    else:
        server_bin = Config().paths.llama_server_bin_intel

    cmd = [
        server_bin,
        "--model",
        cfg.model,
        "--alias",
        cfg.alias,
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--split-mode",
        "layer",
        "--ctx-size",
        str(cfg.ctx_size),
        "--flash-attn",
        "on",
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--batch-size",
        str(cfg.batch_size),
        "--ubatch-size",
        str(cfg.ubatch_size),
        "--threads",
        str(cfg.threads),
        "--poll",
        str(cfg.poll_ms),
        "--n-predict",
        str(cfg.n_predict),
        "--parallel",
        str(cfg.parallel),
        "--host",
        cfg.bind_address,
        "--port",
        str(cfg.port),
        "--no-webui",
    ]

    if cfg.threads_batch > 0:
        cmd.extend(["--threads-batch", str(cfg.threads_batch)])
    if cfg.mmproj:
        cmd.extend(["--mmproj", cfg.mmproj])
    _append_speculative_flags(cmd, cfg)
    _append_optional_server_flags(cmd, cfg)

    return cmd


def _append_optional_server_flags(cmd: list[str], cfg: ServerConfig) -> None:
    """Append non-required server flags."""
    spec = cfg.spec_decode
    if cfg.main_gpu != 0:
        cmd.extend(["--main-gpu", str(cfg.main_gpu)])
    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if spec.reasoning_mode:
        cmd.extend(["--reasoning", spec.reasoning_mode])
    if spec.reasoning_format:
        cmd.extend(["--reasoning-format", spec.reasoning_format])
    if cfg.tensor_split:
        cmd.extend(["--tensor-split", cfg.tensor_split])
    if cfg.chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", cfg.chat_template_kwargs])
    if spec.reasoning_budget:
        cmd.extend(["--reasoning-budget", spec.reasoning_budget])
    if cfg.use_jinja:
        cmd.append("--jinja")
    if cfg.kv_unified:
        cmd.append("--kv-unified")
    if not cfg.mmproj_offload:
        cmd.append("--no-mmproj-offload")
    if cfg.mmap:
        cmd.append("--mmap")
    else:
        cmd.append("--no-mmap")
    if cfg.mlock:
        cmd.append("--mlock")
    if cfg.no_host_buffer:
        cmd.append("--no-host")


def _append_speculative_flags(cmd: list[str], cfg: ServerConfig) -> None:
    """Append llama-server speculative decoding flags when configured."""
    spec = cfg.spec_decode
    if spec.spec_type == _SPEC_TYPE_NGRAM_MOD:
        _append_ngram_speculative_flags(cmd, spec)
        return
    if spec.spec_type not in (_SPEC_TYPE_DRAFT_MTP, _SPEC_TYPE_DFLASH):
        return
    if spec.spec_type == _SPEC_TYPE_DRAFT_MTP:
        _append_draft_mtp_flags(cmd, spec)
        return
    _append_dflash_flags(cmd, spec)


def _append_ngram_speculative_flags(cmd: list[str], spec: Any) -> None:
    cmd.extend(
        [
            _SPEC_TYPE_FLAG,
            _SPEC_TYPE_NGRAM_MOD,
            "--spec-ngram-size-n",
            str(spec.spec_ngram_size_n),
            "--draft-min",
            str(spec.draft_min),
            "--draft-max",
            str(spec.draft_max),
        ]
    )


def _append_draft_mtp_flags(cmd: list[str], spec: Any) -> None:
    cmd.extend(
        [_SPEC_TYPE_FLAG, _SPEC_TYPE_DRAFT_MTP, "--spec-draft-n-max", str(spec.spec_draft_n_max)]
    )
    if spec.spec_draft_p_min > 0:
        cmd.extend(["--spec-draft-p-min", str(spec.spec_draft_p_min)])
    # llama-server flags omit "cache" (--spec-draft-type-k/v), unlike field names.
    if spec.spec_draft_cache_type_k:
        cmd.extend(["--spec-draft-type-k", spec.spec_draft_cache_type_k])
    if spec.spec_draft_cache_type_v:
        cmd.extend(["--spec-draft-type-v", spec.spec_draft_cache_type_v])
    if spec.spec_draft_device:
        cmd.extend(["--spec-draft-device", spec.spec_draft_device])


def _append_dflash_flags(cmd: list[str], spec: Any) -> None:
    cmd.extend([_SPEC_TYPE_FLAG, _SPEC_TYPE_DFLASH])
    if spec.spec_draft_model:
        cmd.extend(["--spec-draft-model", spec.spec_draft_model])
    if spec.spec_draft_hf:
        cmd.extend(["--spec-draft-hf", spec.spec_draft_hf])
    if spec.spec_draft_ngl:
        cmd.extend(["--spec-draft-ngl", str(spec.spec_draft_ngl)])
    if spec.spec_dflash_cross_ctx > 0:
        cmd.extend(["--spec-dflash-cross-ctx", str(spec.spec_dflash_cross_ctx)])


def sort_validation_errors(
    results: Sequence[ErrorDetail],
) -> list[ErrorDetail]:
    """Sort validation errors deterministically for T003 stable ordering."""
    slot_order: dict[str, int] = {}
    for i, r in enumerate(results):
        if r.slot_id not in slot_order:
            slot_order[r.slot_id] = i

    def sort_key(r: ErrorDetail) -> tuple[int, str]:
        slot_idx = slot_order[r.slot_id]
        failed_check = r.failed_check or ""
        return (slot_idx, failed_check)

    return sorted(results, key=sort_key)


def build_dry_run_slot_payload(
    cfg: ServerConfig,
    slot_id: str,
    validation_results: DryRunValidationSummary | None = None,
    warnings: list[str] | None = None,
) -> DryRunSlotPayload:
    """FR-003: Build canonical dry-run slot payload from ServerConfig + slot_id."""
    cmd = build_server_cmd(cfg)
    command_args = cmd[1:]

    environment_redacted = _build_environment_redacted()
    openai_flag_bundle = _build_openai_flag_bundle(cfg)
    hardware_notes = _build_hardware_notes(cfg)

    vllm_eligibility = VllmEligibility(
        eligible=False,
        reason="vllm is not launch-eligible in PRD M1 - only llama_cpp supported",
    )

    if validation_results is None:
        validation_results = DryRunValidationSummary(
            passed=True,
            checks=[],
        )

    if warnings is None:
        warnings = []

    return DryRunSlotPayload(
        slot_id=slot_id,
        binary_path=cmd[0],
        command_args=command_args,
        model_path=cfg.model,
        bind_address=cfg.bind_address,
        port=cfg.port,
        environment_redacted=environment_redacted,
        openai_flag_bundle=openai_flag_bundle,
        hardware_notes=hardware_notes,
        vllm_eligibility=vllm_eligibility,
        warnings=warnings,
        validation_results=validation_results,
        server_config=cfg,
    )


def _build_environment_redacted() -> dict[str, str]:
    """FR-007: Build environment variable map with sensitive values redacted."""
    env_vars_to_check = [
        "PATH",
        "HOME",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "SYCL_DEVICE_SELECTOR",
        "HF_HOME",
        "HF_HUB_CACHE",
    ]

    result: dict[str, str] = {}

    for key in env_vars_to_check:
        value = os.environ.get(key, "")
        result[key] = redact_env_value(value, key)

    for key in sorted(os.environ):
        if key not in result:
            result[key] = redact_env_value(os.environ[key], key)

    return result


def _build_openai_flag_bundle(cfg: ServerConfig) -> dict[str, str | int | bool | None]:
    """Build OpenAI API compatibility flag bundle."""
    chat_completion_supported = cfg.spec_decode.reasoning_mode in ("auto", "enabled")

    bundle: dict[str, str | int | bool | None] = {
        "--chat-format": "chatml" if chat_completion_supported else None,
        "--host": "127.0.0.1",
        "--openai": True,
        "--port": cfg.port,
    }

    return dict(sorted(bundle.items()))


def _build_hardware_notes(cfg: ServerConfig) -> dict[str, str | None]:
    """Build hardware notes dict describing backend and hardware."""
    backend = cfg.backend or "llama_cpp"
    device = cfg.device or "auto"
    device_id, device_name = _parse_device_details(device)

    return {
        "backend": backend,
        "device_id": device_id,
        "device_name": device_name,
        "driver_version": None,
        "runtime_version": None,
    }


def _parse_device_details(device: str) -> tuple[str | None, str]:
    normalized = device.strip()
    lower = normalized.lower()
    upper = normalized.upper()
    if lower == "auto":
        return (None, device)

    if lower.startswith("cuda:"):
        return _cuda_device_details(normalized)

    if upper.startswith("SYCL"):
        return _sycl_device_details(normalized)

    if lower.startswith("sycl:"):
        return _sycl_dotted_device_details(normalized)

    return (None, device)


def _cuda_device_details(normalized: str) -> tuple[str | None, str]:
    parts = normalized.split(":", maxsplit=1)
    if len(parts) == 2 and parts[1]:
        return (parts[1], "NVIDIA GPU")
    return (None, normalized)


def _sycl_device_details(normalized: str) -> tuple[str, str]:
    if ":" in normalized:
        parts = normalized.split(":")
        if len(parts) >= 3:
            return (f"{parts[1]}:{parts[2]}", f"SYCL Device {parts[1]}")
        if len(parts) > 1:
            return (":".join(parts[1:]), normalized)
    ordinal = normalized[4:]
    return (ordinal or "0", "Intel SYCL GPU")


def _sycl_dotted_device_details(normalized: str) -> tuple[str | None, str]:
    parts = normalized.split(":")
    if len(parts) >= 3:
        return (f"{parts[1]}:{parts[2]}", f"SYCL Device {parts[1]}")
    if len(parts) > 1:
        return (":".join(parts[1:]), normalized)
    return (None, normalized)


def _get_lspci_output() -> str | None:
    """Run lspci and return stdout, or None on failure."""
    import subprocess

    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def _get_cpu_model() -> str | None:
    """Extract CPU model name from /proc/cpuinfo."""
    import subprocess

    try:
        result = subprocess.run(
            ["cat", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if line.startswith("model name"):
                    return "cpu:" + line.split(":", 1)[1].strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def _get_os_name() -> str | None:
    """Extract OS name from /etc/os-release."""
    import subprocess

    try:
        result = subprocess.run(
            ["cat", "/etc/os-release"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if line.startswith("NAME="):
                    return "os:" + line.split("=", 1)[1].strip().strip('"')
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def compute_machine_fingerprint() -> str | None:
    """Compute a deterministic machine fingerprint from hardware identifiers."""
    parts: list[str] = []

    gpu_output = _get_lspci_output()
    if gpu_output is not None:
        parts.append("gpu:" + gpu_output)

    cpu_model = _get_cpu_model()
    if cpu_model is not None:
        parts.append(cpu_model)

    os_name = _get_os_name()
    if os_name is not None:
        parts.append(os_name)

    if not parts:
        return None

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check_hardware_allowlist(
    fingerprint: str,
    allowlist: list[str] | None = None,
) -> str:
    """Check a machine fingerprint against a hardware allowlist."""
    if allowlist is None:
        raw = os.environ.get("LLM_RUNNER_HARDWARE_ALLOWLIST", "")
        allowlist = [f.strip() for f in raw.split(",") if f.strip()] if raw else []

    if not allowlist:
        return "invalidated"

    if fingerprint in allowlist:
        return "match"

    return "mismatch"


def assess_vram_risk(
    vram_free_gb: float,
    model_size_gb: float,
) -> VRamRecommendation:
    """Assess VRAM risk for loading a model."""
    from ...config import VRamRecommendation

    if model_size_gb <= 0:
        return VRamRecommendation.PROCEED

    _WARN_THRESHOLD: Final[float] = 1.2 / 0.85
    _PROCEED_THRESHOLD: Final[float] = 1.5

    ratio = vram_free_gb / model_size_gb

    if ratio >= _PROCEED_THRESHOLD:
        return VRamRecommendation.PROCEED
    if ratio >= _WARN_THRESHOLD:
        return VRamRecommendation.WARN
    return VRamRecommendation.CONFIRM_REQUIRED
