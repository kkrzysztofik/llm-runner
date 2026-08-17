"""Input validation functions for server configuration."""

import os

from ..common.validators import PORT_MAX, PORT_MIN, validate_port_range
from ..config import (
    ErrorCode,
    ErrorDetail,
    ServerConfig,
)


def validate_port(port: int, name: str = "port") -> ErrorDetail | None:
    """Validate port number."""
    err = validate_port_range(port)
    if err is not None:
        return ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked=f"{name}: {err}",
            how_to_fix=f"ensure {name} is an integer between {PORT_MIN} and {PORT_MAX}",
        )
    return None


def require_model(model_path: str) -> ErrorDetail | None:
    """Check if model file exists."""
    if not os.path.isfile(model_path):
        return ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_path_exists",
            why_blocked=f"model not found: {model_path}",
            how_to_fix="verify model path exists and is accessible",
        )
    return None


def require_executable(bin_path: str, name: str = "binary") -> ErrorDetail | None:
    """Check if executable exists."""
    if not os.access(bin_path, os.X_OK):
        return ErrorDetail(
            error_code=ErrorCode.PERMISSION_DENIED,
            failed_check="executable_exists",
            why_blocked=f"{name} not found or not executable: {bin_path}",
            how_to_fix="verify executable path exists and has execute permissions",
        )
    return None


def validate_server_config(cfg: ServerConfig) -> ErrorDetail | None:
    """FR-011: Validate ServerConfig for M1 launch eligibility."""
    if cfg.backend.lower() == "vllm":
        return ErrorDetail(
            error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
            failed_check="vllm_launch_eligibility",
            why_blocked="vllm is not launch-eligible in PRD M1",
            how_to_fix="change backend to 'llama_cpp' for M1",
        )
    return None


def validate_ports(
    port1: int, port2: int, name1: str = "port1", name2: str = "port2"
) -> ErrorDetail | None:
    """Validate ports are different."""
    if port1 == port2:
        return ErrorDetail(
            error_code=ErrorCode.PORT_CONFLICT,
            failed_check="port_uniqueness",
            why_blocked=f"{name1} and {name2} must be different, got: {port1}",
            how_to_fix="ensure both ports are unique values between 1 and 65535",
        )
    return None


def detect_risky_operations(cfg: ServerConfig) -> list[str]:
    """Detect potentially risky operations in server configuration."""
    risks: list[str] = []

    if cfg.port < 1024:
        risks.append("privileged_port")

    if cfg.bind_address not in ("127.0.0.1", "::1"):
        risks.append("non_loopback")

    if "warning_bypass" in cfg.risky_acknowledged:
        risks.append("warning_bypass")

    return risks
