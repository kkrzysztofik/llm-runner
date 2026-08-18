"""NVIDIA GPU power-cap side effects (best-effort, per-launch)."""

import subprocess
from collections.abc import Callable

_TIMEOUT_S = 2


def cuda_ordinals(device: str) -> list[int]:
    """Return the CUDA ordinals referenced by a device string.

    Empty/auto devices default to ``[0]`` (CUDA auto-detect). Non-CUDA
    devices and unparseable input return ``[]`` (no cap applied).
    """
    stripped = device.strip().upper()
    if not stripped or stripped == "AUTO":
        return [0]
    if not stripped.startswith("CUDA"):
        return []
    rest = stripped[4:].lstrip(":")
    if not rest:
        return [0]
    ordinals: list[int] = []
    for part in rest.split(","):
        part = part.strip()
        if part.upper().startswith("CUDA"):
            part = part[4:].lstrip(":")
        if part.isdigit():
            ordinals.append(int(part))
    return ordinals


def apply_nvidia_power_limit(device: str, watts: int, warn: Callable[[str], None]) -> None:
    """Apply a power cap to every CUDA device in *device* (best-effort).

    ``watts <= 0`` disables the cap. Failures are reported through *warn*
    and never raised, so a missing driver or missing sudo permission cannot
    block a server launch.
    """
    if watts <= 0:
        return
    for ordinal in cuda_ordinals(device):
        try:
            result = subprocess.run(
                ["sudo", "-n", "nvidia-smi", "-i", str(ordinal), "-pl", str(watts)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        # PEP 758 (Python 3.14): comma-form except is legal on the py314 target
        # and is what ruff format enforces; not a Python 2 leftover.
        except OSError, subprocess.SubprocessError:
            warn(f"failed to set NVIDIA power limit {watts}W on GPU {ordinal}")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            warn(
                f"failed to set NVIDIA power limit {watts}W on GPU {ordinal}"
                + (f": {detail}" if detail else "")
            )
