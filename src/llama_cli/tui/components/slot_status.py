"""Slot status resolution helpers."""

from typing import Any

import psutil

from llama_manager import SlotState

BACKEND_LABELS: dict[str, str] = {
    "sycl": "SYCL",
    "cuda": "CUDA",
    "llama_cpp": "CPU",
}

STATUS_COLORS: dict[str, str] = {
    SlotState.RUNNING.value: "green",
    SlotState.LAUNCHING.value: "yellow",
    SlotState.DEGRADED.value: "yellow",
    SlotState.CRASHED.value: "red",
    SlotState.OFFLINE.value: "dim",
    SlotState.IDLE.value: "dim",
}


class SlotStatusResolver:
    """Resolves final slot status from tracked state and process liveness."""

    def resolve(
        self,
        alias: str,
        slot_states: dict[str, str],
        server_processes: dict[str, Any],
    ) -> str:
        state = slot_states.get(alias, SlotState.OFFLINE.value)
        status = state
        if state == SlotState.RUNNING.value:
            proc = server_processes.get(alias)
            if not proc:
                status = SlotState.CRASHED.value
            elif hasattr(proc, "poll"):
                # Process-like object: check if it has exited
                if proc.poll() is not None:
                    status = SlotState.CRASHED.value
            else:
                # PID-based object: check via psutil
                pid = getattr(proc, "pid", None)
                if not (pid and psutil.pid_exists(pid)):
                    status = SlotState.CRASHED.value
        return status
