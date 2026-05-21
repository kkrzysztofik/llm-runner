"""Slot status presentation constants."""

# pragma: no cover — pure constant data, imported by slot_state.py
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
