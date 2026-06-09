"""Dashboard controller — TUI business logic for slot profile management.

Handles slot-profile CRUD operations and form submissions for the
Textual dashboard.  Pure library — no I/O except sys.stderr.
"""

import json
import sys
from dataclasses import dataclass


@dataclass
class SlotProfilePayload:
    """Form payload for creating/updating a slot profile in the TUI.

    Attributes:
        profile_id: Unique profile identifier (will be normalised).
        label: Display label for the profile.
        server_bin: Path to the llama-server binary.
        model: Path to the model file.
        port: TCP port for the server.
        ctx_size: Context window size.
        ubatch_size: Ubatch size.
        n_gpu_layers: Number of GPU layers or ``"all"``.
        threads: Number of CPU threads.
        chat_template_kwargs: Optional JSON string of chat template kwargs.
    """

    profile_id: str
    label: str
    server_bin: str
    model: str
    port: int
    ctx_size: int
    ubatch_size: int
    n_gpu_layers: str | int
    threads: int
    chat_template_kwargs: str | dict[str, object] = ""


class DashboardController:
    """Controller for dashboard TUI operations.

    Handles slot-profile persistence and form submissions.
    """

    def save_slot_profile_from_form(self, payload: SlotProfilePayload) -> bool:
        """Save a slot profile from the TUI modal.

        Returns True if saved successfully, False otherwise.
        Validates the payload before saving.
        """
        profile_id = payload.profile_id.strip().lower().replace(" ", "-")
        if not profile_id:
            self._log_error("Profile ID cannot be empty")
            return False

        if not (1024 <= payload.port <= 65535):
            self._log_error("Port must be between 1024 and 65535")
            return False

        if payload.ctx_size <= 0 or payload.ubatch_size <= 0 or payload.threads <= 0:
            self._log_error("ctx_size, ubatch_size, and threads must be positive")
            return False

        if not self._validate_n_gpu_layers(payload.n_gpu_layers):
            return False

        ctk = payload.chat_template_kwargs
        if ctk and not self._validate_chat_template_kwargs(ctk):
            return False

        from .config.profiles import SlotProfileSpec
        from .slot_profile_store import save_custom_slot_profile

        profile = SlotProfileSpec(
            profile_id=profile_id,
            model=payload.model,
            alias=payload.label or profile_id,
            device="",
            port=payload.port,
            ctx_size=payload.ctx_size,
            ubatch_size=payload.ubatch_size,
            threads=payload.threads,
            n_gpu_layers=payload.n_gpu_layers
            if payload.n_gpu_layers == "all"
            else int(payload.n_gpu_layers),
            server_bin=payload.server_bin,
            chat_template_kwargs=self._stringify_chat_template_kwargs(ctk),
        )

        try:
            save_custom_slot_profile(profile)
        except ValueError as e:
            self._log_error(f"Save failed: {e}")
            return False
        return True

    @staticmethod
    def _log_error(message: str) -> None:
        sys.stderr.write(f"{message}\n")

    @classmethod
    def _validate_n_gpu_layers(cls, ngl: str | int) -> bool:
        if ngl == "all":
            return True
        try:
            ngl_int = int(ngl)
        except TypeError, ValueError:
            cls._log_error("n_gpu_layers must be an integer or 'all'")
            return False
        if ngl_int < 0:
            cls._log_error("n_gpu_layers must be >= 0 or 'all'")
            return False
        return True

    @classmethod
    def _validate_chat_template_kwargs(cls, ctk: str | dict[str, object]) -> bool:
        if not isinstance(ctk, str):
            return True
        try:
            json.loads(ctk)
        except TypeError, ValueError:
            cls._log_error("chat_template_kwargs must be valid JSON")
            return False
        return True

    @staticmethod
    def _stringify_chat_template_kwargs(ctk: str | dict[str, object]) -> str:
        if isinstance(ctk, str):
            return ctk
        if isinstance(ctk, dict):
            return json.dumps(ctk)
        return ""
