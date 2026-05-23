"""Dashboard controller — TUI business logic for profile management.

Handles run-profile CRUD operations and form submissions for the
Textual dashboard.  Pure library — no I/O except sys.stderr.
"""

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class RunProfilePayload:
    """Form payload for creating/updating a run profile in the TUI.

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

    Handles run-profile persistence and form submissions.
    """

    def save_run_profile_from_form(self, payload: RunProfilePayload) -> bool:
        """Save a run profile from the TUI modal.

        Returns True if saved successfully, False otherwise.
        Validates the payload before saving.
        """
        # Validation
        profile_id = payload.profile_id.strip().lower().replace(" ", "-")
        if not profile_id:
            sys.stderr.write("Profile ID cannot be empty\n")
            return False

        # Port validation
        if not (1024 <= payload.port <= 65535):
            sys.stderr.write("Port must be between 1024 and 65535\n")
            return False

        # Size validations
        if payload.ctx_size <= 0 or payload.ubatch_size <= 0 or payload.threads <= 0:
            sys.stderr.write("ctx_size, ubatch_size, and threads must be positive\n")
            return False

        # n_gpu_layers validation
        ngl = payload.n_gpu_layers
        if ngl != "all":
            try:
                ngl_int = int(ngl)
                if ngl_int < 0:
                    sys.stderr.write("n_gpu_layers must be >= 0 or 'all'\n")
                    return False
            except ValueError, TypeError:
                sys.stderr.write("n_gpu_layers must be an integer or 'all'\n")
                return False

        # chat_template_kwargs validation (if non-empty)
        ctk = payload.chat_template_kwargs
        if ctk:
            try:
                if isinstance(ctk, str):
                    json.loads(ctk)
            except json.JSONDecodeError, TypeError:
                sys.stderr.write("chat_template_kwargs must be valid JSON\n")
                return False

        # Build and save
        from .config.profiles import RunProfileSpec
        from .run_profile_store import save_custom_run_profile

        profile = RunProfileSpec(
            profile_id=profile_id,
            model=payload.model,
            alias=payload.label or profile_id,
            device="",
            port=payload.port,
            ctx_size=payload.ctx_size,
            ubatch_size=payload.ubatch_size,
            threads=payload.threads,
            n_gpu_layers=ngl if ngl == "all" else int(ngl),
            server_bin=payload.server_bin,
            chat_template_kwargs=ctk
            if isinstance(ctk, str)
            else json.dumps(ctk)
            if isinstance(ctk, dict)
            else "",
        )

        try:
            save_custom_run_profile(profile)
            return True
        except ValueError as e:
            sys.stderr.write(f"Save failed: {e}\n")
            return False
