"""Build modal screen for the TUI.

Contains the legacy BuildPanel (unused, kept for reference) and the active
BuildModalScreen that drives the modal build workflow.
"""

from __future__ import annotations

from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, ProgressBar, Static

from llama_cli.tui.types import BuildViewState

# ---------------------------------------------------------------------------
# Legacy BuildPanel — kept for reference but no longer mounted.
# ---------------------------------------------------------------------------


class BuildPanel(Container):
    """Build panel widget — always mounted, visibility toggled via CSS."""

    view_state: reactive[BuildViewState] = reactive(BuildViewState(), init=False)

    def __init__(self) -> None:
        super().__init__(id="build-panel", classes="build-panel")
        self._title = Static("", id="build-title")
        self._message = Static("", id="build-message")
        self._progress = ProgressBar(id="build-progress", total=None, show_eta=False)
        self._result = Static("", id="build-result")
        self._retry_info = Static("", id="build-retry-info")
        self._error = Static("", id="build-error")
        self._target_prompt = Static("", id="build-target-prompt")

    def compose(self) -> ComposeResult:
        with Container(id="build-content"):
            yield self._title
            yield self._message
            yield self._progress
            yield self._result
            yield self._retry_info
            yield self._error
            yield self._target_prompt

    def watch_view_state(self, state: BuildViewState) -> None:
        """Update child widgets when view state changes."""
        # Toggle visibility
        if state.visible:
            self.add_class("-visible")
        else:
            self.remove_class("-visible")

        # Target selection state
        if state.build_request and not state.selected_backend:
            self._target_prompt.update("Select build target: [1] SYCL  [2] CUDA  [3] Both")
            self._title.update("Build")
            self._message.update("")
            self._progress.update()
            self._result.update()
            self._retry_info.update()
            self._error.update()
            return

        # In-progress state
        if state.in_progress:
            stage_label = state.stage.upper() if state.stage else "BUILD"
            self._title.update(f"Build [{stage_label}]")
            self._message.update(state.message or f"Building for {state.selected_backend}...")

            if state.progress_percent > 0:
                self._progress.update(total=100, progress=state.progress_percent)
            else:
                self._progress.update()

            if state.is_retrying and state.retries_remaining > 0:
                self._retry_info.update(
                    f"Retrying... ({state.retries_remaining} retries remaining)"
                )
                self._retry_info.remove_class("hidden")
            else:
                self._retry_info.update()
                self._retry_info.add_class("hidden")

            self._result.update()
            self._error.update()
            return

        # Success state
        if state.last_result_success is True:
            self._title.update("Build Complete")
            self._result.update(
                f"Build completed successfully! ({state.selected_backend or 'unknown'})"
            )
            if state.artifact_path:
                self._result.update(
                    f"Build completed successfully! "
                    f"({state.selected_backend or 'unknown'})\n"
                    f"  Binary: {state.artifact_path}"
                )
            self._message.update("")
            self._progress.update(total=100, progress=100)
            self._retry_info.update()
            self._retry_info.add_class("hidden")
            self._error.update()
            return

        # Failure state
        if state.last_result_success is False:
            self._title.update("Build Failed")
            self._message.update(f"Build failed for {state.selected_backend or 'unknown'} backend")
            self._error.update(state.error_message or "Unknown error")
            self._error.remove_class("hidden")
            self._progress.update()
            self._retry_info.update()
            self._retry_info.add_class("hidden")
            self._result.update()
            return

        # Default: hide everything
        self._title.update()
        self._message.update()
        self._progress.update()
        self._result.update()
        self._retry_info.update()
        self._retry_info.add_class("hidden")
        self._error.update()
        self._error.add_class("hidden")
        self._target_prompt.update()


# ---------------------------------------------------------------------------
# BuildModalScreen — active modal build workflow
# ---------------------------------------------------------------------------


class BuildModalScreen(ModalScreen[dict[str, Any] | None]):
    """Modal screen for building llama.cpp.

    Drives the full lifecycle:
    1. Target selection (SYCL / CUDA / Both)
    2. Build progress (stage-by-stage updates)
    3. Completion (success or failure)

    Returns a ``dict`` with ``"backends"`` (list of str) when the user starts
    a build, or ``None`` on cancel/dismiss.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("1", "select_sycl", "SYCL", show=False),
        Binding("2", "select_cuda", "CUDA", show=False),
        Binding("3", "select_both", "Both", show=False),
    ]

    # Reactive view state — the controller updates this from the build thread.
    view_state: reactive[BuildViewState] = reactive(BuildViewState(), init=False)

    def __init__(self) -> None:
        super().__init__()
        self._title = Static("Build llama.cpp", id="build-title", classes="build-title")
        self._target_prompt = Static(
            "Select build target:  [1] SYCL    [2] CUDA    [3] Both",
            id="build-target-prompt",
            classes="build-target-prompt",
        )
        self._message = Static("", id="build-message", classes="build-message")
        self._progress = ProgressBar(id="build-progress", total=None, show_eta=False)
        self._retry_info = Static("", id="build-retry-info", classes="build-retry-info")
        self._result = Static("", id="build-result", classes="build-result")
        self._error = Static("", id="build-error", classes="build-error")
        self._cancel_button = Button("Cancel", id="build-cancel", classes="modal-button-cancel")

    def compose(self) -> ComposeResult:
        with Container(classes="build-modal"):
            yield self._title
            yield self._target_prompt
            yield self._message
            yield self._progress
            yield self._retry_info
            yield self._result
            yield self._error
            yield self._cancel_button

    def on_mount(self) -> None:
        self._cancel_button.focus()

    # -- Actions -----------------------------------------------------------

    def action_cancel(self) -> None:
        """Cancel the build modal."""
        if self.view_state.in_progress:
            # Signal cancellation to controller if available
            controller = getattr(self, "controller", None)
            if controller is not None and hasattr(controller, "cancel_build"):
                controller.cancel_build()
        self.dismiss(None)

    def action_select_sycl(self) -> None:
        self._start_build("sycl")

    def action_select_cuda(self) -> None:
        self._start_build("cuda")

    def action_select_both(self) -> None:
        self._start_build("both")

    # -- Button handling ----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "build-cancel":
            self.action_cancel()

    # -- Public API for controller updates ----------------------------------

    def set_building(
        self, backend: str, stage: str | None = None, message: str | None = None
    ) -> None:
        """Update modal to show building state."""
        self.view_state = BuildViewState(
            visible=True,
            in_progress=True,
            selected_backend=backend,
            stage=stage,
            message=message or f"Building for {backend}...",
        )

    def set_progress(
        self,
        percent: int = 0,
        stage: str | None = None,
        message: str | None = None,
        is_retrying: bool = False,
        retries_remaining: int = 0,
    ) -> None:
        """Update progress within the build."""
        state = self.view_state
        self.view_state = BuildViewState(
            visible=True,
            in_progress=True,
            selected_backend=state.selected_backend,
            stage=stage or state.stage,
            message=message or state.message,
            progress_percent=percent,
            is_retrying=is_retrying,
            retries_remaining=retries_remaining,
        )

    def set_success(self, backend: str, artifact_path: str | None = None) -> None:
        """Update modal to show successful build."""
        self.view_state = BuildViewState(
            visible=True,
            in_progress=False,
            selected_backend=backend,
            last_result_success=True,
            artifact_path=artifact_path,
            progress_percent=100,
        )

    def set_failure(self, backend: str, error_message: str) -> None:
        """Update modal to show build failure."""
        self.view_state = BuildViewState(
            visible=True,
            in_progress=False,
            selected_backend=backend,
            last_result_success=False,
            error_message=error_message,
        )

    # -- Internal helpers ---------------------------------------------------

    def _start_build(self, backend: Literal["sycl", "cuda", "both"]) -> None:
        """Initiate the build via the controller and dismiss with result."""
        self._target_prompt.update(f"Starting build for {backend}...")
        self.dismiss({"backends": [backend]})

    def watch_view_state(self, state: BuildViewState) -> None:
        """Update child widgets when view state changes."""
        if not hasattr(self, "_title"):
            return

        # Title updates
        if state.in_progress:
            stage_label = state.stage.upper() if state.stage else "BUILD"
            self._title.update(f"Build [{stage_label}]")
        elif state.last_result_success is True:
            self._title.update("Build Complete")
        elif state.last_result_success is False:
            self._title.update("Build Failed")
        else:
            self._title.update("Build llama.cpp")

        # Target prompt — hide once backend is selected
        if state.build_request and not state.selected_backend:
            self._target_prompt.update("Select build target:  [1] SYCL    [2] CUDA    [3] Both")
        elif (
            state.in_progress
            or state.last_result_success is True
            or state.last_result_success is False
        ):
            self._target_prompt.update("")
        else:
            self._target_prompt.update("Select build target:  [1] SYCL    [2] CUDA    [3] Both")

        # Message
        if state.in_progress:
            self._message.update(state.message or f"Building for {state.selected_backend}...")
        elif state.last_result_success is True:
            self._message.update("")
        elif state.last_result_success is False:
            self._message.update(f"Build failed for {state.selected_backend or 'unknown'} backend")
        else:
            self._message.update("")

        # Progress bar
        if state.in_progress and state.progress_percent > 0:
            self._progress.update(total=100, progress=state.progress_percent)
        elif state.last_result_success is True:
            self._progress.update(total=100, progress=100)
        else:
            self._progress.update()

        # Retry info
        if state.is_retrying and state.retries_remaining > 0:
            self._retry_info.update(f"Retrying... ({state.retries_remaining} retries remaining)")
            self._retry_info.remove_class("hidden")
        else:
            self._retry_info.update()
            self._retry_info.add_class("hidden")

        # Result
        if state.last_result_success is True:
            msg = f"Build completed successfully! ({state.selected_backend or 'unknown'})"
            if state.artifact_path:
                msg = (
                    f"Build completed successfully! "
                    f"({state.selected_backend or 'unknown'})\n"
                    f"  Binary: {state.artifact_path}"
                )
            self._result.update(msg)
        else:
            self._result.update("")

        # Error
        if state.last_result_success is False:
            self._error.update(state.error_message or "Unknown error")
            self._error.remove_class("hidden")
        else:
            self._error.update()
            self._error.add_class("hidden")
