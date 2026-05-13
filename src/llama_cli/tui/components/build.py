"""Build status panel widget for the TUI."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import ProgressBar, Static

from llama_cli.tui.types import BuildViewState


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
