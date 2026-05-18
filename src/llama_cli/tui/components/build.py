"""Build wizard modal screen for the TUI.

Contains the legacy BuildPanel (unused, kept for reference) and the active
BuildModalScreen that drives the multi-step build wizard workflow.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Log,
    ProgressBar,
    RadioButton,
    RadioSet,
    Static,
)

from llama_manager.build_pipeline import (
    BuildConfig,
    BuildProgress,
    BuildStatus,
    get_build_status,
)
from llama_manager.build_pipeline.models import BuildBackend

from ..types import BuildWizardResult

if TYPE_CHECKING:
    from ..textual_app import DashboardApp


# ---------------------------------------------------------------------------
# Legacy BuildPanel — kept for reference but no longer mounted.
# ---------------------------------------------------------------------------


class BuildPanel(Container):
    """Build panel widget — always mounted, visibility toggled via CSS."""

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


# ---------------------------------------------------------------------------
# BuildWizardScreen — multi-step build wizard
# ---------------------------------------------------------------------------

_BACKEND_OPTIONS: list[tuple[str, str]] = [
    ("SYCL", "sycl"),
    ("CUDA", "cuda"),
    ("Both (SYCL + CUDA)", "both"),
]


class BuildModalScreen(ModalScreen[BuildWizardResult | None]):
    """Multi-step build wizard modal screen.

    Wizard steps:
    1. Select target backend + display build/source status
    2. SYCL build options (if SYCL or Both selected)
    3. CUDA build options (if CUDA or Both selected)
    4. Building progress (live updates)
    5. Result (success/failure)
    """

    STEP_SELECT = 1
    STEP_SYCL_OPTS = 2
    STEP_CUDA_OPTS = 3
    STEP_BUILDING = 4
    STEP_RESULT = 5

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, *, last_backend: str = "sycl") -> None:
        super().__init__()
        self._last_backend = last_backend
        self._wizard_state: dict[str, Any] = {
            "step": self.STEP_SELECT,
            "selected_backend": last_backend,
            "sycl_options": None,
            "cuda_options": None,
            "progress_backend": "",
            "sycl_status": None,
            "cuda_status": None,
            "build_result_success": None,
            "build_result_artifact": None,
            "build_result_error": "",
        }

        # Widgets — created lazily per step
        self._select_backend: RadioSet | None = None
        self._status_panel: DataTable | None = None
        self._btn_next: Button | None = None
        self._btn_back: Button | None = None
        self._btn_cancel: Button | None = None
        self._btn_done: Button | None = None
        self._btn_stop: Button | None = None
        self._progress_bar: ProgressBar | None = None
        self._build_message: Static | None = None
        self._build_log: Log | None = None
        self._retry_info: Static | None = None
        self._result_panel: Static | None = None

        # Form inputs per backend — keyed by field name (not widget.id)
        self._sycl_inputs: dict[str, Input | Checkbox] = {}
        self._cuda_inputs: dict[str, Input | Checkbox] = {}

        # Lock for thread-safe progress updates
        self._progress_lock = threading.Lock()

    # -- Helpers -------------------------------------------------------------

    @property
    def _dashboard_app(self) -> DashboardApp:
        """Return the DashboardApp instance."""
        return self.app  # type: ignore[return-value]

    def _get_inputs(self, backend: str) -> dict[str, Input | Checkbox]:
        if backend == "sycl":
            return self._sycl_inputs
        return self._cuda_inputs

    # -- Composition --------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Yield a single placeholder container that holds each wizard step.

        _render_step() replaces the placeholder's children (not the placeholder
        itself) so we never call self.mount() on the Screen — avoiding the
        MountError that occurs when self.mount() is called before the screen
        is fully attached.
        """
        yield Container(id="build-wizard-placeholder", classes="build-modal")

    # -- Lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        """Fetch build status for both backends and schedule step 1 render."""
        config = self._dashboard_app.controller.config
        self._wizard_state["sycl_status"] = get_build_status(BuildBackend.SYCL, config)
        self._wizard_state["cuda_status"] = get_build_status(BuildBackend.CUDA, config)
        # Defer rendering until the screen is fully attached
        self.call_later(self._render_step)

    def _render_step(self) -> None:
        """Clear and re-compose the current step inside the placeholder."""
        placeholder = self.query_one("#build-wizard-placeholder", Container)
        placeholder.remove_children()
        step = self._wizard_state["step"]
        if step == self.STEP_SELECT:
            self._compose_step_select(placeholder)
            self.call_after_refresh(self._focus_step_select)
        elif step == self.STEP_SYCL_OPTS:
            self._compose_step_options(placeholder, "sycl")
            self.call_after_refresh(self._focus_step_options)
        elif step == self.STEP_CUDA_OPTS:
            self._compose_step_options(placeholder, "cuda")
            self.call_after_refresh(self._focus_step_options)
        elif step == self.STEP_BUILDING:
            self._compose_step_building(placeholder)
        elif step == self.STEP_RESULT:
            self._compose_step_result(placeholder)

    def _focus_step_select(self) -> None:
        """Set focus on the backend RadioSet."""
        if self._select_backend:
            self.set_focus(self._select_backend)

    def _focus_step_options(self) -> None:
        """Set focus on the Next/Start Build button."""
        if self._btn_next:
            self.set_focus(self._btn_next)

    def _clear_mounted(self) -> None:
        """Remove all children from the placeholder (for re-compose on back navigation)."""
        placeholder = self.query_one("#build-wizard-placeholder", Container)
        placeholder.remove_children()

    # -- Step 1: Select target + status ------------------------------------

    def _compose_step_select(self, parent: Container) -> None:
        title = Static("Build Wizard", id="build-title", classes="build-title")
        label = Static("Select build target:", classes="build-step-label")

        sel = RadioSet(
            *[
                RadioButton(label, value=(value == self._wizard_state["selected_backend"]))
                for label, value in _BACKEND_OPTIONS
            ],
            id="backend-select",
        )
        self._select_backend = sel

        status = self._build_status_table()
        self._status_panel = status

        self._btn_cancel = Button("Cancel", id="build-cancel", classes="modal-button-cancel")
        self._btn_next = Button("Next", id="build-next", classes="modal-button-success")
        actions = Horizontal(self._btn_cancel, self._btn_next, classes="modal-actions")

        parent.mount(
            Container(title, label, sel, status, actions, classes="build-wizard-step-select")
        )

    def _build_status_table(self) -> DataTable:
        """Build a Textual DataTable showing artifact/source/remote status per backend."""
        table: DataTable[str] = DataTable(
            id="build-status",
            show_cursor=False,
            cursor_type="none",
        )
        table.can_focus = False
        table.add_columns("Backend", "Artifact", "Source", "Remote")

        for backend_str, status in [
            ("SYCL", self._wizard_state.get("sycl_status")),
            ("CUDA", self._wizard_state.get("cuda_status")),
        ]:
            if status is None:
                artifact_txt = source_txt = remote_txt = "—"
            else:
                artifact_txt = _artifact_status_text(status)
                source_txt = _source_status_text(status)
                remote_txt = _remote_status_text(status)

            table.add_row(backend_str, artifact_txt, source_txt, remote_txt)
        return table

    # -- Step 2/3: Build options -------------------------------------------

    def _compose_step_options(self, parent: Container, backend: str) -> None:
        config = self._dashboard_app.controller.config
        title_label = "SYCL" if backend == "sycl" else "CUDA"

        source_dir = Path(config.llama_cpp_root)
        build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
        output_dir = config.builds_dir / backend

        inputs = self._get_inputs(backend)

        title = Static(
            f"Build Wizard — {title_label} Options",
            id="build-title",
            classes="build-title",
        )

        # Build all form widgets up-front
        git_branch_input = Input(
            value=getattr(config, "build_git_branch", "master"),
            classes="build-option-input",
        )
        git_commit_input = Input(value="", classes="build-option-input")
        jobs_input = Input(value="", classes="build-option-input")
        retry_attempts_input = Input(
            value=str(config.build_retry_attempts), classes="build-option-input"
        )
        retry_delay_input = Input(value=str(config.build_retry_delay), classes="build-option-input")
        shallow_cb = Checkbox(
            f"Shallow clone (default: {getattr(config, 'build_shallow_clone', True)})",
            classes="build-option-checkbox",
            value=getattr(config, "build_shallow_clone", True),
        )
        update_cb = Checkbox("Update sources", classes="build-option-checkbox", value=True)
        timeout_input = Input(value="3600", classes="build-option-input")

        inputs["git_branch"] = git_branch_input
        inputs["git_commit"] = git_commit_input
        inputs["jobs"] = jobs_input
        inputs["retry_attempts"] = retry_attempts_input
        inputs["retry_delay"] = retry_delay_input
        inputs["shallow_clone"] = shallow_cb
        inputs["update_sources"] = update_cb
        inputs["build_timeout_seconds"] = timeout_input

        form = Vertical(
            Static("  Git branch:", classes="build-option-label"),
            git_branch_input,
            Static("  Git commit (optional):", classes="build-option-label"),
            git_commit_input,
            Static("  Jobs (empty = auto):", classes="build-option-label"),
            jobs_input,
            Static("  Retry attempts:", classes="build-option-label"),
            retry_attempts_input,
            Static("  Retry delay (seconds):", classes="build-option-label"),
            retry_delay_input,
            shallow_cb,
            update_cb,
            Static("  Build timeout (seconds):", classes="build-option-label"),
            timeout_input,
            classes="build-options-form",
        )

        # Action buttons
        self._btn_cancel = Button("Cancel", id="build-cancel", classes="modal-button-cancel")
        selected = self._wizard_state["selected_backend"]
        actions_children: list[Button] = [self._btn_cancel]
        if selected == "both":
            self._btn_back = Button("Back", id="build-back", classes="modal-button-cancel")
            actions_children.append(self._btn_back)
        self._btn_next = Button("Start Build", id="build-next", classes="modal-button-success")
        actions_children.append(self._btn_next)
        actions = Horizontal(*actions_children, classes="modal-actions")

        parent.mount(
            Container(
                title,
                form,
                Static("", classes="build-step-label"),
                Static(f"  Source: {source_dir}", classes="build-derived-path"),
                Static(f"  Build:  {build_dir}", classes="build-derived-path"),
                Static(f"  Output: {output_dir}", classes="build-derived-path"),
                actions,
                classes="build-wizard-step-options",
            )
        )

    def _collect_options(self, backend: str) -> BuildConfig | None:
        """Collect form values into a BuildConfig override."""
        inputs = self._get_inputs(backend)
        if not inputs:
            return None

        def _str_val(key: str) -> str | None:
            widget = inputs.get(key)
            if isinstance(widget, Input):
                v = widget.value.strip()
                return v or None
            return None

        def _int_val(key: str) -> int | None:
            raw = _str_val(key)
            if raw is None:
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        def _bool_val(key: str) -> bool | None:
            widget = inputs.get(key)
            if isinstance(widget, Checkbox):
                return widget.value
            return None

        git_branch = _str_val("git_branch")
        git_commit = _str_val("git_commit")
        jobs = _int_val("jobs")
        retry_attempts = _int_val("retry_attempts")
        retry_delay = _int_val("retry_delay")
        shallow_clone = _bool_val("shallow_clone")
        update_sources = _bool_val("update_sources")
        build_timeout = _int_val("build_timeout_seconds")

        # Only create override if at least one field is set
        if all(
            v is None
            for v in [
                git_branch,
                git_commit,
                jobs,
                retry_attempts,
                retry_delay,
                shallow_clone,
                update_sources,
                build_timeout,
            ]
        ):
            return None

        return BuildConfig(
            backend=BuildBackend.SYCL if backend == "sycl" else BuildBackend.CUDA,
            source_dir=Path("/dev/null"),
            build_dir=Path("/dev/null"),
            output_dir=Path("/dev/null"),
            git_remote_url="",
            git_branch=git_branch or "master",
            git_commit=git_commit,
            jobs=jobs,
            retry_attempts=retry_attempts or 3,
            retry_delay=float(retry_delay) if retry_delay is not None else 5.0,
            shallow_clone=shallow_clone if shallow_clone is not None else True,
            update_sources=update_sources if update_sources is not None else True,
            build_timeout_seconds=build_timeout or 3600,
        )

    # -- Step 4: Building --------------------------------------------------

    def _compose_step_building(self, parent: Container) -> None:
        backend = self._wizard_state.get("progress_backend", "unknown")

        title = Static("Build Wizard", id="build-title", classes="build-title")
        msg = Static(f"Building {backend.upper()}...", classes="build-message")
        self._build_message = msg

        pb = ProgressBar(id="build-progress", total=100, show_eta=False)
        self._progress_bar = pb

        if self._build_log is not None:
            log = self._build_log
        else:
            log = Log(id="build-log", highlight=False)
            log.can_focus = False
            self._build_log = log

        retry = Static("", id="build-retry-info", classes="build-retry-info")
        self._retry_info = retry

        self._btn_stop = Button("Stop", id="build-stop", classes="modal-button-warning")
        actions = Horizontal(self._btn_stop, classes="modal-actions")

        parent.mount(
            Container(title, msg, pb, log, retry, actions, classes="build-wizard-step-building")
        )

    # -- Step 5: Result ----------------------------------------------------

    def _compose_step_result(self, parent: Container) -> None:
        success = self._wizard_state.get("build_result_success")

        title = Static("Build Wizard", id="build-title", classes="build-title")
        panel = Static(self._render_result_text(success), classes="build-result-panel")
        self._result_panel = panel

        self._btn_done = Button("Done", id="build-done", classes="modal-button-success")
        actions = Horizontal(self._btn_done, classes="modal-actions")

        parent.mount(Container(title, panel, actions, classes="build-wizard-step-result"))

    def _render_result_text(self, success: bool | None) -> str:
        if success is True:
            artifact = self._wizard_state.get("build_result_artifact")
            lines: list[str] = ["[green]Build completed successfully![/green]"]
            if artifact:
                lines.append(f"  Binary: {artifact}")
            return "\n".join(lines)
        error = self._wizard_state.get("build_result_error", "Unknown error")
        return f"[red]Build failed:[/red]\n  {error}"

    # -- Public API for controller progress updates ------------------------

    def update_progress(self, progress: BuildProgress) -> None:
        """Thread-safe progress update from the build pipeline."""
        with self._progress_lock:
            backend = self._wizard_state.get("progress_backend", "")

            # Ensure we're on the building step
            if self._wizard_state["step"] != self.STEP_BUILDING:
                self._wizard_state["step"] = self.STEP_BUILDING
                self.call_later(self._render_step)
                return

            # Stream live compiler output lines
            if progress.output_line is not None and self._build_log:
                self._build_log.write_line(progress.output_line)
                return

            msg_text = f"Building {backend.upper()}... [{progress.stage}]"
            if self._build_message:
                self._build_message.update(msg_text)

            if self._build_log:
                status_tag = {"success": "OK", "failed": "ERR", "retrying": "RTY"}.get(
                    progress.status, progress.status.upper()[:3]
                )
                self._build_log.write_line(f"[{status_tag}] {progress.stage}: {progress.message}")

            if self._progress_bar:
                pct = int(progress.progress_percent)
                self._progress_bar.update(total=100, progress=max(0, min(100, pct)))

            if self._retry_info:
                if progress.is_retrying and progress.retries_remaining is not None:
                    self._retry_info.update(
                        f"Retrying... ({progress.retries_remaining} retries remaining)"
                    )
                    self._retry_info.remove_class("hidden")
                else:
                    self._retry_info.update("")
                    self._retry_info.add_class("hidden")

    def set_building_backend(self, backend: str) -> None:
        """Mark the start of building for a specific backend."""
        with self._progress_lock:
            self._wizard_state["progress_backend"] = backend
            self._wizard_state["step"] = self.STEP_BUILDING
            self.call_later(self._render_step)

    def set_build_result(
        self, success: bool, artifact_path: str | None = None, error_message: str = ""
    ) -> None:
        """Set the final build result."""
        with self._progress_lock:
            self._wizard_state["build_result_success"] = success
            self._wizard_state["build_result_artifact"] = artifact_path
            self._wizard_state["build_result_error"] = error_message
            self._wizard_state["step"] = self.STEP_RESULT
            self.call_later(self._render_step)

    # -- Actions ------------------------------------------------------------

    def action_cancel(self) -> None:
        """Cancel the wizard."""
        self.dismiss(None)

    # -- Button handling ----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "build-cancel":
            self.action_cancel()
            return

        if bid == "build-stop":
            self.action_cancel()
            return

        if bid == "build-done":
            self._dismiss_with_result()
            return

        if bid == "build-back":
            self._wizard_state["step"] = self.STEP_SELECT
            self._render_step()
            return

        if bid == "build-next":
            self._handle_next()
            return

    def _handle_next(self) -> None:
        """Handle Next button press — navigate or start build."""
        step = self._wizard_state["step"]
        selected = self._wizard_state["selected_backend"]

        if step == self.STEP_SELECT:
            if selected == "sycl":
                self._wizard_state["step"] = self.STEP_SYCL_OPTS
            elif selected == "cuda":
                self._wizard_state["step"] = self.STEP_CUDA_OPTS
            elif selected == "both":
                self._wizard_state["step"] = self.STEP_SYCL_OPTS
            self._render_step()
            return

        if step == self.STEP_SYCL_OPTS:
            self._wizard_state["sycl_options"] = self._collect_options("sycl")
            if selected == "both":
                self._wizard_state["step"] = self.STEP_CUDA_OPTS
                self._render_step()
            else:
                self._start_build_from_wizard()
            return

        if step == self.STEP_CUDA_OPTS:
            self._wizard_state["cuda_options"] = self._collect_options("cuda")
            self._start_build_from_wizard()
            return

    def _start_build_from_wizard(self) -> None:
        """Store options in model and delegate build to controller.

        The modal stays open showing progress (step 4) while the controller
        runs the build in a background thread. When the build completes, the
        controller calls back to set the result and dismiss.
        """
        selected = self._wizard_state["selected_backend"]
        backends = ["sycl", "cuda"] if selected == "both" else [selected]

        sycl_opts = self._collect_options("sycl") if selected in ("sycl", "both") else None
        cuda_opts = self._collect_options("cuda") if selected in ("cuda", "both") else None

        # Store options in the model for the controller to pick up
        self._dashboard_app.controller.model.build_selected_backends_options = {
            "sycl": sycl_opts,
            "cuda": cuda_opts,
        }

        # Transition to building step
        self._wizard_state["step"] = self.STEP_BUILDING
        self._wizard_state["progress_backend"] = backends[0]
        self._render_step()

        # Delegate to controller — it will call back via update_progress / set_build_result
        self._dashboard_app.controller.handle_build_with_wizard(backends, self)

    def _dismiss_with_result(self) -> None:
        """Dismiss after result step."""
        success = self._wizard_state.get("build_result_success")
        if success is True:
            selected = self._wizard_state["selected_backend"]
            backends = ["sycl", "cuda"] if selected == "both" else [selected]
            result = BuildWizardResult(
                backends=backends,
                options=self._dashboard_app.controller.model.build_selected_backends_options,
            )
            self.dismiss(result)
        else:
            self.dismiss(None)

    # -- Backend selection via RadioSet ----------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Sync wizard state when the user picks a different backend."""
        if self._wizard_state["step"] != self.STEP_SELECT:
            return
        self._wizard_state["selected_backend"] = _BACKEND_OPTIONS[event.index][1]


# -- Status text helpers (module-level for staticmethod compatibility) -------


def _artifact_status_text(status: BuildStatus) -> str:
    if not status.artifact_exists:
        return "No artifact"
    a = status.artifact
    if a is None:
        return "Artifact (parse error)"
    sha = a.git_commit_sha[:8] if a.git_commit_sha else "unknown"
    ver = ""
    if status.binary_version_output:
        ver = f" v:{status.binary_version_output[:30]}"
    return f"[green]{sha}[/]{ver}"


def _source_status_text(status: BuildStatus) -> str:
    if not status.source_exists:
        return "Not cloned"
    if not status.source_is_repo:
        return "Dir (not git)"
    parts: list[str] = []
    if status.source_branch:
        parts.append(status.source_branch)
    if status.source_head_sha:
        parts.append(status.source_head_sha[:8])
    return " / ".join(parts) if parts else "—"


def _remote_status_text(status: BuildStatus) -> str:
    branch = status.configured_branch
    if status.remote_branch_sha:
        return f"{branch} @{status.remote_branch_sha[:8]}"
    return f"{branch} (unreachable)"
