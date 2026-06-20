"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, cast

logger = logging.getLogger(__name__)

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches
from textual.widgets import Footer, Log, Static

from llama_manager.slot_stats import load_profile_stats

if TYPE_CHECKING:
    from .controller import DashboardController


from .components.about_modal import AboutModal
from .components.build import BuildModalScreen
from .components.config_modal import ConfigModal, ConfigPayload
from .components.gpu_stats import GPUStatsPanel
from .components.modal import AddSlotModal, RemoveSlotModal
from .components.profile_stats_screen import ProfileStatsScreen
from .components.server_column import ServerColumnPanel
from .components.server_log import ServerLogPanel
from .components.slot_profile_modal import SlotProfileModal, SlotProfilePayload
from .components.system_health import (
    CPUUsageWidget,
    MemorySwapWidget,
    SystemInfoWidget,
)
from .components.system_status import SystemStatusWidget
from .types import BuildWizardResult, MemoryUsageSnapshot, ServerColumnState, SystemInfoSnapshot

_CONTENT_CONTAINER_ID = "#content"

# ---------------------------------------------------------------------------
# Extracted pure helper: profile options caching
# ---------------------------------------------------------------------------


def _split_log_update(
    previous: tuple[str, ...],
    current: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    """Return (reload, lines_to_write) for a Textual Log widget."""
    if current == previous:
        return False, ()
    if len(current) >= len(previous) and current[: len(previous)] == previous:
        return False, current[len(previous) :]
    return True, current


def _profile_options_cached(
    view_model: object,
    config: object,
    cache: list[tuple[str, str]] | None,
    cache_config_id: int | None,
) -> tuple[list[tuple[str, str]], int | None]:
    """Return (options, config_id) with caching logic.

    Extracted from ``DashboardApp._build_profile_options`` for testability.
    """
    config_id = id(config)
    if cache is not None and cache_config_id == config_id:
        return cache, config_id
    options = view_model.profile_options(config)  # type: ignore[union-attr]
    return options, config_id


_RISK_HIDDEN_ACTIONS = frozenset(
    {
        "refresh_dashboard",
        "add_slot",
        "remove_slot",
        "build",
        "open_config",
        "about",
        "profile_stats",
    },
)
_NORMAL_HIDDEN_ACTIONS = frozenset({"confirm", "reject"})
_REMOVE_SLOT_TITLE = "Remove Slot"


class DashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    CSS_PATH = [
        "textual_app.tcss",
        "system_status.tcss",
        "dashboard_panels.tcss",
        "modals.tcss",
    ]
    BINDINGS = [
        Binding("q", "quit_dashboard", "Quit", priority=True),
        Binding(
            "ctrl+c",
            "cancel_pending_prompt",
            "Cancel",
            key_display="^C",
        ),
        Binding("escape", "cancel_pending_prompt", "Cancel", show=False),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("b", "build", "Build"),
        Binding("a", "add_slot", "Add Slot"),
        Binding("d", "remove_slot", _REMOVE_SLOT_TITLE),
        Binding("c", "open_config", "Config"),
        Binding("p", "manage_profiles", "Profiles"),
        Binding("s", "profile_stats", "Stats"),
        Binding("h", "about", "About"),
        Binding("y", "confirm", "Confirm"),
        Binding("n", "reject", "Abort"),
    ]

    def __init__(self, controller: DashboardController) -> None:
        super().__init__()
        self.controller = controller
        self.view_model = controller.view_model
        self._last_notified_status_ts: float = 0.0
        self._active_notice_toasts: set[str] = set()
        self._profile_options_cache: list[tuple[str, str]] | None = None
        self._profile_cache_config_id: int | None = None
        self._last_model_index_notice_scanned = 0
        self._gpu_stats_refresh_active = False
        self._system_health_refresh_active = False
        self._slot_stats_refresh_active = False
        self._slot_operation_active = False
        self._pending_remove_slot_alias: str | None = None
        self.last_build_backend: str = "sycl"

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield SystemStatusWidget(self.view_model)
            with Container(id="content"):
                for i in range(self.view_model.server_column_count()):
                    yield ServerLogPanel(i, self.view_model)
        yield Footer(show_command_palette=False)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Control which bindings appear in the footer for the current mode."""
        state = self.view_model.command_menu()

        if state.build_request:
            return action == "cancel_pending_prompt"

        if state.risk_prompt is not None:
            if action in _RISK_HIDDEN_ACTIONS:
                return False
            return not (action == "quit_dashboard" and state.risk_prompt.kind == "vram")

        return action not in _NORMAL_HIDDEN_ACTIONS

    def on_mount(self) -> None:
        interval_s = max(0.1, self.controller.config.tui_refresh_interval_ms / 1000)
        logger.info(
            "DashboardApp.on_mount: initial refresh start interval_s=%.3f configs=%r",
            interval_s,
            [c.alias for c in self.controller.configs],
        )
        self.refresh_dashboard()
        logger.info("DashboardApp.on_mount: initial refresh complete")
        self.set_interval(interval_s, self.refresh_dashboard)
        self.set_interval(max(1.0, interval_s), self._schedule_gpu_stats_refresh)
        self.set_interval(max(1.0, interval_s), self._schedule_system_health_refresh)
        self.set_interval(1.0, self._schedule_slot_stats_refresh)
        self._schedule_gpu_stats_refresh()
        self._schedule_system_health_refresh()
        self._schedule_slot_stats_refresh()
        self._index_models()

    def _schedule_gpu_stats_refresh(self) -> None:
        """Start one background GPU stats refresh if no refresh is already running."""
        if self._gpu_stats_refresh_active:
            logger.debug("_schedule_gpu_stats_refresh: skipped, refresh already active")
            return
        self._gpu_stats_refresh_active = True
        self._refresh_gpu_stats_worker()

    @work(thread=True)
    def _refresh_gpu_stats_worker(self) -> None:
        """Refresh GPU stats off the render thread."""
        start = time.perf_counter()
        stats = list(self.controller.model.gpu_stats)
        snapshot_by_alias: dict[str, dict[str, Any]] = {}
        logger.debug("_refresh_gpu_stats_worker: start count=%d", len(stats))
        try:
            aliases = [cfg.alias for cfg in self.controller.model.configs]
            for index, gpu in enumerate(stats):
                gpu_start = time.perf_counter()
                gpu.update()
                if index < len(aliases):
                    snapshot_by_alias[aliases[index]] = gpu.get_cached_stats_snapshot()
                logger.debug(
                    "_refresh_gpu_stats_worker: updated index=%d device_index=%s duration_ms=%.1f",
                    index,
                    getattr(gpu, "device_index", "unknown"),
                    (time.perf_counter() - gpu_start) * 1000,
                )
        except Exception:
            logger.exception("_refresh_gpu_stats_worker: unhandled exception")
        finally:
            logger.debug(
                "_refresh_gpu_stats_worker: complete count=%d duration_ms=%.1f",
                len(stats),
                (time.perf_counter() - start) * 1000,
            )
            self.call_from_thread(
                self.controller.model.apply_gpu_stats_snapshot,
                snapshot_by_alias,
            )
            self.call_from_thread(self._mark_gpu_stats_refresh_complete)

    def _mark_gpu_stats_refresh_complete(self) -> None:
        self._gpu_stats_refresh_active = False

    def _schedule_system_health_refresh(self) -> None:
        """Start one background system-health refresh if none is running."""
        if self._system_health_refresh_active:
            logger.debug("_schedule_system_health_refresh: skipped, refresh already active")
            return
        self._system_health_refresh_active = True
        self._refresh_system_health_worker()

    @work(thread=True)
    def _refresh_system_health_worker(self) -> None:
        """Collect system-health stats off the render thread."""
        try:
            snapshot = self.controller.model.collect_system_health_snapshot()
            self.call_from_thread(self._apply_system_health_snapshot, *snapshot)
        except Exception:
            logger.exception("_refresh_system_health_worker: unhandled exception")
        finally:
            self.call_from_thread(self._mark_system_health_refresh_complete)

    def _apply_system_health_snapshot(
        self,
        cpu: list[float],
        memory_rows: list[MemoryUsageSnapshot],
        system_info: SystemInfoSnapshot,
    ) -> None:
        """Apply cached system-health data and refresh only those widgets."""
        self.controller.model.apply_system_health_snapshot(cpu, memory_rows, system_info)
        for widget_type in (CPUUsageWidget, MemorySwapWidget, SystemInfoWidget):
            with contextlib.suppress(NoMatches):
                self.query_one(widget_type).refresh(recompose=True)

    def _mark_system_health_refresh_complete(self) -> None:
        self._system_health_refresh_active = False

    def _schedule_slot_stats_refresh(self) -> None:
        """Start one background slot stats refresh if no refresh is already running."""
        if self._slot_stats_refresh_active:
            logger.debug("_schedule_slot_stats_refresh: skipped, refresh already active")
            return
        self._slot_stats_refresh_active = True
        self._refresh_slot_stats_worker()

    @work(thread=True)
    def _refresh_slot_stats_worker(self) -> None:
        """Refresh slot stats off the render thread."""
        try:
            self.controller.refresh_slot_stats()
        except Exception:
            logger.exception("_refresh_slot_stats_worker: unhandled exception")
        finally:
            self.call_from_thread(self._mark_slot_stats_refresh_complete)
            self.call_from_thread(self.refresh_dashboard)

    def _mark_slot_stats_refresh_complete(self) -> None:
        self._slot_stats_refresh_active = False

    def _index_models(self) -> None:
        """Refresh model index in a controller-owned background thread."""
        started = self.controller.refresh_model_index_async(
            progress_callback=self._index_models_progress_from_thread,
            complete_callback=self._index_models_complete_from_thread,
        )
        logger.info("_index_models: background refresh started=%s", started)
        if started:
            self._last_model_index_notice_scanned = 0
            self.notify("Indexing models...", title="Models")

    def _index_models_progress_from_thread(
        self,
        _entries: object,
        scanned: int,
        total: int,
        errors: int,
    ) -> None:
        """Forward model index progress from the controller worker to the UI thread."""
        self.call_from_thread(self._handle_model_index_progress, scanned, total, errors)

    def _index_models_complete_from_thread(
        self,
        _entries: object,
        total: int,
        errors: int,
    ) -> None:
        """Forward model index completion from the controller worker to the UI thread."""
        self.call_from_thread(self._handle_model_index_complete, total, errors)

    def _handle_model_index_progress(self, scanned: int, total: int, errors: int) -> None:
        """Show sparse progress notifications while model indexing continues."""
        if total <= 0:
            return
        notify_every = max(1, total // 5)
        if scanned < total and scanned - self._last_model_index_notice_scanned < notify_every:
            return
        self._last_model_index_notice_scanned = scanned
        message = f"Indexed {scanned}/{total} models"
        if errors:
            message += f" ({errors} errors)"
        self.notify(message, title="Models", severity="warning" if errors else "information")

    def _handle_model_index_complete(self, total: int, errors: int) -> None:
        """Show final model index status."""
        message = f"Indexing complete: {total} models found"
        if errors:
            message += f" ({errors} errors)"
        self.notify(message, title="Models", severity="warning" if errors else "information")

    def action_quit_dashboard(self) -> None:
        self.controller.request_quit()
        if not self.controller.running:
            self.exit()

    def action_interrupt_dashboard(self) -> None:
        self.controller.interrupt()
        if not self.controller.running:
            self.exit()

    def action_refresh_dashboard(self) -> None:
        self.controller.refresh_display()
        self.refresh_dashboard()

    def action_add_slot(self) -> None:
        self.push_screen(
            AddSlotModal(profile_options=self._build_profile_options()),
            self._handle_add_slot_modal_result,
        )

    def action_remove_slot(self) -> None:
        """Open the live-slot removal flow."""
        slot_options = [
            (f"{cfg.alias} ({cfg.device}:{cfg.port})", cfg.alias) for cfg in self.controller.configs
        ]
        if not slot_options:
            self.notify("No slots configured to remove", title="Slot", severity="warning")
            return
        self.push_screen(
            RemoveSlotModal(slot_options=slot_options),
            self._handle_remove_slot_modal_result,
        )

    def action_open_config(self) -> None:
        self.push_screen(
            ConfigModal(config=self.controller.config),
            self._handle_config_modal_result,
        )

    def action_about(self) -> None:
        self.push_screen(AboutModal())

    def action_manage_profiles(self) -> None:
        """Open the profiles management screen."""
        from .components.profiles_screen import ProfilesScreen

        profiles = self.controller.list_slot_profiles()
        in_use_ids = {
            spec.profile_id
            for spec, _ in profiles
            if self.controller.is_profile_in_use(spec.profile_id)
        }
        model_index = self.controller.load_model_index()

        self.push_screen(
            ProfilesScreen(profiles=profiles, in_use_ids=in_use_ids, model_index=model_index),
            self._handle_profiles_screen_result,
        )

    def action_profile_stats(self) -> None:
        """Open the read-only profile aggregate stats screen."""
        self.push_screen(
            ProfileStatsScreen(
                stats_by_profile=load_profile_stats(),
                profiles=self.controller.list_slot_profiles(),
            )
        )

    def _handle_profiles_screen_result(self, result: dict | None) -> None:
        """Handle result from the ProfilesScreen."""
        if result is None:
            return

        action = result.get("action")

        if action == "add":
            model_index = self.controller.load_model_index()
            self.push_screen(
                SlotProfileModal(model_index=model_index, config=self.controller.config),
                self._handle_profile_modal_result,
            )

        elif action == "edit":
            profile_id = result.get("profile_id", "")
            if not profile_id:
                return

            from llama_manager.config.builder import create_tui_profile_registry
            from llama_manager.slot_profile_store import custom_slot_profile_exists

            registry = create_tui_profile_registry(self.controller.config)
            try:
                spec = registry.get_profile(profile_id)
            except KeyError, ValueError:
                self.notify(f"Profile '{profile_id}' not found", severity="error")
                return

            source = "custom" if custom_slot_profile_exists(profile_id) else "builtin"

            model_index = self.controller.load_model_index()
            self.push_screen(
                SlotProfileModal(
                    profile=spec,
                    edit_source=source,
                    model_index=model_index,
                    config=self.controller.config,
                ),
                self._handle_edit_modal_result,
            )

        elif action == "delete":
            profile_id = result.get("profile_id", "")
            if not profile_id:
                return

            success = self.controller.delete_slot_profile(profile_id)
            if success:
                self.notify(f"Profile '{profile_id}' deleted", severity="information")
                self._profile_options_cache = None
                self._profile_cache_config_id = None
            self.refresh_dashboard()

    def _handle_edit_modal_result(self, result: SlotProfilePayload | None) -> None:
        """Handle result from the edit profile modal."""
        if result is None:
            return

        original_id = result.original_profile_id or result.profile_id
        saved = self.controller.update_slot_profile(original_id, result)
        if not saved:
            self.notify("Failed to update profile", severity="error")
            return

        self.notify(f"Profile '{result.profile_id}' updated", severity="information")

        # Invalidate profile options cache
        self._profile_options_cache = None
        self._profile_cache_config_id = None

        if result.save_and_add_slot:
            slot_form: dict[str, str] = {
                "profile": result.profile_id,
                "port": "",
            }
            self._start_add_slot(slot_form, "Slot added for profile")
            return

        self.refresh_dashboard()

    def action_create_profile(self) -> None:
        """Open the slot profile creation modal."""
        self.push_screen(
            SlotProfileModal(config=self.controller.config),
            self._handle_profile_modal_result,
        )

    def _handle_config_modal_result(self, result: ConfigPayload | None) -> None:
        if result is not None:
            if result.clean_cache:
                from .components.confirm_modal import ConfirmModal

                self.push_screen(
                    ConfirmModal(
                        title="Clean Model Cache",
                        message="Delete the cached model index? Models will be re-scanned.",
                    ),
                    self._handle_cache_clean_confirm,
                )
                return

            self.controller.save_config(result)
        self.refresh_dashboard()

    def _handle_cache_clean_confirm(self, confirmed: bool | None) -> None:
        """Handle confirmation result from the cache clean dialog."""
        if confirmed:
            success, message = self.controller.clean_model_cache()
            if success:
                self.notify(message, title="Models", severity="information")
                self._index_models()
            else:
                self.notify(message, title="Models", severity="error")
        else:
            self.notify("Cancelled", title="Models", severity="information")

    def _handle_profile_modal_result(self, result: SlotProfilePayload | None) -> None:
        if result is None:
            return

        saved = self.controller.save_slot_profile_from_form(result)
        if not saved:
            self.notify("Failed to save profile", severity="error")
            return

        self.notify(f"Profile '{result.profile_id}' saved", severity="information")

        # Invalidate profile options cache
        self._profile_options_cache = None
        self._profile_cache_config_id = None

        if result.save_and_add_slot:
            slot_form: dict[str, str] = {
                "profile": result.profile_id,
                "port": "",
            }
            self._start_add_slot(slot_form, "Slot added for profile")
            return

        self.refresh_dashboard()

    def _begin_slot_operation(self, action: str) -> bool:
        """Reserve the slot lifecycle worker lane."""
        if self._slot_operation_active:
            self.notify(
                f"Slot operation already running; cannot {action}.",
                title="Slot",
                severity="warning",
            )
            return False
        self._slot_operation_active = True
        return True

    def _end_slot_operation(self) -> None:
        self._slot_operation_active = False

    def _start_add_slot(self, slot_form: dict[str, str], notify_message: str = "") -> None:
        """Start an add-slot operation when no slot operation is active."""
        if not self._begin_slot_operation("add slot"):
            return
        self.notify("Adding slot…", title="Slot", severity="information")
        self._run_add_slot(slot_form, notify_message)

    def _execute_slot_launch(
        self,
        new_cfg: Any,
        profile_id: str,
        plan: Any,
        messages: list[str],
    ) -> tuple[bool, bool, list[str]]:
        """Run the full slot launch pipeline after form validation.

        Returns (success, layout_changed, updated_messages).
        """
        layout_changed = False
        if not plan.success:
            return False, False, messages
        if plan.old_alias is not None and not self.controller.server_manager.shutdown_slot(
            plan.old_alias
        ):
            messages.append(f"Unable to replace '{plan.old_alias}': shutdown verification failed")
            return False, False, messages

        stage = self.call_from_thread(
            self.controller.stage_async_slot_launch,
            new_cfg,
            plan.old_alias,
        )
        messages.extend(stage.messages)
        if not stage.success or stage.log_buffer is None:
            return False, False, messages

        layout_changed = True
        log_handler = lambda line, buf=stage.log_buffer: buf.add_line(line)  # noqa: E731
        procs = self.controller.server_manager.start_servers(
            [new_cfg],
            {stage.alias: log_handler},
        )
        proc = procs[0] if procs else None
        success, complete_messages = self.call_from_thread(
            self.controller.complete_async_slot_launch,
            stage.alias,
            profile_id,
            plan.old_alias,
            proc,
        )
        messages.extend(complete_messages)
        return success, layout_changed, messages

    @work(thread=True)
    def _run_add_slot(self, slot_form: dict[str, str], notify_message: str = "") -> None:
        """Resolve and apply add-slot form values off the UI thread."""
        logger.debug("_run_add_slot: starting, slot_form=%r", slot_form)
        error: str | None = None
        success: bool | None = None
        messages: list[str] = []
        profile_id = ""
        layout_changed = False
        try:
            success, messages, profile_id, new_cfg = self.controller.compute_add_slot_from_form(
                slot_form
            )
            logger.debug(
                "_run_add_slot: compute_add_slot_from_form returned success=%s profile_id=%r",
                success,
                profile_id,
            )
            if success and new_cfg is not None:
                for msg in messages:
                    self.call_from_thread(self.controller._push_status_message, msg)

                plan = self.call_from_thread(
                    self.controller.prepare_async_slot_launch,
                    new_cfg,
                    profile_id,
                )
                messages.extend(plan.messages)

                success, layout_changed, messages = self._execute_slot_launch(
                    new_cfg, profile_id, plan, messages
                )
        except Exception as exc:
            logger.exception("_run_add_slot: unhandled exception")
            error = f"Add slot failed: {exc}"
            success = False
        self.call_from_thread(
            self._finish_add_slot,
            notify_message,
            error,
            success,
            messages,
            layout_changed,
        )

    def _refresh_add_slot_startup(self, alias: str) -> None:
        """Refresh the dashboard after a slot enters launching state."""
        self.view_model.mark_slot_launching(alias)
        self.refresh_dashboard()

    async def _finish_add_slot(
        self,
        notify_message: str = "",
        error: str | None = None,
        success: bool | None = None,
        messages: list[str] | None = None,
        layout_changed: bool = True,
    ) -> None:
        """Called on the UI thread after background add-slot validation completes."""
        try:
            slot_messages = list(messages or [])
            if error is None and not success:
                for msg in slot_messages:
                    self.controller._push_status_message(msg)

            configs_aliases = [c.alias for c in self.controller.configs]
            try:
                container = self.query_one(_CONTENT_CONTAINER_ID)
                panel_count_before = len(list(container.query(ServerLogPanel)))
            except NoMatches:
                panel_count_before = -1
            logger.debug(
                "_finish_add_slot: configs=%r panels_before=%d column_count=%d",
                configs_aliases,
                panel_count_before,
                self.view_model.server_column_count(),
            )
            if error:
                self.notify(error, title="Add Slot", severity="error")
            elif success and notify_message:
                logger.debug("_finish_add_slot: success=%s notify=%r", success, notify_message)
                self.notify(notify_message, severity="information")

            if success and layout_changed:
                await self._recompose_slots()
            else:
                self.refresh_dashboard()
        finally:
            self._end_slot_operation()

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
            self.refresh_dashboard()
        else:
            self._start_add_slot(result)

    def _handle_remove_slot_modal_result(self, alias: str | None) -> None:
        if not alias:
            return

        from .components.confirm_modal import ConfirmModal

        self._pending_remove_slot_alias = alias
        self.push_screen(
            ConfirmModal(
                title=_REMOVE_SLOT_TITLE,
                message=f"Remove slot '{alias}'? This will stop the server.",
            ),
            self._handle_remove_slot_confirm,
        )

    def _handle_remove_slot_confirm(self, confirmed: bool | None) -> None:
        alias = self._pending_remove_slot_alias
        self._pending_remove_slot_alias = None
        if not confirmed or alias is None:
            return
        if not self._begin_slot_operation("remove slot"):
            return
        self.notify("Removing slot…", title="Slot", severity="information")
        self._run_remove_slot(alias)

    @work(thread=True)
    def _run_remove_slot(self, alias: str) -> None:
        """Stop and remove a live slot off the UI thread."""
        error: str | None = None
        success = False
        try:
            success, messages = self.call_from_thread(
                self.controller.prepare_async_slot_remove,
                alias,
            )
            if success:
                if not self.controller.server_manager.shutdown_slot(alias):
                    messages = [f"Unable to remove '{alias}': shutdown verification failed"]
                    success = False
                else:
                    success, messages = self.call_from_thread(
                        self.controller.commit_async_slot_remove,
                        alias,
                    )
            if not success:
                for msg in messages:
                    self.call_from_thread(self.controller._push_status_message, msg)
        except Exception as exc:
            logger.exception("_run_remove_slot: unhandled exception")
            error = f"Remove slot failed: {exc}"
        self.call_from_thread(self._finish_remove_slot, alias, error, success)

    async def _finish_remove_slot(
        self,
        alias: str,
        error: str | None = None,
        success: bool = False,
    ) -> None:
        """Refresh dashboard state after background slot removal completes."""
        if error:
            self.notify(error, title=_REMOVE_SLOT_TITLE, severity="error")
        elif success:
            self.notify(f"Slot '{alias}' removed", title="Slot", severity="information")
        else:
            self.notify(
                f"Failed to remove slot '{alias}'",
                title=_REMOVE_SLOT_TITLE,
                severity="error",
            )

        try:
            if success:
                await self._recompose_slots()
            else:
                self.refresh_dashboard()
        finally:
            self._end_slot_operation()

    def _handle_build_modal_result(self, result: BuildWizardResult | None) -> None:
        if result is None:
            self.controller.cancel_pending_prompt()
        else:
            self.last_build_backend = result.backends[0] if result.backends else "sycl"
            self.controller.handle_build_selection(result.backends, result.options)
        self.refresh_dashboard()

    async def _reconcile_server_log_panels(self) -> None:
        """Ensure ServerLogPanel widgets match the current slot count."""
        start = time.perf_counter()
        container = self.query_one(_CONTENT_CONTAINER_ID, Container)
        current_panels = list(container.query(ServerLogPanel))
        needed = self.view_model.server_column_count()
        logger.debug(
            "_reconcile_server_log_panels: start current=%d needed=%d",
            len(current_panels),
            needed,
        )
        # Sort panels by _slot_index so removal matches config indices.
        current_panels.sort(key=lambda p: p._slot_index)
        # Remove excess panels (highest indices first).
        if len(current_panels) > needed:
            for panel in current_panels[needed:]:
                logger.debug(
                    "_reconcile_server_log_panels: removing excess slot_index=%d",
                    panel._slot_index,
                )
                await panel.remove()
        # Replace panels whose config no longer exists or is a placeholder.
        for panel in current_panels[:needed]:
            col_state = self.view_model.column(panel._slot_index)
            if col_state is None or not panel.query(ServerColumnPanel):
                # Config at this index was removed or panel is a placeholder.
                logger.debug(
                    "_reconcile_server_log_panels: replacing panel slot_index=%d",
                    panel._slot_index,
                )
                await panel.remove()
                await container.mount(ServerLogPanel(panel._slot_index, self.view_model))
                logger.debug(
                    "_reconcile_server_log_panels: replaced panel duration_ms=%.1f",
                    (time.perf_counter() - start) * 1000,
                )
                return  # one replacement per call
        # Mount any panels still missing.
        existing_indices = {p._slot_index for p in current_panels[:needed]}
        for i in range(needed):
            if i not in existing_indices:
                logger.debug("_reconcile_server_log_panels: mounting missing slot_index=%d", i)
                await container.mount(ServerLogPanel(i, self.view_model))
        logger.debug(
            "_reconcile_server_log_panels: complete current=%d needed=%d duration_ms=%.1f",
            len(current_panels),
            needed,
            (time.perf_counter() - start) * 1000,
        )

    def _build_profile_options(self) -> list[tuple[str, str]]:
        options, config_id = _profile_options_cached(
            self.view_model,
            self.controller.config,
            self._profile_options_cache,
            self._profile_cache_config_id,
        )
        self._profile_options_cache = options
        self._profile_cache_config_id = config_id
        return options

    def action_build(self) -> None:
        screen = BuildModalScreen(last_backend=self.last_build_backend)
        self.push_screen(screen, self._handle_build_modal_result)

    def action_confirm(self) -> None:
        self.controller.acknowledge_risk()
        self.refresh_dashboard()

    def action_reject(self) -> None:
        self.controller.reject_risk()
        if not self.controller.running:
            self.exit()
        self.refresh_dashboard()

    def action_cancel_pending_prompt(self) -> None:
        cancelled = self.controller.cancel_pending_prompt()
        if not cancelled:
            self.controller.interrupt()
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        """Lightweight periodic refresh — update text/CSS, no full recompose.

        Full recompose via ``_recompose_slots`` is only called when the
        slot count changes (add/remove).
        """
        start = time.perf_counter()
        if not self.controller.running:
            logger.info("refresh_dashboard: controller stopped, exiting app")
            self.exit()
            return

        self._emit_status_toasts()

        # Server log panels — lightweight text/CSS updates only.
        for panel in self.query(ServerLogPanel):
            panel_start = time.perf_counter()
            state = self._panel_state(panel)
            if state is not None:
                self._update_panel_widgets(panel, state)
            logger.debug(
                "refresh_dashboard: updated ServerLogPanel slot_index=%s duration_ms=%.1f",
                getattr(panel, "_slot_index", "unknown"),
                (time.perf_counter() - panel_start) * 1000,
            )

        self.refresh_bindings()
        logger.debug(
            "refresh_dashboard: complete panels=%d duration_ms=%.1f",
            len(list(self.query(ServerLogPanel))),
            (time.perf_counter() - start) * 1000,
        )

    def _panel_state(self, panel: ServerLogPanel) -> ServerColumnState | None:
        """Get current display state for a panel."""

        try:
            state = self.view_model.column(panel._slot_index)
            return state
        except Exception:
            return None

    def _update_panel_widgets(self, panel: ServerLogPanel, state: ServerColumnState) -> None:
        """Update status badge and log content on an existing panel."""
        with contextlib.suppress(NoMatches):
            status_widget = cast(Static, panel.query_one(".server-column-status"))
            status_widget.update(state.status_label)
            status_widget.classes = f"server-column-status {state.status_class}"
        with contextlib.suppress(NoMatches):
            profile_widget = cast(Static, panel.query_one(".server-column-profile-name"))
            profile_widget.update(state.profile_name)
        with contextlib.suppress(NoMatches):
            config_widget = cast(Static, panel.query_one(".server-column-config"))
            config_widget.update(state.config_summary)
        with contextlib.suppress(NoMatches):
            backend_widget = cast(Static, panel.query_one(".server-column-backend"))
            backend_widget.update(state.backend_label)
        with contextlib.suppress(NoMatches):
            url_widget = cast(Static, panel.query_one(".server-column-url"))
            url_widget.update(state.url)
        with contextlib.suppress(NoMatches):
            gpu_panel = cast(GPUStatsPanel, panel.query_one(GPUStatsPanel))
            gpu_panel.update_stats(state.gpu_stats)
        with contextlib.suppress(NoMatches):
            stats_values = list(panel.query(".slot-stats-value"))
            for widget, value in zip(
                stats_values,
                (
                    state.runtime_stats.tps,
                    state.runtime_stats.pp,
                    state.runtime_stats.tokens_in,
                    state.runtime_stats.tokens_out,
                ),
                strict=False,
            ):
                cast(Static, widget).update(value)
        with contextlib.suppress(NoMatches):
            log_widget = cast(Log, panel.query_one(".server-log-content"))
            previous: tuple[str, ...] = getattr(log_widget, "_llm_runner_lines", ())  # type: ignore[attr-defined]
            reload, lines = _split_log_update(previous, state.log_lines)
            if reload:
                log_widget.clear()
            if lines:
                log_widget.write_lines(list(lines))
            log_widget._llm_runner_lines = state.log_lines  # type: ignore[attr-defined]

    async def _recompose_slots(self) -> None:
        """Full recompose — call when slot count changes (add/remove)."""
        await self._reconcile_server_log_panels()
        self.refresh_dashboard()

    def _emit_status_toasts(self) -> None:
        notices = self.view_model.system_notices()
        current_notices = set(notices)
        for notice in notices:
            if notice not in self._active_notice_toasts:
                self.notify(notice, title="Alert", severity="warning")
        self._active_notice_toasts = current_notices

        updates = self.controller.get_status_messages_since(self._last_notified_status_ts)
        if not updates:
            return
        for ts, message in updates:
            self.notify(message, title="Status", severity="information")
            self._last_notified_status_ts = max(self._last_notified_status_ts, ts)
