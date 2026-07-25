# Concurrency Fix Plan — TUI ↔ Action Layer

Source: concurrency review of branch `int`, 2026-07-25.
Scope: thread/UI separation between Textual TUI and action layer.
Overall verdict: architecture sound; defects cluster around `ServerManager`
(only shared-mutable structure without a lock) and shutdown running on the
UI thread.

---

## Fix 1 — Quit races in-flight slot launch (MEDIUM-HIGH)

**Defect.** `ServerManager.pids / servers / slot_processes / pid_metadata`
(`src/llama_manager/orchestration/manager.py:52-57`) are mutated by:

- add-slot worker: `_execute_slot_launch` → `start_servers` appends
  (`src/llama_cli/tui/textual_app.py:624`, `manager.py:198-199`)
- remove-slot worker: `shutdown_slot` → `_forget_slot_process` removes
  (`textual_app.py:786`, `manager.py:463-472`)
- UI thread: `q` → `request_quit` → `_graceful_shutdown` → `cleanup_servers`
  (`src/llama_cli/tui/controller.py:882-889`, `manager.py:103-147`)

`_slot_operation_active` serializes slot ops against each other but not
against quit.

**Failure scenario.** Submit add-slot modal, press `q` while the worker is
inside `launcher.launch()` (model load takes seconds). `cleanup_servers`
snapshots `self.pids` before the worker's append → new llama-server orphaned
holding VRAM, lockfile PID alive and ownership-valid → next launch blocked by
lock-collision detection until manual kill.

**Fix (Textual worker model, no locks).**
1. Add `group="slot-ops"` to `_run_add_slot` and `_run_remove_slot`
   (`@work(thread=True, group="slot-ops")`).
2. New `@work(thread=True, group="slot-ops")` method
   `DashboardApp._run_shutdown`:
   - wait for any in-flight slot op (gate on `_slot_operation_active` /
     `self.workers` for the group),
   - run `controller._graceful_shutdown()` off-thread,
   - `self.call_from_thread(self.exit)`.
3. `action_quit_dashboard` / `action_interrupt_dashboard` /
   `action_reject` (quit path): stop calling `_graceful_shutdown` on the UI
   thread; dispatch `_run_shutdown` instead. Keep the risk-prompt short-circuit
   in `request_quit` on the UI thread (pure state, no blocking).
4. Guard double-dispatch: ignore quit if a shutdown worker is already running.

**Verification.** Test: start add-slot worker with a slow fake launcher,
trigger quit mid-launch, assert the launched process is in the cleanup set
(no orphan) and lockfile released. Existing quit tests still pass.

---

## Fix 2 — `cleanup_servers` blocks the UI thread 1–6+ s (MEDIUM)

**Defect.** `cleanup_servers` contains `time.sleep(1)` (`manager.py:130`) plus
`wait(timeout=5)` per process (`launcher.py:198`). Runs on the UI thread from:

- quit path — covered by Fix 1;
- `save_config` restart path (`controller.py:916-922`) — modal callback
  freezes the live UI ≥1 s per stubborn server.

**Fix.** Route the `save_config` restart's `cleanup_servers` through the same
`slot-ops` shutdown worker from Fix 1: controller stages the config save on
the UI thread (unchanged), app dispatches the worker for the
stop-servers + `running = False` part.

**Verification.** Save config with restart while a server runs; UI keeps
rendering (clock/refresh ticks) during shutdown.

---

## Fix 3 — Slot-stats snapshot applied off the UI thread (LOW)

**Defect.** Slot-stats worker calls `controller.refresh_slot_stats(targets)`
which invokes `model.apply_slot_stats_snapshot` directly on the worker
(`controller.py:235`). It prunes against `{cfg.alias for cfg in self.configs}`
(`src/llama_cli/tui/model.py:272`), but `configs` is UI-thread-mutated and not
guarded by `system_health_lock`. The GPU worker already does this correctly
(`textual_app.py:232` applies via `call_from_thread`); docstring on
`snapshot_for_probe` states the rule.

**Fix.** Either:
- (a) pass the apply back through `call_from_thread` like the GPU worker, or
- (b) derive the live-alias set from the UI-thread-taken `targets` argument
  and pass it into `apply_slot_stats_snapshot` instead of reading
  `self.configs` there. (b) is smaller and keeps persistence on the worker.

**Verification.** Existing slot-stats tests; add assertion that
`apply_slot_stats_snapshot` no longer reads `model.configs` (or is called on
the UI thread).

---

## Fix 4 — Dead code / wasted work per tick (LOW)

1. **Delete `ServerManager.wait_for_any`** (`manager.py:227`) — 0.2 s
   busy-poll, no production callers.
2. **Drop the full log copy in `viewmodel.column()`**
   (`src/llama_cli/tui/viewmodel.py:154`): `state.log_lines` copies ≤500 lines
   per panel per tick but the render path uses incremental
   `get_lines_since` (`textual_app.py:1040-1057`). Remove the field or make it
   lazy; update any tests that assert on `log_lines`.
3. **Remove the redundant `refresh_dashboard`** in
   `_refresh_slot_stats_worker`'s `finally` (`textual_app.py:293`) — the
   interval-driven refresh already picks up cached stats within one tick.

**Verification.** `uv run pytest` green; manual TUI smoke: logs still stream,
slot stats still update within ~1 refresh interval.

---

## Suggested order

1. Fix 4.1 (delete dead code) — trivial, isolated.
2. Fix 1 + Fix 2 together (same worker lane; one new method).
3. Fix 3.
4. Fix 4.2 / 4.3 (perf polish).

Non-goals: no lock added to `ServerManager` (worker-lane serialization makes
it single-threaded-in-practice); no change to log streaming, GPU telemetry,
build pipeline, or error propagation — reviewed clean.
