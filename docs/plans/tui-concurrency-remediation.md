# TUI ↔ action-layer concurrency remediation

Date: 2026-07-25 · Branch: `int` · Source: thread/concurrency review of the UI layer
vs. the launch / build / slot-lifecycle / telemetry action layer.

Scope note: `doctor` and `smoke` are CLI-only (never reachable from the TUI — no
references in `src/llama_cli/tui/`). They run synchronously with
`subprocess.run(timeout=…)` / `httpx.Timeout(...)`. No UI-thread exposure, nothing
to change.

Legend: `#T` = Textual event-loop thread · `#W` = `@work(thread=True)` worker ·
`#R` = raw `threading.Thread`.

---

## Key facts driving the plan

1. **`App.call_from_thread` is a blocking RPC.** `textual/app.py` ends it in
   `future.result()` with no timeout — the caller parks until the loop schedules
   *and completes* the callback. `post_message` is thread-safe and non-blocking.
   ~15 call sites in this repo; only 5 need a return value.
2. **Textual cannot cancel a thread worker.** `Worker.cancel()` cancels the
   awaiting asyncio task only. At shutdown `asyncio.run()` →
   `loop.shutdown_default_executor(300)` **joins** the executor those workers run
   in, so an in-flight `#W` delays process exit after the terminal is restored.
3. **The `configs[i] ↔ gpu_indices[i] ↔ gpu_stats[i]` invariant holds only on
   `#T`.** `commit_async_slot_remove` (`controller.py:719`) even raises on desync,
   but nothing protects background *readers*.

---

## Phase 1 — Build responsiveness (small diffs, highest visible win)

### 1.1 Stop marshalling one blocking round-trip per compiler line
Files: `tui/components/build.py:1007`, `tui/controller.py:1104`
Chain today: `_read_stream` (#R) → `emit_line` → `_handle_build_progress` →
`wizard.update_progress` → `call_from_thread` — **per output line**.

Closes: `cmake --build -j32` emits thousands of lines/s; each blocks the drainer
for an event-loop round trip; when the drainer stalls the 64 KiB pipe buffer
fills and ninja blocks on `write()`. Compile throughput becomes a function of TUI
render latency. stdout and stderr drainers hit the wizard concurrently.

Do: replace the body of `update_progress` with `self.post_message(...)` plus an
`on_*` handler, **or** append straight into `_pending_output_lines` as a
`deque(maxlen=2000)` (atomic append) drained by the existing
`set_timer(0.08, self._flush_build_output_buffer)`. Keep `call_from_thread` only
for stage transitions (~10 per build).

Check: build a backend; drainer never blocks — verify with a counter or by
confirming `cmake` wall time is unchanged vs. `--no-tui` build.

### 1.2 Take `terminate_process_tree` off the event loop
Files: `tui/components/build.py:1156`, `tui/controller.py:1164`,
`build_pipeline/utils.py:145`

Closes: Stop button runs `terminate_process_tree` on `#T`, which does
`while proc.poll() is None …: time.sleep(0.05)` up to 2.0 s before SIGKILL. Ninja
under load routinely eats the full 2 s → hard UI freeze, no repaint, no keys, and
every F1 drainer stacks behind it.

Do: `_start_cancel_watcher` (`utils.py:172`) already polls `cancel_event` at
100 ms and kills the tree off-thread. Drop `kill_active_subprocess()` from the
UI path so `cancel_build()` only sets the event. (Fallback if the watcher must
stay optional: `self.run_worker(controller.cancel_build, thread=True)`.)

Check: press Stop mid-compile; UI stays responsive; process tree gone within
~0.2 s.

---

## Phase 2 — Lifecycle: stop orphaning processes on quit

Files: `tui/textual_app.py:341`, `tui/controller.py:848`,
`orchestration/manager.py:103-108`

`action_quit_dashboard` cancels nothing — not the build `#R`, not the
model-index `#R`, not in-flight `_run_add_slot` / `_run_remove_slot` `#W`.

Closes three concrete failures:

- **A — orphaned llama-server.** Quit while `_run_add_slot` sits between
  `cleanup_servers()` and `start_servers()`. The new server is spawned *after*
  cleanup. `ServerManager.shutting_down` is a one-way latch (`manager.py:108`,
  never reset — verified by grep), so `controller.run()`'s
  `finally: self._cleanup()` returns instantly with `already_shutting_down`.
  Server survives the TUI holding VRAM + port + slot lockfile; next launch
  collides.
- **B — orphaned compiler.** `run_command_with_cancel` uses
  `start_new_session=True`, so cmake/ninja never sees the terminal's signals; the
  daemon build thread dies at interpreter exit *before*
  `BuildPipeline.run`'s `finally: self._release_lock()`. Compile keeps running at
  `-j$(nproc)`. The lockfile self-heals (`is_lock_stale` checks PID liveness) but
  a later build writes into the same `build_dir` **concurrently with the orphan
  ninja** → corrupt / half-linked artifacts.
- **C — post-exit hang.** Quit while a worker is inside `shutdown_slot` (10 s
  SIGTERM + 5 s SIGKILL). Terminal is restored, then `Runner.close()` joins the
  default executor → process lingers up to ~15 s. The worker's trailing
  `call_from_thread(self._finish_add_slot, …)` (`textual_app.py:670`, outside the
  `try`) then runs against a cleared DOM and raises `NoMatches`.

Do (minimum, closes A and B):
```python
# textual_app.py
def on_unmount(self) -> None:
    self.controller.cancel_build()  # sets cancel_event; watcher kills the tree

    # orchestration/manager.py — end of cleanup_servers()
    self.shutting_down = False  # re-entrant, not a one-way latch
```

Ceiling (defer unless C bites): a `shutdown` `threading.Event` the slot workers
check before each `call_from_thread`, and `run_worker(..., group="slot",
exclusive=True)` so `cancel_all()` flips `Worker.is_cancelled` for them to poll.

Check: quit mid-build → no `cmake`/`ninja` in `ps` after exit. Quit mid-add-slot
→ no orphan `llama-server`, no stale lockfile in the runtime dir.

---

## Phase 3 — Log rendering cost at steady state

File: `tui/textual_app.py:49` (`_split_log_update`), `:993`;
`llama_manager/log_buffer.py`

Closes: `LogBuffer` is `deque(maxlen=500)`. Once full, the prefix test
`current[:len(previous)] == previous` is **permanently false** (oldest lines
dropped from the front), so every refresh does `clear()` + `write_lines(500)`.
Cadence is `tui_refresh_interval_ms=1000` **plus** the extra `refresh_dashboard`
forced by `_refresh_slot_stats_worker` (`textual_app.py:286`) → ~2 full rewrites
/s/panel, ~2000 `Log` writes/s at two slots, on `#T`, precisely when the server is
busiest. Reads as "Textual is slow", not as a logic bug.

Do: add a monotonic counter to `LogBuffer` (`self._seq += 1` per `add_line`,
inside the existing lock) plus `get_lines_since(seq) -> tuple[int, list[str]]`;
the widget consumes only the tail delta. The deque's own eviction remains the
ceiling. ~10 lines.

Check: with a slot logging continuously past 500 lines, per-refresh
`_update_panel_widgets` duration stays flat (existing `logger.debug` timing lines
already report it).

---

## Phase 4 — The snapshot cluster (one change, retires three bugs)

All three are the same defect: **background readers straddle multi-field shared
state with no snapshot boundary.**

- **4a `textual_app.py:192-211`** — `_refresh_gpu_stats_worker` takes
  `list(gpu_stats)` and `[cfg.alias for cfg in configs]` as *two separate reads*.
  Remove slot 0 of 2 between them and index 0 maps the removed card's
  utilisation / temp / power onto the surviving slot's panel for a full cycle
  (≥1 s, longer if the next `xpu-smi` probe hits its 2 s timeout). Wrong hardware
  data on screen, not just a gap.
- **4b `tui/model.py:209`, `:226`** — `apply_gpu_stats_snapshot` /
  `apply_slot_stats_snapshot` **replace** the whole dict while
  `set_cached_gpu_stats` / `set_cached_slot_stats` write single entries from `#T`.
  A worker that started before slot B existed lands `{A: …}` and deletes B's
  entry that `stage_async_slot_launch` just wrote → blank GPU panel on the slot
  the user is watching come up. Symmetrically, a stale snapshot can resurrect a
  removed alias.
- **4c `controller.py:184`** — `refresh_slot_stats` iterates `self.model.configs`
  live while `#T` appends/deletes; also never prunes removed aliases from the
  persisted stats. It now takes a `targets` snapshot argument (defaulting to a
  fresh `snapshot_for_probe()` for UI-thread callers) and
  `_refresh_slot_stats_worker` passes one taken via `call_from_thread`.

Do: add one `DashboardModel.snapshot_for_probe()` returning a frozen
`tuple[tuple[str, GPUStats, ServerConfig], ...]` built on `#T` under
`system_health_lock`; workers call it via `call_from_thread` and iterate that.
Change the two `apply_*_snapshot` methods from replace to merge-and-prune:
`cache.update(new)` then drop keys not in `live_aliases`, inside the existing
lock.

Check: add/remove slots in a loop while GPU telemetry refreshes; no panel ever
shows another slot's device, no blank GPU panel after add.

---

## Phase 5 — Cheap cleanups (independent, any order)

| # | File | Change | Closes |
|---|------|--------|--------|
| 5.1 | `controller.py:176` | Hoist `registry = self._build_tui_registry()` out of the per-config loop; cache `alias → profile_id`; persist stats on change, not every cycle | 2 TOML parses **per config per second**; 4 dead slots = 1.6 s of connect timeouts in a 1 s cadence → stats silently degrade to ~2 s and drag the forced `refresh_dashboard` with them |
| 5.2 | `slot_profile_store.py:101`, `textual_app.py:383/400` | Replace per-profile `custom_slot_profile_exists` with one `{p.profile_id for p in load_custom_slot_profiles()}` set; move `action_manage_profiles` / `action_profile_stats` gathering into `@work(thread=True)` and `push_screen` via `call_from_thread` | Pressing `p` with 20 profiles + cold index = 42 TOML parses + a multi-MB `json.loads` on `#T`; seconds of frozen TUI on a network `models_dir` |
| 5.3 | `build_pipeline/utils.py:188` | `stdout_lines` / `stderr_lines` → `deque(maxlen=200)` | A full SYCL build holds 10⁴–10⁵ lines (tens of MB) for consumers that only want `_tail_lines(…, 12)` |
| 5.4 | `controller.py:1104` | Store the immutable `BuildProgress` as one attribute, derive the rest; collapse the duplicated `build_in_progress` (controller + model, set non-atomically; `_signal_handler` reads only the controller's) | UI can render `stage="configure"` with a percent from the build stage |
| 5.5 | `textual_app.py:943` | `logger.exception(...)` before `return None` in `_panel_state` | `except Exception: return None` wraps an unguarded `log_buffers[cfg.alias]` (`viewmodel.py:154`) — a desync `KeyError` renders as a permanently frozen panel with no log line anywhere. This is why the races above are invisible in production |

---

## Phase 6 — Architectural follow-up

Files: `controller.py`, `textual_app.py`, `components/build.py`

Three concurrency models coexisted: `@work(thread=True)` for telemetry and slot
lifecycle, raw `threading.Thread(daemon=True)` for build and model-index, and a
`ThreadPoolExecutor` nested inside a Textual worker (`build.py:440`). The raw
threads were invisible to `self.workers` (so `cancel_all()` never touched them —
the root cause of Phase 2) and their exceptions never surfaced as
`WorkerState.ERROR`.

Done for the **build**: `_run_build_background` split into
`DashboardController.begin_build` (UI-thread state reservation) and
`run_build_loop` (blocking pipeline body). `DashboardApp.start_build` now owns
the thread via `@work(thread=True, group="build")`. Beyond hygiene this buys a
real correctness win: shutdown joins workers, so `BuildPipeline.run`'s
`finally: self._release_lock()` actually executes instead of being skipped by a
daemon thread killed at interpreter exit — the build lock no longer depends on
PID-staleness detection to recover. `start_build` also refuses a second
concurrent build (`build_in_progress`), which nothing previously prevented;
`exclusive=True` would have been *worse* here, since cancelling a thread worker
does not stop the thread and would have left two builds racing the same lock.

**Not done, deliberately — the model-index refresh stays a daemon thread**
(`controller.py:refresh_model_index_async`). Converting it would make shutdown
join a full rescan, stalling quit for as long as a large or network-mounted
`models_dir` takes to walk. Nothing is gained in exchange: the index is written
atomically and the scan is idempotent, so there is no `finally` worth waiting
for. The reasoning is recorded in the method docstring so it does not get
"fixed" later.

Cancellation wiring (`Worker.cancelled_event` → `build_cancel_event`) was also
skipped: `on_unmount` already sets the cancel event and kills the process tree,
and it runs before the executor join, so the join stays short.

---

## What is already correct (do not "fix")

- GPU probes carry `subprocess.run(timeout=1..2)` and run **only** in workers —
  no vendor tool ever executes on `#T`.
- `LogBuffer` and every `DashboardModel` telemetry cache are lock-guarded.
- All three periodic refresh workers have correct re-entrancy guards.
- `stage_async_slot_launch` / `complete_async_slot_launch` marshal *all* shared
  state mutation to `#T` while keeping `Popen` and lockfile I/O on the worker —
  that is the right split.
- `cli_main` disables the stderr sink in TUI mode (`server_runner.py:413`), so
  per-line `logger.info` from `stream_pipe` cannot corrupt the alternate screen.
