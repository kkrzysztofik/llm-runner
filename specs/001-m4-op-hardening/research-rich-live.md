# Research: Rich Live Key Polling for Hardware Warnings

**Date**: 2026-04-23
**Feature**: M4 — Operational Hardening and Smoke Verification
**Status**: Complete

---

## Decision 1: Non-blocking Input Polling Architecture

### Decision
Use the **existing `_input_poller` daemon thread + `_keypress_queue` pattern** already present in `tui_app.py`, extended with a **state machine** that gates which keys are valid at any given time. The poller thread runs continuously (cbreak mode, `select.select` with 50ms timeout on POSIX, `msvcrt.kbhit()` on Windows). The main render loop drains the queue in `_process_keypresses()` and checks a `_warning_state` flag to determine whether to handle y/n/q keys or ignore them.

This leverages the proven pattern already in the codebase (used for profile flavor selection) and requires no new infrastructure.

```python
# State machine for hardware warning acknowledgment
class _WarningState:
    NONE = "none"           # Normal operation, no prompt shown
    WAITING = "waiting"     # Warning displayed, waiting for y/n/q
    RESOLVED = "resolved"   # User responded or timed out

# In TUIApp.__init__():
self._warning_state: str = _WarningState.NONE
self._warning_ack_deadline: float = 0.0  # monotonic time when timeout fires
self._warning_result: bool | None = None  # True=continue, False=abort
```

### Rationale
- **Already working**: The `_input_poller` + `_keypress_queue` pattern is proven — it handles profile flavor selection (keys `1`, `2`, `3`) and abort profiling (`^C`) using exactly this architecture.
- **Zero blocking**: The poller thread does the blocking I/O; the main thread only does `queue.get_nowait()`, which never blocks.
- **Rich Live compatible**: Since `_process_keypresses()` runs inside the `while self.running` loop that calls `live.update(self.render(), refresh=True)`, the display updates continuously while polling.
- **State-driven**: The `_warning_state` flag ensures keys are only interpreted in the right context. During `_WarningState.NONE`, y/n/q keys are silently ignored (or could be bound to other actions in the future). During `_WarningState.WAITING`, only those keys matter.

### Alternatives Considered
- **Blocking `input()` inside Live context**: Rejected — Rich's `Live` context manager captures stdout and prevents `input()` from working correctly. The display would freeze.
- **Rich's `Prompt` widget**: Rejected — Rich's `Prompt` is designed for single-shot synchronous prompts, not for use inside a `Live` loop. It would require exiting the `Live` context, showing the prompt, then re-entering — which would cause a visible screen flicker.
- **`select.select()` with timeout in the main loop**: Rejected — the main loop drives the `Live` render; blocking it for up to 30 seconds would freeze the display. The daemon thread approach keeps the render loop responsive.

---

## Decision 2: Cross-Platform Key Detection

### Decision
**Reuse the existing `_cbreak_stdin()` + `_input_poller()` pattern** from `tui_app.py`. It already handles:
- POSIX: `select.select([sys.stdin], [], [], 0.05)` with `tty.setcbreak(fd)`
- Windows: `msvcrt.kbhit()` + `msvcrt.getch()`
- Non-tty stdin: Falls through without cbreak mode

No changes needed to the polling infrastructure itself.

```python
# Already in tui_app.py — no changes needed
@contextmanager
def _cbreak_stdin() -> Any:
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return

    try:
        import termios
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (ImportError, OSError, ValueError):
        yield
        return

    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
```

### Rationale
- **Already tested**: This pattern is in production in the current codebase. The `select.select` approach works on Linux/macOS; `msvcrt` works on Windows.
- **cbreak mode**: `tty.setcbreak()` is essential — it makes each keypress available immediately without requiring Enter. Without it, `select.select` would only report stdin as ready after a newline is typed.
- **Non-tty fallback**: When stdin is not a TTY (piped input, CI), the context manager yields immediately without setting cbreak mode. This is important for testing and automated scenarios.

### Cross-Platform Details

| Platform | Detection | Mode | Notes |
|----------|-----------|------|-------|
| Linux/macOS | `select.select([stdin], [], [], 0.05)` | `tty.setcbreak()` | Each byte available immediately |
| Windows | `msvcrt.kbhit()` → `msvcrt.getch()` | Console input | `kbhit()` is non-blocking; `getch()` returns immediately |
| Non-TTY | No-op | N/A | Keys never arrive; queue stays empty |

### Polling Interval
The existing poller uses `select.select(..., 0.05)` — a 50ms timeout. This is a good balance:
- **Responsive enough**: User presses y/n/q and it's detected within ~50ms.
- **CPU-friendly**: 50ms sleep between checks means ~20 iterations/sec, using negligible CPU.
- **Compatible with TUI refresh**: The TUI loop sleeps 100ms (`time.sleep(0.1)` in `run()`), so the poller runs roughly twice as fast as the render loop. This is fine — the queue just accumulates keys that get drained each render cycle.

---

## Decision 3: Timeout Handling

### Decision
Use a **monotonic deadline** checked on each render cycle. The poller thread continues running; the main loop checks `time.monotonic() >= self._warning_ack_deadline` in `_process_keypresses()` or in the `run()` loop. If the deadline has passed and no key was pressed, the state transitions to `_WarningState.RESOLVED` with `self._warning_result = False` (abort).

```python
# In TUIApp.run() or _process_keypresses():
def _check_warning_timeout(self) -> bool:
    """Check if the warning acknowledgment has timed out.

    Returns True if the warning was resolved (timed out or user responded).
    """
    if self._warning_state != _WarningState.WAITING:
        return False

    if time.monotonic() >= self._warning_ack_deadline:
        self._warning_state = _WarningState.RESOLVED
        self._warning_result = False  # Safe default: abort
        return True
    return False
```

### Rationale
- **No extra threads**: Adding a second timer thread would be overkill. The main render loop already runs at ~10Hz (100ms sleep). Checking a monotonic deadline is a single float comparison — negligible cost.
- **Monotonic clock**: `time.monotonic()` is immune to system clock adjustments (NTP sync, manual changes). `time.time()` can jump backward, causing premature timeouts.
- **Safe default**: Per the spec, the 30-second timeout default is `n`/abort. This is the conservative choice — if the operator walks away, the system doesn't proceed unsafely.
- **Deadline, not elapsed counter**: Storing a single deadline (`time.monotonic() + 30`) is simpler than tracking elapsed time. No need to update a counter on each cycle.

### Why Not `threading.Timer`?
`threading.Timer` would fire a callback after 30 seconds, but:
1. It requires thread-safe state mutation (the callback would need to set `_warning_state` and `_warning_result` under a lock).
2. It adds complexity for a simple deadline check that the main loop already does every 100ms.
3. The callback would need to trigger a TUI refresh — but `live.update()` is only safe from the main thread.

### Implementation in the Run Loop
```python
# In TUIApp.run():
with Live(self.render(), screen=True, refresh_per_second=10,
          auto_refresh=False, vertical_overflow="ellipsis") as live:
    while self.running:
        self._check_warning_timeout()  # Check deadline
        self._process_keypresses()     # Drain keypress queue
        time.sleep(0.1)
        live.update(self.render(), refresh=True)
```

---

## Decision 4: Layout Management for Warning Prompt

### Decision
**Update the `risk_panel` attribute** (already in `TUIApp` as `self.risk_panel: Panel | None`) and **append a prompt line** to it. The `render()` method already includes `self.risk_panel` in the alerts section (lines 197-199 of `tui_app.py`). When a hardware warning is active, `risk_panel` is set to a `Panel` containing the warning text plus a prompt line at the bottom.

```python
def _build_warning_panel(
    self,
    slot_id: str,
    warning_text: str,
    time_remaining: float,
) -> Panel:
    """Build the hardware warning panel shown during acknowledgment wait."""
    lines = [
        f"[bold red]HARDWARE WARNING[/bold red] — {slot_id}",
        f"[dim]{warning_text}[/dim]",
        "",
        f"[yellow]Time remaining: {time_remaining:.0f}s[/yellow]",
        "  [bold]y[/bold] Continue  [bold]n[/bold] Abort  [bold]q[/bold] Quit",
    ]
    return Panel(
        "\n".join(lines),
        title="[red]Hardware Warning[/red]",
        border_style="red",
    )
```

The `render()` method already handles `self.risk_panel` in the alerts section:
```python
# tui_app.py render() — already handles risk_panel
alerts: list[Panel] = []
if self.risk_panel is not None:
    alerts.append(self.risk_panel)
# ... other panels ...
if alerts:
    layout["alerts"].update(Panel(Group(*alerts), title="System Alerts", border_style="yellow"))
```

### Rationale
- **Minimal layout change**: The `risk_panel` is already part of the alerts section. We just update its content. No new layout slots needed.
- **All panels stay visible**: The spec says "all existing panels stay visible" during the warning prompt. By updating `risk_panel` in-place (not replacing the entire layout), all other panels (logs, GPU stats) continue rendering.
- **Time remaining display**: Shows countdown to give the user a sense of urgency without being alarming.
- **Prompt line at bottom**: The `[y] Continue [n] Abort [q] Quit` line is part of the panel content, not a separate layout element. This keeps the layout simple.

### Why Not a Separate Layout Slot?
Adding a new layout slot (e.g., `Layout(name="warning_prompt", size=3)`) would:
1. Require modifying `build_layout()` to always include this slot.
2. Take up terminal space even when no warning is active (showing blank or dim text).
3. Add complexity to the layout tree without clear benefit.

The existing `risk_panel` approach is simpler and more flexible — it appears only when needed and replaces the content of the alerts panel entirely.

---

## Decision 5: Keyboard Interrupt (Ctrl+C) Handling

### Decision
**Treat Ctrl+C as "abort" during the warning prompt**, and then exit the TUI normally (clean shutdown of running servers). Do NOT raise an exception or call `sys.exit()` directly — the `finally` block in `run()` ensures `_stop_input_polling()` and `_cleanup()` run.

```python
def _process_keypresses(self) -> None:
    while not self._keypress_queue.empty():
        try:
            key = self._keypress_queue.get_nowait()
        except queue.Empty:
            break

        # Handle Ctrl+C during warning prompt
        if self._warning_state == _WarningState.WAITING and key == "^C":
            self._warning_state = _WarningState.RESOLVED
            self._warning_result = False  # Abort
            self._push_status_message("Hardware warning: abort (Ctrl+C).")
            continue

        # ... rest of key handling ...
```

### Rationale
- **Consistent with existing behavior**: The existing code already maps `^C` to abort during profile operations (line 419-421 of `tui_app.py`). Extending this pattern to the warning prompt is consistent.
- **Ctrl+C is "abort" in all contexts**: The user pressed Ctrl+C, which semantically means "stop what you're doing." During a warning prompt, "stop" = don't proceed = abort.
- **Clean shutdown via `finally`**: The `run()` method wraps the `Live` context in a `try/finally` that calls `_stop_input_polling()`. The `_cleanup()` method calls `self.server_manager.cleanup_servers()` which sends SIGTERM to all running servers. Ctrl+C during the warning prompt should trigger this same cleanup.
- **No `sys.exit()` in TUI loop**: Calling `sys.exit()` inside the `Live` context can leave the terminal in a broken state (cursor hidden, alternate screen not restored). Setting `self.running = False` is the safe approach — the loop exits naturally, `finally` runs, and `_cleanup()` handles graceful server shutdown.

### What About the `_signal_handler`?
The existing `_signal_handler` method (line 136) is registered for `SIGINT` and `SIGTERM`. It calls `self.stop()` which sets `self.running = False`. This is the handler for actual OS signals (e.g., user presses Ctrl+C at the shell level, or the system sends SIGTERM).

The `_input_poller` thread converts `^C` (Ctrl+C byte from stdin) to a keypress in the queue. This is different from the OS signal — it's the raw byte `0x03` read from the terminal in cbreak mode.

Both paths lead to the same outcome:
1. **OS signal (`SIGINT`)**: `_signal_handler` → `self.stop()` → `self.running = False` → loop exits → `finally` → `_cleanup()`
2. **Ctrl+C keypress**: `_input_poller` → queue `^C` → `_process_keypresses` → during warning: set `_warning_result = False` → loop continues → next render shows resolved state

If Ctrl+C is pressed during the warning, the operator gets a clear "abort" message in the status panel, then the TUI continues running (they can still see logs from other slots). If they want to exit entirely, they press Ctrl+C again (OS signal path) or press `q` (which we handle as "quit TUI").

### The `q` Key: Quit vs Abort
Per the spec: "`n` or `q` to abort". Both `n` and `q` mean "abort the warning and don't proceed." This is distinct from "quit the TUI." If the operator wants to quit the TUI entirely, they press Ctrl+C (OS signal).

```python
if self._warning_state == _WarningState.WAITING:
    if key in ("y",):
        self._warning_state = _WarningState.RESOLVED
        self._warning_result = True  # Continue
        self._push_status_message(f"Hardware warning: continuing.")
    elif key in ("n", "q"):
        self._warning_state = _WarningState.RESOLVED
        self._warning_result = False  # Abort
        self._push_status_message(f"Hardware warning: aborting.")
    elif key == "\r" or key == "\n":
        # Enter without y/n/q — re-display prompt (no state change)
        pass  # Don't advance; next render re-shows the panel
    else:
        # Any other key — silently ignore
        pass
```

---

## Integration with Existing Code

### Where the Changes Go

All changes go in `src/llama_cli/tui_app.py`. No new files needed.

```
src/llama_cli/
  tui_app.py    # Add _warning_state, _build_warning_panel(),
                # extend _process_keypresses() for y/n/q,
                # extend render() to show warning panel
```

### State Machine Flow

```
                    ┌─────────────┐
                    │ _WARNING    │
                    │ _STATE =    │
                    │   "none"    │
                    └──────┬──────┘
                           │
                    Warning detected
                    (from process manager /
                     hardware check)
                           │
                           ▼
                    ┌─────────────┐
                    │ _WARNING    │◄──────────────────────┐
                    │ _STATE =    │                       │
                    │  "waiting"  │                       │
                    │             │                       │
                    │ Show panel  │                       │
                    │ Update      │                       │
                    │ risk_panel  │                       │
                    └──┬──────┬───┘                       │
                       │      │                          │
              User presses │      │ 30s timeout           │
              y / n / q    │      │ (safe default: n)     │
                       │      │                          │
              ┌────┐  │      │  ┌────┐                   │
              │ y  │  │      │  │ n  │                   │
              │cont│  │      │  │abort│                   │
              └──┬─┘  │      │  └──┬─┘                   │
                 │     │      │     │                     │
                 ▼     │      │     │                     │
              ┌────────┴─┐    │     │                     │
              │_WARNING_│    │     │                     │
              │ _STATE = │    │     │                     │
              │ "resolved"│   │     │                     │
              │_RESULT=T/F│   │     │                     │
              └─────┬────┘    │     │                     │
                    │         │     │                     │
                    └─────────┴─────┘                     │
                              │                          │
                      TUI continues                    │
                      running (other slots              │
                      still visible)                    │
                              │                          │
                              └──────────────────────────┘
                              (Enter on empty re-displays)
```

### Key Integration Points

1. **`_build_warning_panel()`**: New method — builds the `Panel` with warning text + prompt.
2. **`_process_keypresses()`**: Extended — checks `_warning_state` and handles y/n/q/^C.
3. **`render()`**: No changes needed — already includes `self.risk_panel` in alerts.
4. **`run()`**: Add `_check_warning_timeout()` call before `_process_keypresses()`.
5. **Triggering the warning**: The warning is triggered from outside the TUI loop (e.g., after `launch_result` is computed). The caller sets `self.risk_panel`, `self._warning_state`, and `self._warning_ack_deadline`, then the render loop handles the rest.

---

## Testing Strategy

### Unit Tests (in `tests/test_tui_app.py`)

```python
import time
import pytest
import queue
from llama_cli.tui_app import TUIApp, _WarningState

class TestWarningTimeout:
    def test_timeout_fires_after_deadline(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._warning_ack_deadline = time.monotonic() + 0.1  # 100ms

        # Should not have timed out yet
        assert app._check_warning_timeout() is False
        assert app._warning_state == _WarningState.WAITING

        # Wait for deadline
        time.sleep(0.15)

        # Should have timed out
        assert app._check_warning_timeout() is True
        assert app._warning_state == _WarningState.RESOLVED
        assert app._warning_result is False  # Safe default: abort

    def test_no_timeout_before_deadline(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._warning_ack_deadline = time.monotonic() + 10.0  # 10 seconds

        assert app._check_warning_timeout() is False
        assert app._warning_state == _WarningState.WAITING


class TestWarningKeypresses:
    def test_y_key_resolves_to_continue(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("y")
        app._process_keypresses()

        assert app._warning_state == _WarningState.RESOLVED
        assert app._warning_result is True

    def test_n_key_resolves_to_abort(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("n")
        app._process_keypresses()

        assert app._warning_state == _WarningState.RESOLVED
        assert app._warning_result is False

    def test_q_key_resolves_to_abort(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("q")
        app._process_keypresses()

        assert app._warning_state == _WarningState.RESOLVED
        assert app._warning_result is False

    def test_enter_re_displays_prompt(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("\r")
        app._process_keypresses()

        # State should NOT change — prompt re-displayed
        assert app._warning_state == _WarningState.WAITING

    def test_other_keys_ignored(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("x")
        app._process_keypresses()

        assert app._warning_state == _WarningState.WAITING

    def test_ctrl_c_aborts(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.WAITING
        app._keypress_queue.put("^C")
        app._process_keypresses()

        assert app._warning_state == _WarningState.RESOLVED
        assert app._warning_result is False

    def test_warning_keys_ignored_when_not_waiting(self) -> None:
        app = TUIApp(configs=[], gpu_indices=[])
        app._warning_state = _WarningState.NONE
        app._keypress_queue.put("y")
        app._process_keypresses()

        # State should NOT change
        assert app._warning_state == _WarningState.NONE
```

### Mocking stdin in Tests
Tests don't need to mock `select.select` or `msvcrt` — they directly populate `_keypress_queue` to simulate keypresses. The `_input_poller` thread is not started in unit tests (it only runs when `run()` is called, which would require a real terminal).

---

## CI Quality Gates

All plans must note the CI gates:

1. **lint** — `uv run ruff check .` + `uv run ruff format --check`
2. **typecheck** — `uv run pyright`
3. **test** — `uv run pytest` with coverage

### Ruff Considerations
- The new `_WarningState` class should use `class` (not `Enum`) for simplicity — it's an internal state machine, not a serializable type.
- `_check_warning_timeout()` should be a private method (`_` prefix) since it's only called from `run()`.
- The `time` import is already present in `tui_app.py` (line 14).

### Pyright Considerations
- `_warning_state: str` is typed as `str` for simplicity. If strict type checking is desired, use a `Literal` type:
  ```python
  _WarningStateType = Literal["none", "waiting", "resolved"]
  _warning_state: _WarningStateType = "none"
  ```
- `_warning_result: bool | None` — the `| None` is necessary because the result is not set until the user responds or times out.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `tty.setcbreak()` leaves terminal in bad state if process crashes | `_cbreak_stdin()` context manager restores original settings in `finally` block. |
| Poller thread keeps running after TUI exits | `_stop_input_polling()` joins the thread with 1s timeout in `_cleanup()`. |
| Key press lost between poller and render loop | Queue is unbounded; keys are not dropped. The `_input_poller` uses `queue.Queue.put()` which blocks only if the queue is full (impossible for an unbounded queue). |
| Warning panel text too long for narrow terminals | Rich's `Panel` wraps text automatically. The prompt line is short. If needed, `layout["alerts"].size` could be made dynamic based on terminal height. |
| 30-second timeout too long/short for operator | Configurable via `Config.warning_ack_timeout_s` (default 30). Can be tuned per deployment. |
| Ctrl+C during warning vs Ctrl+C to quit TUI | Both work: Ctrl+C during warning = abort the warning (TUI continues). Second Ctrl+C = OS signal → `self.stop()` → TUI exits. Clear status message distinguishes the two. |

---

## Implementation Sequence

1. **Add `_WarningState` class** — Simple class with `NONE`, `WAITING`, `RESOLVED` attributes.
2. **Add state attributes to `TUIApp.__init__()`** — `_warning_state`, `_warning_ack_deadline`, `_warning_result`.
3. **Add `_build_warning_panel()` method** — Returns a `Panel` with warning text, countdown, and prompt.
4. **Extend `_process_keypresses()`** — Check `_warning_state == _WarningState.WAITING` and handle y/n/q/^C.
5. **Add `_check_warning_timeout()` method** — Checks `time.monotonic() >= _warning_ack_deadline` and resolves to abort.
6. **Integrate into `run()` loop** — Call `_check_warning_timeout()` before `_process_keypresses()`.
7. **Wire up the trigger** — After `launch_result` is computed (or from the hardware check), set the warning state and panel.

---

## Summary of Decisions

| Decision | Choice | Key Reason |
|----------|--------|-----------|
| Input polling | Reuse existing `_input_poller` + `_keypress_queue` | Already working, proven pattern |
| Cross-platform | `select` + `tty.setcbreak` (POSIX), `msvcrt` (Windows) | Standard, tested approach |
| Timeout | Monotonic deadline checked each render cycle | No extra threads, simple, accurate |
| Layout | Update `self.risk_panel` in-place | Minimal change, all panels stay visible |
| Ctrl+C | Treat as "abort" during warning | Consistent with existing behavior, clean shutdown |
| `q` key | Same as `n` (abort), not "quit TUI" | Per spec: "`n` or `q` to abort" |
| Enter key | Re-display prompt (no state change) | Per spec: "re-display prompt" |
| Default on timeout | Abort (`n`) | Safe default — don't proceed unsafely |
