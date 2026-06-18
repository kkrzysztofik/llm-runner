# Slot Log Pane Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the per-slot TUI panel so it shows the profile name instead of a dead `UNSAVED` badge, uses readable status text, separates the URL, adds mocked runtime stats, and makes logs truly scrollable.

**Architecture:** Keep the change inside the existing Textual TUI view-model/component layer. `DashboardViewModel` should prepare display-ready slot state, `ServerColumnPanel` should compose the redesigned layout, and `DashboardApp.refresh_dashboard()` should update existing widgets without full recomposition. Use Textual's `Log` widget for scrollable plain-text logs.

**Tech Stack:** Python 3.14, Textual 8.x, Rich/Textual widgets, pytest, ruff, pyright.

---

## Operating Rules For The Implementing Agent

- Follow `AGENTS.md`: do not delete files, do not run destructive git/filesystem commands, and pipe every non-interactive shell command through `distill` unless exact raw output is required.
- Make manual edits only; do not run scripts/codemods that rewrite code files.
- Use `apply_patch` for file edits.
- Do not touch Speckit or PRD files.
- Do not commit unless the user explicitly asks. If committing is requested, first run the mandatory local gate:

```bash
uv run pre-commit run --all-files 2>&1 | distill "Did pre-commit pass? Return PASS or FAIL, then failing hook names and key errors."
uv run pytest 2>&1 | distill "Did pytest pass? Return PASS or FAIL, then failing test nodeids and short errors."
```

## Target Files

- Modify: `src/llama_cli/tui/types.py`
- Modify: `src/llama_cli/tui/viewmodel.py`
- Modify: `src/llama_cli/tui/components/server_column.py`
- Modify: `src/llama_cli/tui/components/gpu_stats.py`
- Modify: `src/llama_cli/tui/textual_app.py`
- Modify: `src/llama_cli/tui/dashboard_panels.tcss`
- Modify: `src/llama_manager/log_buffer.py`
- Modify tests: `src/tests/tui/test_viewmodel.py`
- Modify tests: `src/tests/tui/test_tui.py`
- Optional test update if needed: `src/tests/runtime/test_log_buffer.py`

## Task 1: Extend Slot View State

**Files:**
- Modify: `src/llama_cli/tui/types.py`
- Test: `src/tests/tui/test_viewmodel.py`

**Step 1: Write failing view-model assertions**

In `src/tests/tui/test_viewmodel.py`, update existing `ServerColumnState` construction and column assertions to expect:

```python
from llama_cli.tui.types import SlotRuntimeStats

assert result.profile_name == "test-server"
assert result.status_label == "Offline"
assert result.log_lines == ("Waiting for output...",)
assert result.runtime_stats == SlotRuntimeStats(
    tps="--",
    pp="--",
    tokens_in="0",
    tokens_out="0",
)
```

Also update any direct `ServerColumnState(...)` fixtures to include:

```python
profile_name="slot-a",
status_label="Offline",
log_lines=("Waiting for output...",),
runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
```

Remove fixture use of:

```python
logs_text="Waiting for output...",
is_unsaved=False,
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest src/tests/tui/test_viewmodel.py -q 2>&1 | distill "Expected FAIL before implementation. Return failing nodeids and missing/changed fields only."
```

Expected: failures mention missing `SlotRuntimeStats`, `profile_name`, `status_label`, `log_lines`, or changed `ServerColumnState` constructor.

**Step 3: Implement type changes**

In `src/llama_cli/tui/types.py`, add:

```python
@dataclass(frozen=True)
class SlotRuntimeStats:
    """Display-ready per-slot runtime counters."""

    tps: str
    pp: str
    tokens_in: str
    tokens_out: str
```

Update `ServerColumnState` to:

```python
@dataclass(frozen=True)
class ServerColumnState:
    """State needed to render one server column."""

    alias: str
    profile_name: str
    status: str
    status_label: str
    status_class: str
    backend_label: str
    url: str
    config_summary: str
    log_lines: tuple[str, ...]
    runtime_stats: SlotRuntimeStats
    gpu_stats: dict[str, Any] | None
    stale_warning: str | None
```

**Step 4: Implement view-model changes**

In `src/llama_cli/tui/viewmodel.py`, import `SlotRuntimeStats` and update `DashboardViewModel.column()`:

```python
log_lines = tuple(self.model.log_buffers[cfg.alias].get_lines())
if not log_lines:
    log_lines = ("Waiting for output...",)
```

Build `ServerColumnState` with:

```python
profile_name=cfg.alias,
status_label=status.replace("_", " ").title(),
log_lines=log_lines,
runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
```

Remove `logs_text=...` and `is_unsaved=...` from the state construction.

Update the debug log length from `len(state.logs_text)` to:

```python
sum(len(line) for line in state.log_lines)
```

**Step 5: Run focused tests**

```bash
uv run pytest src/tests/tui/test_viewmodel.py -q 2>&1 | distill "Did viewmodel tests pass after type/state changes? Return PASS or FAIL and failing nodeids."
```

Expected: PASS for view-model tests, or only component fixture failures if other tests still use the old state shape.

## Task 2: Redesign Slot Header And Add Runtime Stats Panel

**Files:**
- Modify: `src/llama_cli/tui/components/server_column.py`
- Modify: `src/llama_cli/tui/dashboard_panels.tcss`
- Test: `src/tests/tui/test_tui.py`

**Step 1: Write failing component tests**

Add or update tests in `src/tests/tui/test_tui.py` near existing server panel tests.

Test header shape:

```python
def test_server_column_header_uses_profile_name_and_readable_status() -> None:
    from textual.widgets import Static

    from llama_cli.tui.components.server_column import ServerColumnPanel
    from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats

    state = ServerColumnState(
        alias="slot-a",
        profile_name="summary-balanced",
        status="crashed",
        status_label="Crashed",
        status_class="server-column-status-crashed",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="Device: SYCL0 | Ctx: 2048 | Threads: 4",
        log_lines=("line 1",),
        runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
        gpu_stats=None,
        stale_warning=None,
    )

    sections = list(ServerColumnPanel(state).compose())
    header = sections[0]
    header_text = [
        child.renderable
        for child in header.query(Static)
        if isinstance(getattr(child, "renderable", None), str)
    ]

    assert "summary-balanced" in header_text
    assert "Crashed" in header_text
    assert "UNSAVED" not in header_text
```

Test URL row and stats panel:

```python
def test_server_column_has_separate_url_row_and_mocked_stats() -> None:
    from textual.widgets import Static

    from llama_cli.tui.components.server_column import ServerColumnPanel
    from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats

    state = ServerColumnState(
        alias="slot-a",
        profile_name="summary-balanced",
        status="running",
        status_label="Running",
        status_class="server-column-status-running",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="Device: SYCL0 | Ctx: 2048 | Threads: 4",
        log_lines=("line 1",),
        runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
        gpu_stats=None,
        stale_warning=None,
    )

    sections = list(ServerColumnPanel(state).compose())
    all_static_text = [
        child.renderable
        for section in sections
        for child in section.query(Static)
        if isinstance(getattr(child, "renderable", None), str)
    ]

    assert "URL" in all_static_text
    assert "http://127.0.0.1:8080" in all_static_text
    assert "TPS" in all_static_text
    assert "PP" in all_static_text
    assert "Tok In" in all_static_text
    assert "Tok Out" in all_static_text
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest src/tests/tui/test_tui.py -q 2>&1 | distill "Expected FAIL before component redesign. Return failing nodeids and missing classes/text only."
```

Expected: failures due old `UNSAVED` header, missing URL row, missing runtime stats.

**Step 3: Implement component layout**

In `src/llama_cli/tui/components/server_column.py`:

- Import `Log` and `SlotRuntimeStats`:

```python
from textual.widgets import Log, Static

from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats
```

- Replace `_build_header()` with a header that has:
  - row 1: profile name and status badge
  - row 2: backend/config summary
  - row 3: URL on its own line

Use this shape:

```python
def _build_header(self) -> Container:
    header_children: list[Widget] = [
        Horizontal(
            Static(self._state.profile_name, classes="server-column-profile-name"),
            Static(
                self._state.status_label,
                classes=f"server-column-status {self._state.status_class}",
            ),
            classes="server-column-title-row",
        ),
        Horizontal(
            Static(self._state.backend_label, classes="server-column-backend"),
            Static(self._state.config_summary, classes="server-column-config"),
            classes="server-column-meta-row",
        ),
        Horizontal(
            Static("URL", classes="server-column-url-label"),
            Static(self._state.url, classes="server-column-url"),
            classes="server-column-url-row",
        ),
    ]
    if self._state.stale_warning:
        header_children.append(Static(self._state.stale_warning, classes="server-column-warning"))
    return Container(*header_children, classes="server-column-header")
```

- Add a runtime stats panel helper:

```python
def _build_runtime_stats(self) -> Container:
    stats = self._state.runtime_stats
    return Container(
        Static("Stats", classes="panel-title slot-stats-title"),
        Horizontal(
            self._stat_cell("TPS", stats.tps),
            self._stat_cell("PP", stats.pp),
            self._stat_cell("Tok In", stats.tokens_in),
            self._stat_cell("Tok Out", stats.tokens_out),
            classes="slot-stats-row",
        ),
        classes="slot-stats",
    )

@staticmethod
def _stat_cell(label: str, value: str) -> Container:
    return Container(
        Static(label, classes="slot-stats-label"),
        Static(value, classes="slot-stats-value"),
        classes="slot-stats-cell",
    )
```

- Update `compose()` order:

```python
yield self._build_header()
yield GPUStatsPanel(self._state.gpu_stats)
yield self._build_runtime_stats()
yield self._build_logs()
```

- Add `_build_logs()`:

```python
def _build_logs(self) -> Container:
    log = Log(max_lines=500, auto_scroll=True, classes="server-log-content")
    log.write_lines(list(self._state.log_lines))
    setattr(log, "_llm_runner_lines", self._state.log_lines)
    return Container(
        Static("Logs", classes="panel-title server-log-title"),
        log,
        classes="server-logs",
    )
```

**Step 4: Implement CSS**

In `src/llama_cli/tui/dashboard_panels.tcss`:

- Replace old `.server-column-header-row`, `.server-column-alias`, `.server-column-unsaved` rules with new selectors.
- Keep old status color classes.
- Add:

```css
.server-column-title-row,
.server-column-meta-row,
.server-column-url-row {
    height: 1;
    layout: horizontal;
}

.server-column-profile-name {
    width: 1fr;
    color: $accent;
    text-style: bold;
}

.server-column-status {
    width: auto;
    margin-left: 1;
    padding: 0 1;
    text-style: bold;
}

.server-column-backend {
    width: auto;
    margin-right: 1;
    color: ansi_bright_cyan;
    text-style: bold;
}

.server-column-config {
    width: 1fr;
    color: ansi_bright_cyan;
}

.server-column-url-label {
    width: auto;
    margin-right: 1;
    color: ansi_bright_black;
    text-style: bold;
}

.server-column-url {
    width: 1fr;
    color: ansi_white;
}

.slot-stats {
    height: auto;
    border: round ansi_bright_black;
    padding: 0 1;
    margin-bottom: 1;
    layout: vertical;
    background: black;
}

.slot-stats-title {
    width: auto;
    text-style: bold;
}

.slot-stats-row {
    height: 2;
    layout: horizontal;
}

.slot-stats-cell {
    width: 1fr;
    height: 2;
    layout: vertical;
}

.slot-stats-label {
    height: 1;
    color: ansi_bright_black;
}

.slot-stats-value {
    height: 1;
    color: ansi_bright_white;
    text-style: bold;
}
```

Update `.server-log-content` for Textual `Log`:

```css
.server-log-content {
    height: 1fr;
    color: ansi_bright_white;
    background: black;
    overflow-y: auto;
}

.server-log-content:focus {
    border: tall ansi_bright_cyan;
}
```

If focus border causes cramped content, move focus styling to `.server-logs:focus-within` instead.

**Step 5: Run focused component tests**

```bash
uv run pytest src/tests/tui/test_tui.py -q 2>&1 | distill "Did TUI component tests pass after layout redesign? Return PASS or FAIL and failing nodeids."
```

Expected: PASS or only failures in refresh code because `DashboardApp` still expects `Static` logs.

## Task 3: Make Log Pane Scrollable And Incrementally Updated

**Files:**
- Modify: `src/llama_cli/tui/textual_app.py`
- Modify: `src/llama_manager/log_buffer.py`
- Test: `src/tests/tui/test_tui.py`
- Optional test: `src/tests/runtime/test_log_buffer.py`

**Step 1: Write failing refresh test**

In `src/tests/tui/test_tui.py`, add a focused test for the log update helper. If direct widget mounting is hard, test the pure comparison helper after extracting it in Step 3.

Expected behavior:

- same tuple: no writes
- append-only tuple: writes only new lines
- rotated/replaced tuple: clears and reloads

Target helper signature:

```python
def _split_log_update(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    ...
```

Test:

```python
def test_split_log_update_appends_only_new_lines() -> None:
    from llama_cli.tui.textual_app import _split_log_update

    assert _split_log_update(("a", "b"), ("a", "b", "c")) == (False, ("c",))


def test_split_log_update_reloads_after_rotation() -> None:
    from llama_cli.tui.textual_app import _split_log_update

    assert _split_log_update(("a", "b"), ("b", "c")) == (True, ("b", "c"))
```

**Step 2: Run tests to verify failure**

```bash
uv run pytest src/tests/tui/test_tui.py::TestSystemHealthAlignment -q 2>&1 | distill "Expected FAIL for new log update helper. Return missing helper/test failure only."
```

If the exact test class is not where the new tests are added, run the specific new nodeids.

**Step 3: Implement log update helper**

In `src/llama_cli/tui/textual_app.py`, import `Log`:

```python
from textual.widgets import Footer, Log, Static
```

Add near `_profile_options_cached()`:

```python
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
```

**Step 4: Update dashboard refresh**

In `_update_panel_widgets()`:

- Update status with `state.status_label`, not `state.status.upper()`.
- Update status CSS class as before.
- Update these extra widgets because replacement can keep the same panel count:

```python
profile_widget = cast(Static, panel.query_one(".server-column-profile-name"))
profile_widget.update(state.profile_name)
config_widget = cast(Static, panel.query_one(".server-column-config"))
config_widget.update(state.config_summary)
backend_widget = cast(Static, panel.query_one(".server-column-backend"))
backend_widget.update(state.backend_label)
url_widget = cast(Static, panel.query_one(".server-column-url"))
url_widget.update(state.url)
```

- Update mocked stats:

```python
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
```

- Replace old `Static` log update with `Log` update:

```python
with contextlib.suppress(NoMatches):
    log_widget = cast(Log, panel.query_one(".server-log-content"))
    previous = cast(tuple[str, ...], getattr(log_widget, "_llm_runner_lines", ()))
    reload, lines = _split_log_update(previous, state.log_lines)
    if reload:
        log_widget.clear()
    if lines:
        log_widget.write_lines(list(lines))
    setattr(log_widget, "_llm_runner_lines", state.log_lines)
```

**Step 5: Increase retained log history**

In `src/llama_manager/log_buffer.py`, change:

```python
def __init__(self, max_lines: int = 50, redact_sensitive: bool = True) -> None:
```

to:

```python
def __init__(self, max_lines: int = 500, redact_sensitive: bool = True) -> None:
```

Do not change explicit `LogBuffer(max_lines=...)` behavior.

**Step 6: Run focused tests**

```bash
uv run pytest src/tests/tui/test_tui.py src/tests/runtime/test_log_buffer.py -q 2>&1 | distill "Did log widget and LogBuffer tests pass? Return PASS or FAIL and failing nodeids."
```

Expected: PASS.

## Task 4: Remove CPU Fallback From GPU Stats Panel

**Files:**
- Modify: `src/llama_cli/tui/components/gpu_stats.py`
- Test: `src/tests/tui/test_tui.py`
- Optional: `src/llama_cli/tui/viewmodel.py`

**Step 1: Write failing test**

In `src/tests/tui/test_tui.py`, add:

```python
def test_gpu_stats_panel_unknown_gpu_does_not_render_cpu_fallback() -> None:
    from textual.widgets import Static

    from llama_cli.tui.components.gpu_stats import GPUStatsPanel

    sections = list(
        GPUStatsPanel({"device": "N/A", "cpu": "12%", "mem": "42%"}).compose()
    )
    text = [
        child.renderable
        for section in sections
        for child in section.query(Static)
        if isinstance(getattr(child, "renderable", None), str)
    ]

    assert "CPU" not in text
    assert "Mem" not in text
    assert "GPU" in text
    assert "VRAM" in text
```

**Step 2: Run test to verify failure**

```bash
uv run pytest src/tests/tui/test_tui.py -q 2>&1 | distill "Expected FAIL for CPU fallback removal. Return failing nodeid and rendered-text assertion only."
```

Expected: failure because current panel renders `CPU` and `Mem`.

**Step 3: Implement GPU-only usage labels**

In `src/llama_cli/tui/components/gpu_stats.py`, replace the fallback branch:

```python
else:
    yield Horizontal(
        self._usage_item("CPU", cpu_pct, _fmt(stats.get("cpu"))),
        self._usage_item("Mem", sys_mem_pct, _fmt(stats.get("mem"))),
        classes="gpu-stats-usage-row",
    )
```

with:

```python
else:
    yield Horizontal(
        self._usage_item("GPU", None, "N/A"),
        self._usage_item("VRAM", None, "N/A"),
        classes="gpu-stats-usage-row",
    )
```

Then remove unused local variables:

```python
cpu_pct = self._parse_percent(stats.get("cpu"))
sys_mem_pct = self._parse_percent(stats.get("mem"))
```

Optional cleanup: in `DashboardViewModel._format_gpu_stats_text()`, change the fallback line from `CPU: ...` to `GPU: N/A | VRAM: N/A` so legacy telemetry text also avoids CPU wording.

**Step 4: Run focused tests**

```bash
uv run pytest src/tests/tui/test_tui.py::TestGPUTelemetryPanel -q 2>&1 | distill "Did GPU panel tests pass? Return PASS or FAIL and failing nodeids. If class name differs, report no matching tests."
```

If class name differs, run:

```bash
uv run pytest src/tests/tui/test_tui.py -q 2>&1 | distill "Did TUI tests pass after GPU fallback removal? Return PASS or FAIL and failing nodeids."
```

Expected: PASS.

## Task 5: Full TUI Regression Pass

**Files:**
- Modify only if tests reveal stale fixtures: `src/tests/tui/test_textual_app.py`
- Modify only if type checker reports stale imports: affected TUI files

**Step 1: Run all TUI tests**

```bash
uv run pytest src/tests/tui -q 2>&1 | distill "Did all TUI tests pass? Return PASS or FAIL, failing nodeids, and shortest useful error excerpts."
```

Expected: PASS.

**Step 2: Run type/lint checks**

```bash
uv run ruff check src/llama_cli/tui src/llama_manager/log_buffer.py src/tests/tui src/tests/runtime/test_log_buffer.py 2>&1 | distill "Did ruff check pass for touched files? Return PASS or FAIL and rule IDs."
uv run pyright 2>&1 | distill "Did pyright pass? Return PASS or FAIL and diagnostics for touched files only."
```

Expected: PASS.

**Step 3: Manual TUI smoke check**

Run dry-run launch:

```bash
uv run llm-runner dry-run both 2>&1 | distill "Did llm-runner dry-run both complete? Return PASS or FAIL and any TUI/CLI errors."
```

If a real TUI check is allowed in the environment, launch:

```bash
uv run llm-runner 2>&1 | distill "Did the TUI start without immediate traceback? Return PASS or FAIL and first traceback if any."
```

Pass criteria for visual check:

- Header shows profile name, not `UNSAVED`.
- Status is readable title-case.
- URL appears on its own row.
- GPU panel does not show `CPU`.
- Stats panel shows `TPS`, `PP`, `Tok In`, `Tok Out` with mocked values.
- Logs can receive focus and scroll with retained history.

**Step 4: Inspect diff**

```bash
git diff -- src/llama_cli/tui src/llama_manager/log_buffer.py src/tests/tui src/tests/runtime/test_log_buffer.py 2>&1 | distill "Summarize implementation diff. Return files changed and one-line behavior per file. Flag accidental unrelated changes."
```

Expected: only planned files changed.

## Final Acceptance Criteria

- `UNSAVED` is not rendered in slot headers.
- Slot/profile name renders prominently in the header.
- Status renders as `Running`, `Launching`, `Crashed`, `Offline`, or equivalent readable title-case text.
- URL is on a dedicated row.
- Runtime stats panel is present with mocked values for `TPS`, `PP`, `Tok In`, and `Tok Out`.
- GPU stats panel never uses `CPU` as a fallback label.
- Log pane uses Textual `Log`, retains more than 50 lines by default, and supports scrolling/focus.
- Focused TUI tests pass.
- Ruff and pyright pass for touched files.

## Known Non-Goals

- Do not implement real TPS/PP/token metrics in this pass.
- Do not persist unsaved slot state differently.
- Do not change slot launch/replacement behavior.
- Do not modify profile storage, Speckit files, or build pipeline code.
