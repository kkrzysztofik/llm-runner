# Slot Stats Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Populate the TUI per-slot `Stats` card with live llama-server slot counters, persist the latest counters to runtime JSON, and show the persisted view immediately on TUI restart before the next poll.

**Architecture:** Keep collection and persistence in `llama_manager` so it is testable without Textual. The TUI model owns the cached display state, the controller refreshes it off the render path, and `DashboardViewModel.column()` only converts already-cached data into `SlotRuntimeStats`.

**Tech Stack:** Python 3.12, stdlib `urllib.request` + `json`, Textual widgets, dataclasses, `resolve_runtime_dir()`, `atomic_write_json()`, pytest, ruff, pyright.

---

## Context

Current UI already renders a `Stats` card:

- `src/llama_cli/tui/components/server_column.py` builds `TPS`, `PP`, `Tok In`, and `Tok Out`.
- `src/llama_cli/tui/textual_app.py` updates `.slot-stats-value` during `refresh_dashboard()`.
- `src/llama_cli/tui/types.py` already defines display dataclass `SlotRuntimeStats`.
- `src/llama_cli/tui/viewmodel.py` currently hardcodes `SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0")`.

Use llama.cpp `GET /slots` as the live source. Upstream documents `/slots` as enabled by default and useful for per-slot speed and processed-token metrics. Do not require `--metrics`; `/metrics` is a separate Prometheus endpoint that requires `--metrics`.

Persist only display-safe numeric/runtime metadata. Do not persist prompts, request payloads, user data from `/slots`, or raw endpoint responses.

Runtime store path:

- Use `llama_manager.orchestration.lockfile.resolve_runtime_dir()`.
- Store JSON at `<runtime_dir>/slot-stats.json`.
- Write through `llama_manager.common.file_ops.atomic_write_json()`.

Display semantics:

- `TPS`: generation throughput, formatted as one decimal token/s from the best available live field.
- `PP`: prompt processing throughput, formatted as one decimal token/s when available, else `--`.
- `Tok In`: cumulative prompt/input tokens for this server process, integer string.
- `Tok Out`: cumulative generated/output tokens for this server process, integer string.
- Offline/unreachable slot: show persisted last-known values if present; else show `--`, `--`, `0`, `0`.
- Restarted process: keep persisted values until first successful live poll, then replace with live values.

## No-Go

- Do not edit Speckit files.
- Do not remove smoke/profile CLI behavior.
- Do not parse or persist raw prompt text.
- Do not add a new dependency.
- Do not collect HTTP stats from `DashboardViewModel.column()` or widget `compose()`; render path must stay cache-only.
- Do not delete files.

## Data Model

Add a manager-level stats model in `src/llama_manager/slot_stats.py`.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotStatsSnapshot:
    alias: str
    port: int
    updated_at: float
    tps: float | None = None
    prompt_tps: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    source: str = "slots"

    def to_display(self) -> dict[str, str]:
        return {
            "tps": _format_rate(self.tps),
            "pp": _format_rate(self.prompt_tps),
            "tokens_in": str(max(0, self.tokens_in)),
            "tokens_out": str(max(0, self.tokens_out)),
        }


def _format_rate(value: float | None) -> str:
    if value is None or value < 0:
        return "--"
    return f"{value:.1f}"
```

Parser rule for `/slots`:

- Accept `list[dict[str, object]]`; ignore anything else.
- Sum across all returned llama.cpp internal slots for one server instance.
- `tokens_out`: sum `next_token.n_decoded` when present.
- `tokens_in`: use the best available prompt/processed token fields if present. Check keys defensively because llama.cpp changes shape over time:
  - top-level `n_prompt_tokens_processed`, `n_prompt_tokens`
  - nested `prompt.n_tokens`, `prompt.n_tokens_processed`
  - fallback `0`
- `tps`: use first available numeric generation-rate field:
  - `next_token.tps`, `tokens_per_second`, `generation_tps`, `tps`
  - fallback `None`
- `prompt_tps`: use first available numeric prompt-rate field:
  - `prompt_tps`, `prompt_tokens_per_second`, `prompt.tokens_per_second`
  - fallback `None`

The parser should be intentionally tolerant. Unknown keys must not fail collection.

## Task 1: Add Pure Stats Parser Tests

**Files:**

- Create: `src/llama_manager/slot_stats.py`
- Create: `src/tests/slot/test_slot_stats.py`

**Step 1: Write failing parser tests**

Add tests:

```python
from llama_manager.slot_stats import SlotStatsSnapshot, parse_slots_payload


def test_parse_slots_payload_sums_decoded_tokens() -> None:
    payload = [
        {"next_token": {"n_decoded": 7}, "n_prompt_tokens_processed": 11},
        {"next_token": {"n_decoded": 3}, "n_prompt_tokens_processed": 5},
    ]

    result = parse_slots_payload("summary", 8080, payload, now=10.0)

    assert result == SlotStatsSnapshot(
        alias="summary",
        port=8080,
        updated_at=10.0,
        tokens_in=16,
        tokens_out=10,
    )


def test_parse_slots_payload_reads_rate_fields_defensively() -> None:
    payload = [
        {
            "next_token": {"n_decoded": 12, "tps": 4.25},
            "prompt": {"n_tokens_processed": 20, "tokens_per_second": 123.4},
        }
    ]

    result = parse_slots_payload("code", 8081, payload, now=20.0)

    assert result.tps == 4.25
    assert result.prompt_tps == 123.4
    assert result.tokens_in == 20
    assert result.tokens_out == 12


def test_parse_slots_payload_ignores_invalid_shape() -> None:
    result = parse_slots_payload("code", 8081, {"bad": "shape"}, now=20.0)

    assert result.tokens_in == 0
    assert result.tokens_out == 0
    assert result.tps is None
    assert result.prompt_tps is None
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: FAIL because `llama_manager.slot_stats` does not exist.

**Step 3: Implement minimal parser**

Add `SlotStatsSnapshot`, `_format_rate()`, `_number()`, `_int_value()`, `_nested()`, `_first_number()`, and `parse_slots_payload()` to `src/llama_manager/slot_stats.py`.

Implementation constraints:

- Use only stdlib.
- Avoid broad response persistence.
- Keep helpers private except `SlotStatsSnapshot` and `parse_slots_payload`.
- Annotate all functions.

**Step 4: Run test to verify parser passes**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/llama_manager/slot_stats.py src/tests/slot/test_slot_stats.py
git commit -m "feat: parse llama slot runtime stats"
```

## Task 2: Add Runtime Stats Persistence

**Files:**

- Modify: `src/llama_manager/slot_stats.py`
- Modify: `src/tests/slot/test_slot_stats.py`

**Step 1: Write failing store tests**

Add tests:

```python
from pathlib import Path

from llama_manager.slot_stats import (
    SlotStatsSnapshot,
    load_slot_stats,
    save_slot_stats,
    slot_stats_file_path,
)


def test_slot_stats_file_path_uses_runtime_dir(tmp_path: Path) -> None:
    assert slot_stats_file_path(tmp_path) == tmp_path / "slot-stats.json"


def test_save_and_load_slot_stats_round_trip(tmp_path: Path) -> None:
    stats = {
        "summary": SlotStatsSnapshot(
            alias="summary",
            port=8080,
            updated_at=10.0,
            tps=4.2,
            prompt_tps=111.0,
            tokens_in=20,
            tokens_out=9,
        )
    }

    save_slot_stats(stats, runtime_dir=tmp_path)

    assert load_slot_stats(runtime_dir=tmp_path) == stats


def test_load_slot_stats_returns_empty_for_missing_or_invalid_file(tmp_path: Path) -> None:
    assert load_slot_stats(runtime_dir=tmp_path) == {}
    slot_stats_file_path(tmp_path).write_text("{not-json", encoding="utf-8")
    assert load_slot_stats(runtime_dir=tmp_path) == {}
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: FAIL because store functions do not exist.

**Step 3: Implement persistence**

Add:

```python
def slot_stats_file_path(runtime_dir: Path | None = None) -> Path:
    base = runtime_dir or resolve_runtime_dir()
    return base / "slot-stats.json"


def load_slot_stats(runtime_dir: Path | None = None) -> dict[str, SlotStatsSnapshot]:
    ...


def save_slot_stats(
    stats_by_alias: dict[str, SlotStatsSnapshot],
    runtime_dir: Path | None = None,
) -> None:
    ...
```

JSON shape:

```json
{
  "version": 1,
  "slots": {
    "summary": {
      "alias": "summary",
      "port": 8080,
      "updated_at": 10.0,
      "tps": 4.2,
      "prompt_tps": 111.0,
      "tokens_in": 20,
      "tokens_out": 9,
      "source": "slots"
    }
  }
}
```

Implementation constraints:

- Use `atomic_write_json()`.
- Create parent directory before write.
- On invalid/missing JSON, return `{}`.
- On individual invalid slot entries, skip only that entry.
- Do not delete or truncate the file on load failure.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/llama_manager/slot_stats.py src/tests/slot/test_slot_stats.py
git commit -m "feat: persist slot runtime stats"
```

## Task 3: Add HTTP Collector

**Files:**

- Modify: `src/llama_manager/slot_stats.py`
- Modify: `src/tests/slot/test_slot_stats.py`

**Step 1: Write failing collector tests**

Add tests with monkeypatches for `urllib.request.urlopen`:

```python
import json
from io import BytesIO
from typing import Any

from llama_manager.slot_stats import collect_slot_stats


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_collect_slot_stats_fetches_slots_endpoint(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: float):
        calls.append(request.full_url)
        return _Response([{"next_token": {"n_decoded": 3}}])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0)

    assert calls == ["http://127.0.0.1:8080/slots"]
    assert result.alias == "summary"
    assert result.tokens_out == 3


def test_collect_slot_stats_returns_none_on_http_failure(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert collect_slot_stats("summary", "127.0.0.1", 8080, now=30.0) is None
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: FAIL because `collect_slot_stats()` does not exist.

**Step 3: Implement collector**

Add:

```python
def collect_slot_stats(
    alias: str,
    host: str,
    port: int,
    *,
    timeout_s: float = 0.2,
    now: float | None = None,
) -> SlotStatsSnapshot | None:
    ...
```

Implementation constraints:

- Use `urllib.request.Request`.
- URL: `http://{host}:{port}/slots`.
- Add `Accept: application/json`.
- Return `None` for `OSError`, `TimeoutError`, invalid JSON, non-list payload.
- Do not log raw body.
- Use `time.time()` when `now is None`.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/llama_manager/slot_stats.py src/tests/slot/test_slot_stats.py
git commit -m "feat: collect live llama slot stats"
```

## Task 4: Wire Stats Cache Into Dashboard Model

**Files:**

- Modify: `src/llama_cli/tui/model.py`
- Modify: `src/llama_cli/tui/viewmodel.py`
- Modify: `src/tests/tui/test_viewmodel.py`

**Step 1: Write failing viewmodel test**

Update `test_column_valid()` in `src/tests/tui/test_viewmodel.py` to seed cached runtime stats and expect displayed values.

Example expectation:

```python
from llama_manager.slot_stats import SlotStatsSnapshot

...
vm = _make_viewmodel(
    configs=[cfg],
    ...
    slot_runtime_stats={
        "my-server": SlotStatsSnapshot(
            alias="my-server",
            port=9000,
            updated_at=10.0,
            tps=5.25,
            prompt_tps=99.9,
            tokens_in=123,
            tokens_out=45,
        )
    },
)
...
assert result.runtime_stats == SlotRuntimeStats(
    tps="5.2",
    pp="99.9",
    tokens_in="123",
    tokens_out="45",
)
```

Add/update `_make_viewmodel()` helper to copy `slot_runtime_stats` into `DashboardModel`.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/tui/test_viewmodel.py::test_column_valid -q
```

Expected: FAIL because `DashboardModel` has no runtime stats cache and viewmodel still hardcodes stats.

**Step 3: Implement model cache**

In `src/llama_cli/tui/model.py`:

- Import `SlotStatsSnapshot`.
- Add `self.cached_slot_stats_by_alias: dict[str, SlotStatsSnapshot] = {}` in `DashboardModel.__init__`.
- Add thread-safe methods using `self.system_health_lock`:

```python
def apply_slot_stats_snapshot(self, stats_by_alias: dict[str, SlotStatsSnapshot]) -> None:
    with self.system_health_lock:
        self.cached_slot_stats_by_alias = dict(stats_by_alias)


def set_cached_slot_stats(self, alias: str, stats: SlotStatsSnapshot) -> None:
    with self.system_health_lock:
        self.cached_slot_stats_by_alias[alias] = stats


def slot_stats_snapshot(self) -> dict[str, SlotStatsSnapshot]:
    with self.system_health_lock:
        return dict(self.cached_slot_stats_by_alias)
```

In `DashboardViewModel.column()`:

- Load cached stats via `self.model.slot_stats_snapshot().get(cfg.alias)`.
- Convert to display with `stats.to_display()` when present.
- Keep current fallback `--`, `--`, `0`, `0`.

**Step 4: Run focused test**

Run:

```bash
uv run pytest src/tests/tui/test_viewmodel.py::test_column_valid -q
```

Expected: PASS.

**Step 5: Run broader TUI viewmodel tests**

Run:

```bash
uv run pytest src/tests/tui/test_viewmodel.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/llama_cli/tui/model.py src/llama_cli/tui/viewmodel.py src/tests/tui/test_viewmodel.py
git commit -m "feat: show cached slot runtime stats"
```

## Task 5: Load Persisted Stats At TUI Startup

**Files:**

- Modify: `src/llama_cli/tui/controller.py`
- Modify: `src/tests/tui/test_textual_app.py` or `src/tests/tui/test_controller_profiles.py`

**Step 1: Write failing startup test**

Add a controller/model test that patches `llama_cli.tui.controller.load_slot_stats`.

Expected behavior:

- When `DashboardController` is constructed, persisted stats are loaded once.
- Loaded values are applied to `controller.model.cached_slot_stats_by_alias`.
- Load failure must not prevent TUI construction.

Example:

```python
from llama_manager.slot_stats import SlotStatsSnapshot


def test_controller_loads_persisted_slot_stats(monkeypatch) -> None:
    persisted = {
        "summary": SlotStatsSnapshot(
            alias="summary",
            port=8080,
            updated_at=10.0,
            tps=4.0,
            tokens_in=100,
            tokens_out=25,
        )
    }
    monkeypatch.setattr("llama_cli.tui.controller.load_slot_stats", lambda: persisted)

    controller = _make_controller()

    assert controller.model.slot_stats_snapshot() == persisted
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/tui/test_textual_app.py::test_controller_loads_persisted_slot_stats -q
```

If the helper lives in a class, use the exact node id from the file.

Expected: FAIL.

**Step 3: Implement startup load**

In `src/llama_cli/tui/controller.py`:

- Import `load_slot_stats`.
- During `DashboardController.__init__`, after `self.model = DashboardModel(...)`, call a private helper:

```python
def _load_persisted_slot_stats(self) -> None:
    try:
        self.model.apply_slot_stats_snapshot(load_slot_stats())
    except Exception:
        logger.debug("failed to load persisted slot stats", exc_info=True)
```

Implementation constraints:

- Use debug logging only.
- Do not notify the TUI user for missing/invalid stats cache.
- Do not call HTTP in this task.

**Step 4: Run startup test**

Run:

```bash
uv run pytest src/tests/tui/test_textual_app.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/llama_cli/tui/controller.py src/tests/tui/test_textual_app.py
git commit -m "feat: restore persisted slot stats in tui"
```

## Task 6: Add Background Stats Refresh + Persistence

**Files:**

- Modify: `src/llama_cli/tui/controller.py`
- Modify: `src/llama_cli/tui/textual_app.py`
- Modify: `src/tests/tui/test_textual_app.py`

**Step 1: Write failing controller refresh tests**

Add tests:

```python
from llama_manager.slot_stats import SlotStatsSnapshot


def test_controller_refresh_slot_stats_collects_running_configs(monkeypatch) -> None:
    collected: list[tuple[str, str, int]] = []
    saved: list[dict[str, SlotStatsSnapshot]] = []

    def fake_collect(alias: str, host: str, port: int):
        collected.append((alias, host, port))
        return SlotStatsSnapshot(alias=alias, port=port, updated_at=10.0, tokens_out=5)

    monkeypatch.setattr("llama_cli.tui.controller.collect_slot_stats", fake_collect)
    monkeypatch.setattr("llama_cli.tui.controller.save_slot_stats", saved.append)
    controller = _make_controller()

    controller.refresh_slot_stats()

    assert collected
    assert controller.model.slot_stats_snapshot()
    assert saved


def test_controller_refresh_slot_stats_keeps_persisted_value_on_failure(monkeypatch) -> None:
    existing = SlotStatsSnapshot(alias="summary", port=8080, updated_at=1.0, tokens_out=3)
    monkeypatch.setattr("llama_cli.tui.controller.collect_slot_stats", lambda *a, **k: None)
    monkeypatch.setattr("llama_cli.tui.controller.save_slot_stats", lambda stats: None)
    controller = _make_controller()
    controller.model.set_cached_slot_stats("summary", existing)

    controller.refresh_slot_stats()

    assert controller.model.slot_stats_snapshot()["summary"] == existing
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest src/tests/tui/test_textual_app.py -q
```

Expected: FAIL because `refresh_slot_stats()` does not exist.

**Step 3: Implement controller refresh**

In `src/llama_cli/tui/controller.py`:

- Import `collect_slot_stats`, `save_slot_stats`.
- Add:

```python
def refresh_slot_stats(self) -> None:
    current = self.model.slot_stats_snapshot()
    updated = dict(current)
    changed = False
    for cfg in self.model.configs:
        stats = collect_slot_stats(cfg.alias, self.config.deployment.host, cfg.port)
        if stats is None:
            continue
        updated[cfg.alias] = stats
        changed = True
    if changed:
        self.model.apply_slot_stats_snapshot(updated)
        save_slot_stats(updated)
```

Implementation constraints:

- Do not clear stats for failed polls.
- Catch/log exceptions around one slot and continue with remaining slots.
- Catch/log `save_slot_stats()` failures; do not crash the TUI.
- Use controller config host; do not hardcode `127.0.0.1`.

**Step 4: Add Textual worker scheduling**

In `src/llama_cli/tui/textual_app.py`:

- Add `_slot_stats_refresh_active: bool` on app init.
- In `on_mount()`, schedule interval near GPU/system refresh:

```python
self.set_interval(1.0, self._schedule_slot_stats_refresh)
```

- Add `_schedule_slot_stats_refresh()`, `_refresh_slot_stats_worker()`, and `_mark_slot_stats_refresh_complete()` mirroring GPU stats pattern.
- Worker should call `self.controller.refresh_slot_stats()` off the UI thread.
- On completion, call `self.call_from_thread(self.refresh_dashboard)` and clear active flag.

**Step 5: Add app scheduling tests**

Add tests mirroring `TestDashboardAppGpuStatsRefresh`:

- Active flag prevents duplicate worker launch.
- Completion clears flag.
- Worker calls controller refresh and then dashboard refresh through `call_from_thread`.

**Step 6: Run focused tests**

Run:

```bash
uv run pytest src/tests/tui/test_textual_app.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src/llama_cli/tui/controller.py src/llama_cli/tui/textual_app.py src/tests/tui/test_textual_app.py
git commit -m "feat: refresh slot runtime stats in tui"
```

## Task 7: Verify Stats Widget Updates From Cache

**Files:**

- Modify: `src/tests/tui/test_tui.py`
- Modify: `src/tests/tui/test_viewmodel.py` if needed

**Step 1: Write widget composition test**

Add/extend a test near `test_server_log_panel_composes_server_column_widget`:

```python
def test_server_column_panel_renders_runtime_stats_values() -> None:
    from llama_cli.tui.components.server_column import ServerColumnPanel
    from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats

    state = ServerColumnState(
        alias="slot-a",
        profile_name="slot-a",
        status="running",
        status_label="Running",
        status_class="server-column-status-running",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="Device: SYCL0 | Ctx: 2048 | Threads: 4",
        log_lines=("Waiting for output...",),
        runtime_stats=SlotRuntimeStats(tps="5.2", pp="99.9", tokens_in="123", tokens_out="45"),
        gpu_stats=None,
        stale_warning=None,
    )

    panel = ServerColumnPanel(state)
    stats_container = list(panel.compose())[2]
    cells = stats_container._pending_children[1]._pending_children

    values = [cell._pending_children[1].renderable for cell in cells]
    assert values == ["5.2", "99.9", "123", "45"]
```

**Step 2: Run focused widget test**

Run:

```bash
uv run pytest src/tests/tui/test_tui.py::TestDashboardComponents::test_server_column_panel_renders_runtime_stats_values -q
```

Expected: PASS after Task 4. If class name differs, use exact node id from the file.

**Step 3: Add update-path test if missing**

Add a test for `_update_panel_widgets()` that mounts/constructs a panel with old stats, calls update with new `ServerColumnState`, and asserts `.slot-stats-value` widgets update in order.

Keep it focused; do not use real HTTP or real subprocesses.

**Step 4: Run TUI component tests**

Run:

```bash
uv run pytest src/tests/tui/test_tui.py src/tests/tui/test_textual_app.py src/tests/tui/test_viewmodel.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/tests/tui/test_tui.py src/tests/tui/test_textual_app.py src/tests/tui/test_viewmodel.py
git commit -m "test: cover slot stats display updates"
```

## Task 8: Manual TUI Verification

**Files:**

- No code changes expected.

**Step 1: Run static checks**

Run:

```bash
uv run ruff check .
```

Expected: PASS.

Run:

```bash
uv run ruff format --check .
```

Expected: PASS.

Run:

```bash
uv run pyright
```

Expected: PASS.

**Step 2: Run focused tests**

Run:

```bash
uv run pytest src/tests/slot/test_slot_stats.py src/tests/tui/test_viewmodel.py src/tests/tui/test_textual_app.py src/tests/tui/test_tui.py -q
```

Expected: PASS.

**Step 3: Run full tests**

Run:

```bash
uv run pytest
```

Expected: PASS.

**Step 4: Dry-run launch**

Run:

```bash
uv run llm-runner dry-run both
```

Expected: PASS; server command still includes `--parallel` and does not need `--metrics`.

**Step 5: Optional live TUI check**

Run:

```bash
uv run llm-runner
```

Expected:

- Stats card initially shows persisted values if `<runtime_dir>/slot-stats.json` exists.
- After one successful poll, values update without layout shift.
- If a slot is offline, TUI does not crash and keeps last-known values.

**Step 6: Final pre-commit gate**

Run the mandatory local gate before any commit/push:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Expected: both PASS.

## Acceptance Criteria

- `Stats` card no longer stays hardcoded at `TPS --`, `PP --`, `Tok In 0`, `Tok Out 0` when `/slots` returns counters.
- TUI restart shows last persisted stats before live HTTP refresh completes.
- Failed stats polling does not clear persisted values.
- Stats collection is off the Textual render path.
- Invalid stats cache JSON is ignored without crashing.
- No prompt/user request content is persisted.
- Focused tests and full pytest pass.
- `uv run pre-commit run --all-files` passes before commit/push.

## Reference

- llama.cpp server README: `GET /slots` is enabled by default and exposes per-slot speed/processed-token state; `GET /metrics` requires `--metrics`.
