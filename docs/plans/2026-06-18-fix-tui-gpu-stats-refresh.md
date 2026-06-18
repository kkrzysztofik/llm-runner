# Fix TUI GPU Stats Refresh

## Goal

Make per-slot GPU stats update in the TUI after the background GPU telemetry worker collects data.

User-visible bug: the slot card shows `Device: N/A`, `GPU ?????????? N/A`, and `VRAM ?????????? N/A` for a running SYCL slot even though the SYCL collector can read the Intel Arc GPU.

## Root Cause

This is not a SYCL collector failure on the current machine. A direct `SYCL0` probe returns real Level Zero stats:

```json
{
  "device": "Intel(R) Arc(TM) B580 Graphics",
  "gpu_util": "99%",
  "mem_util": "57%",
  "vram": "6.8G/11.9G",
  "temp": "85C",
  "power": "238W",
  "source": "level-zero",
  "pci_bdf": "0000:30:00.0"
}
```

The TUI renders the GPU panel from an empty cached snapshot before the worker runs, then never updates that panel during lightweight refreshes.

Relevant flow:

1. `DashboardModel.__init__()` builds `cached_gpu_stats_by_alias` from `gpu.get_cached_stats_snapshot()` at startup.
   - File: `src/llama_cli/tui/model.py`
   - Lines: 56-77
   - At this point each `GPUStats` instance has not run `update()`, so the cached value is `{}`.

2. `DashboardViewModel.column()` passes that cached dict to `ServerColumnState.gpu_stats`.
   - File: `src/llama_cli/tui/viewmodel.py`
   - Lines: 149-169

3. `GPUStatsPanel` treats `{}` as available data and renders missing keys as `N/A`; missing percentages become question-mark meters.
   - File: `src/llama_cli/tui/components/gpu_stats.py`
   - Lines: 21-28, 30-57, 82-120

4. `DashboardApp._refresh_gpu_stats_worker()` correctly runs `gpu.update()` in a worker and applies the new cache with `apply_gpu_stats_snapshot()`.
   - File: `src/llama_cli/tui/textual_app.py`
   - Lines: 172-204

5. `DashboardApp.refresh_dashboard()` calls `_update_panel_widgets()` for each `ServerLogPanel`, but `_update_panel_widgets()` only updates status, profile name, config summary, backend, URL, slot runtime stats, and logs.
   - File: `src/llama_cli/tui/textual_app.py`
   - Lines: 820-902
   - It does not update or recompose `GPUStatsPanel`.

Result: the collector cache becomes correct, but the visible GPU panel stays stuck with the first rendered empty snapshot until a full slot recompose happens.

## Architecture

Keep the current architecture:

- Hardware probes stay off the render thread.
- `DashboardViewModel.column()` must keep using `DashboardSnapshot` cached data.
- `GPUStatsPanel` should remain a pure render widget from `dict[str, Any] | None`.
- Lightweight dashboard refresh should update the existing panel; do not force full slot recomposition every tick.

Minimal fix:

1. Add a public update method to `GPUStatsPanel`.

```python
def update_stats(self, stats: dict[str, Any] | None) -> None:
    self._stats = dict(stats) if stats is not None else None
    self.refresh(recompose=True)
```

Optional guard:

```python
next_stats = dict(stats) if stats is not None else None
if self._stats == next_stats:
    return
self._stats = next_stats
self.refresh(recompose=True)
```

2. Import `GPUStatsPanel` in `src/llama_cli/tui/textual_app.py`.

3. In `_update_panel_widgets()`, query the panel and call `update_stats(state.gpu_stats)`.

```python
with contextlib.suppress(NoMatches):
    gpu_panel = cast(GPUStatsPanel, panel.query_one(GPUStatsPanel))
    gpu_panel.update_stats(state.gpu_stats)
```

Place this before runtime stats/log updates so the visual telemetry is refreshed during the same periodic pass.

## Scope

Do:

- Update the visible per-slot GPU stats panel when cached telemetry changes.
- Add regression tests proving cache updates reach the widget.
- Preserve background collection and cached snapshot behavior.

Do not:

- Rewrite the Level Zero, `nvtop`, `xpu-smi`, or `nvidia-smi` collectors.
- Add hardware probing to render/view-model code.
- Full-recompose all server columns every refresh tick.
- Edit Speckit files.
- Delete files.

## Detailed Tasks

- [ ] **Task 1: Add a unit test for `GPUStatsPanel.update_stats()`.**
  - File: `src/tests/tui/test_tui.py`
  - Add a test near existing `GPUStatsPanel` tests around lines 339-563.
  - Create `panel = GPUStatsPanel({})`.
  - Patch/mock `panel.refresh`.
  - Call `panel.update_stats({"device": "Intel Arc", "gpu_util": "45%", "mem_util": "57%"})`.
  - Assert `_stats` changed to the new dict.
  - Assert `refresh(recompose=True)` was called once.
  - Also test `update_stats(None)` if the implementation supports unavailable state explicitly.

- [ ] **Task 2: Add an app/widget regression test for `_update_panel_widgets()`.**
  - File: `src/tests/tui/test_textual_app.py`
  - Use the existing `DashboardApp` test style near `TestDashboardAppGpuStatsRefresh`.
  - Build a mounted `DashboardApp` with one config and initial cached GPU stats `{}`.
  - Apply a cached snapshot like:
    `{"slot0": {"device": "Intel Arc", "gpu_util": "45%", "mem_util": "57%", "temp": "67C", "power": "120W"}}`.
  - Call `refresh_dashboard()` or `_update_panel_widgets()` with a `ServerColumnState` containing that snapshot.
  - Assert the mounted `GPUStatsPanel` receives the new stats and recomposes, or assert rendered `.gpu-stats-value` text eventually contains `Intel Arc` and `45%`.
  - Pass condition: the test fails on current code because `_update_panel_widgets()` never touches `GPUStatsPanel`.

- [ ] **Task 3: Implement `GPUStatsPanel.update_stats()`.**
  - File: `src/llama_cli/tui/components/gpu_stats.py`
  - Keep the method small.
  - Copy incoming dicts to avoid later mutation leaking into widget state.
  - Do not call collectors or view-model methods from this widget.

- [ ] **Task 4: Wire the update into lightweight refresh.**
  - File: `src/llama_cli/tui/textual_app.py`
  - Add `from .components.gpu_stats import GPUStatsPanel`.
  - In `_update_panel_widgets()`, query `GPUStatsPanel` and call `update_stats(state.gpu_stats)`.
  - Keep all existing `contextlib.suppress(NoMatches)` behavior.

- [ ] **Task 5: Optional first-frame improvement.**
  - Consider initializing `cached_gpu_stats_by_alias` with `None` or a psutil fallback instead of `{}`.
  - This is optional and lower priority.
  - If done, update types first because `DashboardSnapshot.gpu_stats_by_alias` currently requires `dict[str, dict[str, Any]]`.
  - Safer MVP: skip this and only fix widget refresh.

- [ ] **Task 6: Verification.**
  - Run targeted tests first:

```bash
uv run pytest src/tests/tui/test_tui.py src/tests/tui/test_textual_app.py
```

  - Run full local gate before any commit:

```bash
uv run pre-commit run --all-files
uv run pytest
```

  - Per repo instructions, pipe command output through `distill` when run by an agent, for example:

```bash
rtk proxy bash -lc 'uv run pytest src/tests/tui/test_tui.py src/tests/tui/test_textual_app.py 2>&1 | distill "Did targeted TUI tests pass? Return PASS/FAIL and failing test names."'
```

## Pass Criteria

- The TUI per-slot GPU panel changes from startup `N/A`/question-mark values to real cached telemetry after the background GPU worker runs.
- No GPU collector runs from render/view-model code.
- `DashboardViewModel.column()` continues to use cached snapshots only.
- Existing GPU collector tests still pass.
- New regression test fails on the old code and passes after the widget update wiring.

## Risk Notes

- Recompose only the small `GPUStatsPanel`, not the whole server column, to avoid log widget churn.
- The worker interval is at least one second by design (`max(1.0, interval_s)` in `DashboardApp.on_mount()`), so telemetry is not expected to update faster than that.
- The collector stack is already working for `SYCL0` on this host via Level Zero. Avoid changing collector order unless a separate hardware-specific bug is found.

