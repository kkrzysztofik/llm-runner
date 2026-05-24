# Plan: Log Storage & Extended Logging

**Spec:** `002-logging`
**Status:** Approved — pending implementation
**Date:** 2026-05-24

---

## Problem Statement

- No log file is written by default — user must manually set `LLM_RUNNER_LOG_FILE` env var
- Log levels (file + stderr) are not configurable from the TUI — require env var or restart
- 8 operational areas produce zero log output, creating blind spots under the hood:
  GPU fallbacks, slot state transitions, lock lifecycle, profile staleness, TUI status pushes,
  smoke probe events, server launch outcomes, and the `print()` fallback in manager

---

## Goals

1. **Auto-write a log file** on every run under XDG state dir — no user config required
2. **Expose file + stderr log levels** in the TUI Config tab — live update, persisted to `config.toml`
3. **Instrument 8 silent areas** with structured log calls at appropriate levels
4. **Retain existing behaviour** — `LLM_RUNNER_LOG_FILE` env var still overrides auto-path;
   `LLM_RUNNER_LOG_LEVEL` still sets initial stderr level

---

## Design Decisions

### Log File Path

```
$XDG_STATE_HOME/llm-runner/logs/llm-runner-<YYYYMMDD-HHMMSS>.log
```

- Fallback when `XDG_STATE_HOME` unset: `~/.local/state/llm-runner/logs/`
- Directory created automatically at startup (`mkdir(parents=True, exist_ok=True)`)
- `LLM_RUNNER_LOG_FILE` env var overrides auto-path if set
- Rotation: 10 MB / 30 days / gzipped (unchanged from existing file sink config)

### Audit Log Path

```
$XDG_STATE_HOME/llm-runner/logs/audit-<YYYYMMDD-HHMMSS>.log
```

- Auto-wired to `ServerManager._audit_log_path` in `_run_tui()` (currently `None` → unwired)

### Log Level Split

| Sink   | Config field        | Default | Live-updatable |
|--------|---------------------|---------|----------------|
| File   | `log_file_level`    | `DEBUG` | Yes (TUI)      |
| Stderr | `log_stderr_level`  | `INFO`  | Yes (TUI)      |

- Both levels persisted in `~/.config/llm-runner/config.toml`
- Initial values loaded from persisted config at `cli_main()` startup
- Loguru sink IDs stored at module level; live update via `logger.remove(id)` + re-add

### GPU Poll Logging

- Log only when util/mem/temp changes by >5% (change-triggered, not every 500ms tick)
- Prevents ~2 lines/sec noise at DEBUG level

### Structured Extras

All new log calls use Loguru keyword args for JSON mode compatibility:

```python
logger.info("[slot] transition", slot_id=slot_id, old_state=old, new_state=new)
```

---

## Files to Change

| # | File | Changes |
|---|------|---------|
| 1 | `src/llama_manager/config/defaults.py` | Add `logs_dir` property; add `log_file_level: str = "DEBUG"` + `log_stderr_level: str = "INFO"` fields |
| 2 | `src/llama_manager/config/persistence.py` | Add `"log_file_level"` + `"log_stderr_level"` to `_PERSISTED_FIELDS` |
| 3 | `src/llama_manager/logging_setup.py` | Split `level` → `stderr_level` + `file_level`; add `update_file_log_level()` + `update_stderr_log_level()` live-update funcs; track sink IDs at module level |
| 4 | `src/llama_cli/server_runner.py` | Load persisted config first in `cli_main()`; auto-compute log path under `logs_dir`; pass `stderr_level`/`file_level` from config; wire audit log path |
| 5 | `src/llama_cli/tui/components/config_modal.py` | Add `log_file_level` + `log_stderr_level` to `ConfigPayload`; add "Logging" section with two Select widgets (DEBUG/INFO/WARNING/ERROR) |
| 6 | `src/llama_cli/tui/controller.py` | In `save_config()`: call `update_file_log_level()` + `update_stderr_log_level()` when changed; add `logger.debug("[tui] status: {msg}")` in `_push_status_message()` |
| 7 | `src/llama_manager/gpu_stats.py` | Add `DEBUG` on nvtop fallback; add change-triggered `DEBUG` on GPU poll (>5% delta threshold) |
| 8 | `src/llama_manager/orchestration/slot_manager.py` | Add `INFO` on every `SlotRuntime.transition_to()` call |
| 9 | `src/llama_manager/orchestration/manager.py` | Add `INFO` on slot launch success; `WARNING` on failure; replace `print()` fallback (line 670) with `logger.warning()` |
| 10 | `src/llama_manager/orchestration/lockfile.py` | Add `DEBUG` on clean lock release |
| 11 | `src/llama_manager/profile_cache.py` | Add `DEBUG` on staleness check result |
| 12 | `src/llama_cli/commands/smoke.py` | Add `INFO` probe start / result / timeout lifecycle events |
| 13 | `src/tests/test_logging_setup.py` | Tests for split-level params + live-update functions |

---

## Extended Logging — Silent Areas Detail

### Area 1: GPU stats — nvtop fallback (`gpu_stats.py`)

```python
# collect_nvtop_stats() fallback path
logger.debug("[gpu] nvtop probe failed: {exc}; falling back to psutil", exc=exc)

# per-poll change-triggered (>5% delta on util, mem, or temp)
logger.debug("[gpu] poll", idx=idx, util_pct=util, mem_mb=mem, temp_c=temp)
```

### Area 2: Slot state transitions (`slot_manager.py`)

```python
# SlotRuntime.transition_to()
logger.info("[slot] transition", slot_id=self.slot_id, old_state=old, new_state=new)
```

### Area 3: Server launch outcome (`manager.py`)

```python
# success path in _process_slot
logger.info("[manager] slot started", slot_id=slot_id, pid=pid, port=port, model=model)

# failure path
logger.warning("[manager] slot failed", slot_id=slot_id, error_code=code, detail=detail)

# replace print() fallback at line ~670
logger.warning("[manager] no log_handler — output routed to stderr", slot_id=slot_id)
```

### Area 4: Lock release (`lockfile.py`)

```python
# release_lock() clean path
logger.debug("[lock] released", path=str(path))
```

### Area 5: Profile staleness (`profile_cache.py`)

```python
# StalenessResult computation
logger.debug("[profile] staleness", slot=slot, result=result, reason=reason)
```

### Area 6: TUI status push (`controller.py`)

```python
# _push_status_message()
logger.debug("[tui] status", message=message)
```

### Area 7: Smoke probe lifecycle (`commands/smoke.py`)

```python
logger.info("[smoke] probe start", slot=slot, model=model)
logger.info("[smoke] probe result", slot=slot, status=status, latency_ms=latency)
logger.warning("[smoke] probe timeout", slot=slot, timeout_s=timeout)
```

---

## TUI Config Tab — Logging Section

New "Logging" section added to `ConfigModal`, below "Smoke Probes":

```
┌─ Logging ─────────────────────────────────────────┐
│  File log level    [ DEBUG ▼ ]                     │
│  Stderr log level  [ INFO  ▼ ]                     │
└───────────────────────────────────────────────────┘
```

Select options: `DEBUG`, `INFO`, `WARNING`, `ERROR`

Changes take effect immediately on "Save" (no restart required for level changes only).

---

## config.toml Schema Addition

```toml
log_file_level = "DEBUG"
log_stderr_level = "INFO"
```

---

## Test Coverage

`tests/test_logging_setup.py` additions:

- `test_split_levels_stderr_info_file_debug` — verify two sinks get different levels
- `test_update_file_log_level_live` — verify `update_file_log_level("WARNING")` suppresses DEBUG in file
- `test_update_stderr_log_level_live` — verify `update_stderr_log_level("ERROR")` suppresses INFO on stderr
- `test_invalid_level_raises` — verify `ValueError` on bad level string for both update funcs

---

## Out of Scope

- Log shipping / remote log aggregation
- Structured log ingestion pipeline
- Log viewer in TUI (separate feature)
- Changing rotation policy (10 MB / 30 days kept as-is)
