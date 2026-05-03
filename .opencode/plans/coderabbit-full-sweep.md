# CodeRabbit Full Sweep — Implementation Plan

> **Source:** 112 findings from `/home/kmk/llm-runner/out.json` (CodeRabbit review on `integration` vs `main`)
> **Created:** 2026-05-03
> **Branch:** `integration` → `main`

---

## ⚠️ CRITICAL FINDINGS — Must Fix First

These are security-critical or data-correctness bugs that can cause **unauthorized destructive actions** or **silent data corruption**. Fix these before everything else.

| # | File | Finding | Severity |
|---|------|---------|----------|
| 1 | `src/llama_cli/commands/doctor.py` (L711-715) | **Confirmation bypass** — `--yes` flag is ignored; destructive `requires_confirmation=True` actions run unconditionally when `dry_run=False`. | **CRITICAL** |
| 2 | `src/llama_manager/process_manager.py` (L1004-1013) | **Risk evaluation ignored** — `evaluate_risks(...)` result is not checked before `server_manager.launch_all_slots()`. Launch proceeds regardless of risk assessment. | **CRITICAL** |
| 3 | `src/llama_cli/commands/smoke.py` (L54-57) | **Config divergence** — `"qwen35-coding"` entry uses `cfg.model_qwen35_both` in the smoke/both path and `cfg.model_qwen35` in the slot path, causing divergent runtime behavior. | **MAJOR** |

---

## Summary by Severity

| Severity | Count | Files Affected |
|----------|-------|----------------|
| **Critical** | 3 | doctor.py, process_manager.py, smoke.py |
| **Major** | 13 | setup.py, smoke.py, doctor.py, cli_parser.py, test_reports.py, test_dry_run_flag_bundles.py, smoke_cli_cases.py, test_lock_integrity.py, test_foundation_contracts.py, test_launch_degraded_vs_blocked.py, test_launch_no_autobuild, smoke_cases.py, config_cases.py, test_build_lock_backend_values |
| **Minor** | 24 | toolchain.py, profile_status.py, security.py, risk.py, profile.py, build.py, lock.py, builder.py, risk_ack.py, file_ops.py, _binary.py, smoke_cases.py, support/runtime.py, test_profile_cli.py |
| **Trivial** | 72 | 40+ files across all modules |

---

## Summary by Module

| Module | Critical | Major | Minor | Trivial | Total |
|--------|----------|-------|-------|---------|-------|
| `src/llama_manager/` | 1 | 3 | 5 | 22 | **31** |
| `src/llama_cli/` | 2 | 7 | 6 | 19 | **34** |
| `src/tests/` | 0 | 8 | 7 | 25 | **40** |
| `QUICKSTART.md` | 0 | 0 | 0 | 1 | **1** |

---

## Agent Assignments

| Agent | Scope | Files | Total Findings |
|-------|-------|-------|----------------|
| **Python Backend** | `src/llama_manager/` | 22 files | 31 |
| **TUI Developer** | `src/llama_cli/` + `QUICKSTART.md` | 15 files | 35 |
| **Python QA** | `src/tests/` | 31 files | 40 |
| **Documentation** | `QUICKSTART.md` | 1 file | 1 |

---

# SECTION A — Python Backend (`src/llama_manager/`)

> **Assignee:** Python Backend agent
> **Total findings:** 31 (1 critical, 3 major, 5 minor, 22 trivial)
> **Files:** 22

---

## A.1 Critical (1 finding)

### `src/llama_manager/process_manager.py` — Risk evaluation ignored

**Finding #100** (L1004-1013)

**Problem:** After calling `evaluate_risks(...)`, the result is stored but never checked. `server_manager.launch_all_slots(slots)` is called unconditionally, ignoring risk assessment outcomes.

**Fix:**
```python
# After evaluate_risks(...) call:
risk_result = evaluate_risks(...)
# Check for blocking condition before launching
if risk_result.block_launch or risk_result.blocked or risk_result.unacknowledged_risks:
    # Abort orchestration — do NOT call launch_all_slots
    return risk_result with launch_attempt_id/ack_token
# Only proceed when safe
server_manager.launch_all_slots(slots)
```

**Verification:**
- `evaluate_risks` return type has a blocking indicator — check the class
- Ensure `launch_attempt_id` and `ack_token` are returned even on abort
- This is the **single most important fix** in the entire sweep

---

## A.2 Major (3 findings)

### A.2.1 `src/llama_manager/build_pipeline/stages/finalize.py` — Duplicate artifact serialization

**Finding #83** (L76-107)

**Problem:** `write_provenance` manually constructs `artifact_data` dict, duplicating `BuildArtifact.to_dict()` logic. Broad `except Exception` swallows all errors.

**Fix:**
- Replace manual `artifact_data` construction with `artifact.to_dict()`
- Tighten except clause to `except (OSError, ValueError, json.JSONDecodeError) as e:`
- Keep `logger.warning` in the except block

### A.2.2 `src/llama_manager/config/profile_cache.py` — Symlink vulnerability

**Finding #88** (L362-374)

**Problem:** `Path.resolve()` follows symlinks, allowing path traversal via symlinks inside `profiles_dir`.

**Fix:**
```python
real_candidate = os.path.realpath(profiles_dir / filename)
real_profiles = os.path.realpath(profiles_dir)
if not str(real_candidate).startswith(str(real_profiles) + os.sep):
    raise ValueError("Invalid profile path")
```

### A.2.3 `src/llama_manager/config/builder.py` — Port validation range mismatch

**Finding #53** (L58-60)

**Problem:** `_validate_resolved_profile_data` uses a different lower bound for port validation than `_validate_merged_config` (should both be `1024 <= port <= 65535`).

**Fix:** Align the lower bound check in `_validate_resolved_profile_data` to `1024`.

---

## A.3 Minor (5 findings)

### A.3.1 `src/llama_manager/toolchain.py` — Empty string from _extract_version

**Finding #21** (L213-222)

**Problem:** `_extract_version` can return `""` when `output == ""`, causing callers like `_try_tool` to get an empty string instead of `None`.

**Fix:** Change fallback to `return None if not output.strip() else output.split("\n")[0].strip()`

### A.3.2 `src/llama_manager/common/security.py` — AUTH_HEADER not redacted

**Finding #29** (L51-54)

**Problem:** `_LOG_SENSITIVE_PATTERN` lists `AUTH` but not `AUTH_HEADER`, so tokens like `"AUTH_HEADER=..."` are not redacted.

**Fix:** Update pattern alternation to `AUTH_HEADER|AUTH` (matching `SENSITIVE_WORD_PATTERN`).

### A.3.3 `src/llama_manager/risk_ack.py` — Mutating input ServerConfig objects

**Finding #43** (L59-61)

**Problem:** Loop mutates input `ServerConfig` objects by appending to `cfg.risky_acknowledged`.

**Fix:** Make shallow copies:
```python
from copy import copy
new_config = copy(cfg)
new_config.risky_acknowledged = list(cfg.risky_acknowledged)
new_config.risky_acknowledged.append(RISK_ACK_LABEL)
```

### A.3.4 `src/llama_manager/common/file_ops.py` — Type mixing in os.replace

**Finding #52** (L103)

**Problem:** `tmp_path` is `str` (from `mkstemp`) while `path` is `Path`.

**Fix:** `tmp_path = Path(tmp_path)` immediately after `mkstemp` call.

### A.3.5 `src/llama_manager/metadata/_binary.py` — Unsigned int misinterpretation

**Finding #91** (L266-283)

**Problem:** `_read_integer_value` maps unsigned GGUF type tags (0, 2, 4, 10) to signed readers, causing large unsigned values to be misinterpreted.

**Fix:** Add `_read_uint8`, `_read_uint16`, `_read_uint32`, `_read_uint64` and update the `readers` mapping:
- Tags 0, 2, 4, 10 → unsigned readers
- Tags 1, 3, 5, 11 → signed readers (unchanged)

---

## A.4 Trivial (22 findings)

### A.4.1 `src/llama_manager/config/builder.py` — 6 findings

| # | Lines | Fix |
|---|-------|-----|
| 45 | L228-265 | Add optional `registry=None` param to `create_summary_balanced_cfg`, `create_summary_fast_cfg`, `create_qwen35_cfg` for caching |
| 49 | L50-74 | Replace `isinstance(layer, dict)` with `isinstance(layer, Mapping)` and add `StalenessResult` branch |
| 56 | L43-44 | Update `_deep_merge` docstring to clarify list concatenation semantics (or change to replacement if intended) |
| 54 | L683-691 | Remove unnecessary empty `override_dict`; pass `None` directly to `merge_config_overrides` |
| 60 | L659-663 | Replace `except Exception` with specific exceptions: `except (OSError, FileNotFoundError, ValueError, KeyError)` |
| 53 | L58-60 | Port validation range alignment (see Major A.2.3) |

### A.4.2 `src/llama_manager/config/defaults.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 47 | L127 | Replace `profile_staleness_days: int = field(default_factory=lambda: 30)` with `profile_staleness_days: int = 30`; remove unused `field` import if no longer needed |

### A.4.3 `src/llama_manager/config/server.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 24 | L76-85 | Change `seen: dict[str, int] = {}` to `seen: set[str] = set()` with `seen.add(normalized)` |

### A.4.4 `src/llama_manager/build_pipeline/stages/clone.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 20 | L132-136 | Consolidate duplicate `except subprocess.SubprocessError` into single `except Exception as e:` |
| 22 | L109-116 | Add `timeout=120` to `subprocess.run()` and catch `subprocess.TimeoutExpired` |

### A.4.5 `src/llama_manager/build_pipeline/stages/configure.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 31 | L65-67 | Add `timeout` to `subprocess.run()` and catch `subprocess.TimeoutExpired` |

### A.4.6 `src/llama_manager/build_pipeline/stages/build.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 78 | L91-98 | Replace `proc.kill()` with `proc.terminate()`, wait, then `proc.kill()` if still running. Close `proc.stdout`/`proc.stderr`, join reader threads. |

### A.4.7 `src/llama_manager/build_pipeline/lock.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 87 | L104-115 | Replace `os.kill(pid, 0)` with `psutil.pid_exists(pid)` for PID validation |
| 77 | L127-134 | Replace bare `except` with `except (OSError, json.JSONDecodeError, KeyError, TypeError)` |

### A.4.8 `src/llama_manager/build_pipeline/_context.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 80 | L51-64 | Add uniqueness suffix to `write_build_log` filename (e.g., `uuid.uuid4()` or `time.time_ns()`) |

### A.4.9 `src/llama_manager/build_pipeline/models.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 61 | L102-124 | Replace manual `BuildArtifact.to_dict()` with `dataclasses.asdict(self)`, converting `Path` instances to strings via a helper |

### A.4.10 `src/llama_manager/build_pipeline/orchestration.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 84 | L54 | Add `build_shallow_clone: bool = True` field to `Config`; replace hardcoded `shallow_clone=True` with `config.build_shallow_clone` |

### A.4.11 `src/llama_manager/metadata/_binary.py` — 3 findings

| # | Lines | Fix |
|---|-------|-----|
| 75 | L140-169 | Replace manual two's-complement with `int.from_bytes(data[offset:offset+size], "little", signed=True)` |
| 76 | L85-116 | Fix misleading comment: `_ARCH_PATTERNS` is not strictly sorted longest-first |
| 91 | L266-283 | Add unsigned readers (see Minor A.3.5 above) |

### A.4.12 `src/llama_manager/metadata/_reader.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 62 | L108-173 | Replace single `src.read(prefix_cap_bytes)` with chunked loop (64KB chunks) to avoid huge buffer allocation |

### A.4.13 `src/llama_manager/metadata/__init__.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 71 | L22-31 | Remove private helpers (`_detect_gguf_version`, `_parse_architecture`, etc.) from `__all__` |

### A.4.14 `src/llama_manager/metadata/_types.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 82 | L18-19 | Pre-compile regexes in `normalize_filename`: `_WHITESPACE_PATTERN = re.compile(r"\s+")`, `_UNDERSCORE_PATTERN = re.compile(r"_+")` |

### A.4.15 `src/llama_manager/slot_manager.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 48 | L106 | Replace inline lambda with explicit two-step: `log_buffer = state["log_buffers"][alias]; log_handler = lambda line, buf=log_buffer: buf.add_line(line)` |

---

# SECTION B — TUI Developer (`src/llama_cli/` + `QUICKSTART.md`)

> **Assignee:** TUI Developer agent
> **Total findings:** 35 (2 critical, 7 major, 6 minor, 19 trivial, 1 docs)
> **Files:** 15

---

## B.1 Critical (2 findings)

### B.1.1 `src/llama_cli/commands/doctor.py` — Confirmation bypass

**Finding #92** (L711-715)

**Problem:** `cmd_doctor_repair` runs all actions unconditionally when `dry_run=False`, ignoring the parsed `--yes` flag. Destructive actions with `requires_confirmation=True` execute without consent.

**Fix:**
- Add confirmation gating in `_execute_repair_actions` and `_execute_repair_action`
- Check `dry_run or args.yes or action.requires_confirmation` before executing destructive actions
- For non-`--yes` mode, prompt interactively or skip
- Apply to all execution sites (L746-775, L847-851)

### B.1.2 `src/llama_cli/commands/smoke.py` — Config divergence

**Finding #68** (L54-57)

**Problem:** `"qwen35-coding"` entry uses `cfg.model_qwen35_both` in smoke/both path and `cfg.model_qwen35` in slot path.

**Fix:** Pick one canonical attribute (e.g., `model_qwen35_both`), update all `"qwen35-coding"` references to use it consistently.

---

## B.2 Major (7 findings)

### B.2.1 `src/llama_cli/commands/doctor.py` — RepairAction composite command

**Finding #99** (L601-616)

**Problem:** `RepairAction` with `action_type "remove_and_create_directory"` builds a single command list containing `"&&"` which fails with `subprocess.run(shell=False)`.

**Fix:** Split into two separate `RepairAction` entries:
1. `action_type "remove_file_or_directory"`, command `["rm", "-rf", str(dir_path)]`
2. `action_type "create_directory"`, command `["mkdir", "-m", "700", "-p", str(dir_path)]`

### B.2.2 `src/llama_cli/commands/doctor.py` — Default subcommand returns None

**Finding #101** (L860-889)

**Problem:** `parser.set_defaults(func=lambda args: parser.print_help())` returns `None`, causing `main()` to return `None`.

**Fix:**
```python
parser.set_defaults(func=lambda args: (parser.print_help(), 1)[1])
```
Remove dead `hasattr(parsed, "func")` branch.

### B.2.3 `src/llama_cli/commands/setup.py` — Duplicate backend conversion

**Finding #30** (L153-175)

**Problem:** Both `_backend_from_string` and `_resolve_backend_enum` exist, doing the same thing.

**Fix:** Keep one (e.g., `_backend_from_string`), delete the other, update all references.

### B.2.4 `src/llama_cli/commands/smoke.py` — Duplicate slot mappings

**Finding #15** (L83-94)

**Problem:** Separate slot mappings in `_resolve_slot_port` and `_resolve_slot_model` violate DRY.

**Fix:** Create single canonical slot registry: `dict[slot_id -> {port, model}]`. Both resolver functions and `_build_slot_configs` look up from this single source.

### B.2.5 `src/llama_cli/commands/smoke.py` — Manual __post_init__ call

**Finding #16** (L208-210)

**Problem:** `smoke_cfg.__post_init__()` called manually after mutating fields — fragile pattern.

**Fix:** Use `dataclasses.replace()` or a builder/factory to produce a new validated instance.

### B.2.6 `src/llama_cli/cli_parser.py` — Short flags treated as TUI mode

**Finding #103** (L431-445)

**Problem:** `-p` and `-P` are treated as TUI modes because the conditional only excludes long flags.

**Fix:** Add `"-p"` and `"-P"` to the exclusion tuple in the mode-detection conditional.

### B.2.7 `src/llama_cli/server_runner.py` — Return type + duplicate function

| # | Lines | Fix |
|---|-------|-----|
| 81 | L134-148 | Add precise return type to `_build_tui_mode_configs`: `Dict[str, Tuple[List[str], List[str], List[ServerConfig], List[Optional[int]]]]` |
| 89 | L183-187 | Remove local `_gpu_index_for_config`; import and use `slot_manager.gpu_index_for_config(server_cfg)` |

---

## B.3 Minor (6 findings)

### B.3.1 `src/llama_cli/commands/profile.py` — _resolve_bench_bin silent incorrect path

**Finding #67** (L381-384)

**Problem:** `server_bin.replace("llama-server", "llama-bench")` can silently produce incorrect paths.

**Fix:** Use `pathlib.Path` to inspect basename; only swap when basename matches known variants; return `None` if no suitable match.

### B.3.2 `src/llama_cli/tui/components/profile_status.py` — Trailing newline + missing else

**Findings #23, #27** (L19-33)

**Problem 1:** Loop appends `\n` after every entry, creating trailing blank line.
**Problem 2:** Unknown statuses render as blank.

**Fix:**
```python
first = True
for alias, status in profile_status.items():
    if not first:
        text.append("\n")
    first = False
    # ... existing rendering ...
else:
    text.append("? ", style="dim")
    text.append(f"Profile {alias}: {flavor} ", style="dim")
    text.append(f"[{status}]", style="dim")
```

### B.3.3 `src/llama_cli/tui/components/risk.py` — Hardcoded risk message + weak typing

**Findings #36, #63** (L27-32, L10)

**Problem 1:** `acknowledged()` hardcodes `"privileged ports, non-loopback bind"` — inaccurate for VRAM risks.
**Problem 2:** `kind` parameter typed as `str` instead of `Literal["vram", "hardware"]`.

**Fix 1:** Accept `risk_description` parameter; render `"Risky operation(s) acknowledged: {risk_description}"`.
**Fix 2:** `def required(self, kind: Literal["vram", "hardware"] = "hardware") -> Panel`.

### B.3.4 `src/llama_cli/tui/components/system_health.py` — cpu_percent + dead code

**Findings #40, #51** (L19, L121-123)

**Problem 1:** `psutil.cpu_percent(interval=None)` returns unreliable data on first call.
**Problem 2:** `_usage_bar` is dead code.

**Fix 1:** Prime with `psutil.cpu_percent(interval=0.1)` first call, then `interval=None`.
**Fix 2:** Delete `_usage_bar` method.

### B.3.5 `src/llama_cli/tui/components/system_health.py` — Task stats caching

**Finding #39** (L202-212)

**Problem:** `_get_task_stats` iterates all processes on every call.

**Fix:** Add 1-2s TTL cache: store `last_result` and `last_fetch_ts`; return cached if within TTL.

### B.3.6 `src/llama_cli/tui/components/notices.py` — Mutating state during render

**Finding #41** (L42-48)

**Problem:** `render()` calls `add_class`/`remove_class`, triggering render loops.

**Fix:** Move class toggling to a reactive watcher on `self._view_model.system_notices()`. Keep `render()` pure.

---

## B.4 Trivial (19 findings)

### B.4.1 `src/llama_cli/commands/dry_run.py` — 3 findings

| # | Lines | Fix |
|---|-------|-----|
| 72 | L28 | Remove local `RISK_ACK_LABEL = "warning_bypass"`; import from `llama_manager.risk_ack` |
| 70 | L139-156 | Change `slot_payloads: list[Any]` to `slot_payloads: list[DryRunSlotPayload]` |
| 37 | L28-32 | Stop importing private `_ensure_report_dir`; either make it public or replicate inline |

### B.4.2 `src/llama_cli/tui/constants.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 12 | L1-7 | Add `Final[str]` annotations to `RISK_ACK_LABEL`, `RISK_CONFIRM_PROMPT`, `STATUS_PREFIX`, `STYLE_BOLD_RED`, `STYLE_BOLD_YELLOW` |

### B.4.3 `src/llama_cli/tui/textual_app.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 13 | L88-96 | Cache `_profile_registry` / `_profile_options_cache` on `TextualApp`; invalidate when `controller.config` changes |

### B.4.4 `src/llama_cli/tui/controller.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 34 | L71-72 | Add `register_signals: bool = True` to `__init__`; wrap `signal.signal()` calls in `if register_signals` block |

### B.4.5 `src/llama_cli/tui/components/slot_status.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 18 | L99-103 | Call `buffer.get_lines()` once: `lines = buffer.get_lines()` |
| 25 | L43-52 | Add explicit `if hasattr(proc, "poll"):` vs `else:` for process-like vs PID-based objects |

### B.4.6 `src/llama_cli/tui/components/modal.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 38 | L93-100 | Add numeric validation to port Input: `type="integer"` or validate in `_collect_values` |

### B.4.7 `src/llama_cli/tui/viewmodel.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 90 | L111-132 | Change `_get_driver_version` → `get_driver_version` (promoted to public API) |

### B.4.8 `src/llama_cli/commands/smoke.py` — 2 findings (trivial)

| # | Lines | Fix |
|---|-------|-----|
| 15 | L83-94 | Consolidate slot mappings (see Major B.2.4 above) |
| 16 | L208-210 | Replace manual `__post_init__` (see Major B.2.5 above) |

### B.4.9 `QUICKSTART.md` — 1 finding (Documentation)

| # | Lines | Fix |
|---|-------|-----|
| 11 | L48-51 | Change `config_builder.py` to `llama_manager/config/builder.py` or `llama_manager.config.builder` |

---

# SECTION C — Python QA (`src/tests/`)

> **Assignee:** Python QA agent
> **Total findings:** 40 (0 critical, 8 major, 7 minor, 25 trivial)
> **Files:** 31

---

## C.1 Major (8 findings)

### C.1.1 `src/tests/smoke/smoke_cases.py` — _probe_models patched to return final result

**Finding #95** (L336-590)

**Problem:** Tests patch `_probe_models` to return final `SmokeProbeResult`, bypassing HTTP-to-status mapping logic.

**Fix:** Patch the underlying HTTP client (`httpx.Client.request` or `httpx.get`) to simulate real responses (status codes, `TimeoutException`, `ConnectError`, invalid JSON). Remove the `side_effect` that returns final `SmokeProbeResult`.

### C.1.2 `src/tests/config/config_cases.py` — test_launch_no_autobuild never calls real launch

**Finding #96** (L937-967)

**Problem:** Test only checks local `if` and asserts on a mock the production code never sees.

**Fix:** Invoke the actual launch entrypoint that constructs `BuildPipeline`. Assert `mock_pipeline_instance.run.assert_not_called()`.

### C.1.3 `src/tests/config/config_cases.py` — vacuous uniqueness assertion

**Finding #94** (L538-561)

**Problem:** `assert len(unique_filenames) >= 1` doesn't verify uniqueness.

**Fix:** Either assert `len(paths) == len(unique_filenames)` after increasing inter-write delay, or rename test to reflect documented behavior.

### C.1.4 `src/tests/runtime/test_lock_integrity.py` — improbable PIDs

**Findings #108, #109** (L29-80, L326-331)

**Problem:** Tests use real `subprocess.Popen` to get stale PIDs that almost certainly don't exist.

**Fix #108:** Mock `psutil.pid_exists` to return `False` for fake PIDs.
**Fix #109:** Replace `subprocess.Popen` with `stale_pid = 99999` and mock the process-existence check.

### C.1.5 `src/tests/system/test_foundation_contracts.py` — wrong filename pattern

**Finding #111** (L462-465)

**Problem:** Test writes `"lock-corrupted.json"` but `read_lock()` expects `f"{slot}-lock.json"`.

**Fix:** Write to `tmp_path / f"{slot}-lock.json"` using the same slot value passed to `read_lock`.

### C.1.6 `src/tests/runtime/test_launch_degraded_vs_blocked.py` — MockLaunchResult bypasses real logic

**Finding #112** (L17-40)

**Problem:** Local `MockLaunchResult` bypasses the real degraded-vs-blocked decision logic.

**Fix:** Invoke the actual production entry (e.g., `ProcessManager.launch`) and assert on returned `LaunchResult.status`, `.warnings`, `.errors`, `.launch_count`.

### C.1.7 `src/tests/cli/smoke_cli_cases.py` — patch return_value

**Finding #106** (L1165-1174)

**Problem:** Configuring attributes on `mock_cfg` but `run_smoke()` reads `mock_cfg.return_value`.

**Fix:** Set attributes on `mock_cfg.return_value` instead.

### C.1.8 `src/tests/server/test_dry_run_flag_bundles.py` — missing stdout assertions

**Finding #104** (L209-240)

**Problem:** Tests only assert `_run_registry_mode` was called, not actual stdout content.

**Fix:** After `dry_run(...)`, read `capsys.readouterr().out` and assert it includes expected bundle sections like `"OpenAI Bundle"`.

---

## C.2 Minor (7 findings)

### C.2.1 `src/tests/system/test_sc006_performance.py` — 3 findings

| # | Lines | Fix |
|---|-------|-----|
| 55 | L57-83 | Add `@pytest.mark.slow` to `test_performance_validation_paths` |
| 66 | L86-114 | Add `@pytest.mark.slow` and warmup call to `test_performance_dry_run_two_slots` |
| 73 | L75-80 | Remove `assert result is not None` from timing loop; assert once after loop |

### C.2.2 `src/tests/smoke/test_smoke_dry_run_flags.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 46 | L281-291 | Rename `test_dry_run_prompt_is_redacted_for_secrets` → `test_dry_run_shows_user_prompt_text`; update docstring |

### C.2.3 `src/tests/build/test_build_config.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 42 | L416-418 | Widen timing tolerance: `assert 29 <= lock.elapsed_seconds <= 31` or `pytest.approx` |
| 50 | L466-475 | Replace `["sycl", "cuda", "cpu"]` with `["sycl", "cuda", "both"]` in `test_build_lock_backend_values` |

### C.2.4 `src/tests/system/test_gguf_fixtures_gen.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 74 | L102-130 | Update docstring: fixtures are under `tmp_path / "fixtures"`, not `src/tests/fixtures/` |
| 98 | L66-74 | Add missing 8-byte `tensor_count` field to `_write_gguf_v3` header |

### C.2.5 `src/tests/system/test_reports.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 102 | L201-209 | Assert exact permissions: `assert dir_mode == 0o700` and `assert file_mode == 0o600` |

### C.2.6 `src/tests/cli/test_profile_cli.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 97 | L517-548 | Patch `run_benchmark` to simulate non-zero exit (`subprocess.CalledProcessError` or `CompletedProcess(returncode=1)`) |

### C.2.7 `src/tests/system/test_foundation_contracts.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 110 | L382-388 | Actually provide invalid `LLM_RUNNER_RUNTIME_DIR` (a file, not a directory) to exercise fallback |

---

## C.3 Trivial (25 findings)

### C.3.1 `src/tests/system/test_benchmark.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 8 | L23-29 | Extract `_make_temp_bin` to module-level pytest fixture `make_temp_bin(tmp_path)` |

### C.3.2 `src/tests/system/test_profile_foundation.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 9 | L411-414 | Remove redundant `os.environ.pop("XDG_CACHE_HOME", None)` after `patch.dict(os.environ, {}, clear=True)` |

### C.3.3 `src/tests/system/test_setup_venv.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 10 | L170 | Remove redundant local `import sys` (module-level import already exists) |

### C.3.4 `src/tests/cli/test_build_cli.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 14 | L492-499 | Remove unused `tmp_path` from `test_main_generic_exception` |
| 17 | L483-490 | Remove unused `tmp_path` from `test_main_keyboard_interrupt` |

### C.3.5 `src/tests/server/test_dry_run_artifacts.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 19 | L340-341 | Move `import re` to module-level; remove in-function imports |
| 32 | L44-53 | Replace weak `str(data).lower()` assertion with direct dict access: `data["resolved_command"]["model_path"]` |

### C.3.6 `src/tests/server/test_server.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 26 | L471-498 | Replace direct `os.environ` manipulation with `monkeypatch` fixture |
| 28 | L300-304 | Remove unused `monkeypatch.setitem(os.environ, "LSPCI_OUTPUT", ...)` |

### C.3.7 `src/tests/cli/test_setup_cli.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 33 | L202-211 | Remove unused `_ = tmp_path / "test-venv"` assignment |

### C.3.8 `src/tests/smoke/test_smoke_json_schema.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 35 | L403-423 | Remove redundant `{"provenance", "exit_code"}` from `allowed_fields`; use `_REQUIRED_RESULT_FIELDS \| _OPTIONAL_RESULT_FIELDS` |
| 44 | L45-55 | Change `_to_report_dict` return type to `dict[str, Any]` |

### C.3.9 `src/tests/smoke/test_smoke_tui_cli_parity.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 64 | L203-255 | Add human/TUI output assertions (capture capsys for human printer, assert phase_reached and provenance fields) |
| 85 | L178-197 | Replace manual JSON construction with `_print_report_json(report)` call |

### C.3.10 `src/tests/support/runtime.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 86 | L45-56 | Make `valid_artifact_data` use per-slot dicts: `resolved_command: {"slot1": {...}}`, `validation_results: {"slot1": {...}}` |

### C.3.11 `src/tests/cli/doctor_cli_cases.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 69 | L239 | Add type hints: `tmp_path: pathlib.Path`, `capsys: pytest.CaptureFixture` (→ `-> None`) to all test methods |

### C.3.12 `src/tests/system/test_validation_regression_contracts.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 58 | L218-236 | Replace `error.failed_check.split("_")[1]` with `_extract_slot_id()` helper that safely returns `""` on unexpected format |

### C.3.13 `src/tests/system/test_toolchain.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 59 | L408-428 | Replace hardcoded expected counts with `len(SYCL_REQUIRED_TOOLS)` / `len(CUDA_REQUIRED_TOOLS)` |

### C.3.14 `src/tests/runtime/test_launch_flow.py` — 1 finding

| # | Lines | Fix |
|---|-------|-----|
| 57 | L29 | Add `tmp_path: pathlib.Path` type annotation to all test methods using `tmp_path` |

### C.3.15 `src/tests/build/build_pipeline_cases.py` — 2 findings

| # | Lines | Fix |
|---|-------|-----|
| 105 | L28-33 | Remove duplicated `if TYPE_CHECKING:` import block |
| 109 | L326-331 | Replace `subprocess.Popen` for stale PID with `stale_pid = 99999` + mock the process-existence check |

### C.3.16 `src/tests/smoke/smoke_cases.py` — 1 finding (trivial)

| # | Lines | Fix |
|---|-------|-----|
| 93 | L780-791 | Update `test_exit_code_unknown_status_fallback` to actually exercise the unknown-status branch |

---

## C.4 Cross-file Dependencies

| Dependent Fix | Depends On | Reason |
|---------------|-----------|--------|
| `test_build_lock_backend_values` (C.2.2) | N/A | Standalone: fix string values |
| `test_stale_lock` tests (C.1.4) | `psutil.pid_exists` mock available | Uses `psutil` which is already a dep |
| `test_launch_degraded_vs_blocked` (C.1.6) | `ProcessManager.launch` signature | Must match production code |
| `test_empty_results_identical` (C.3.14) | `_print_report_json` exists | Uses existing production formatter |

---

# SECTION D — Documentation

> **Assignee:** Documentation agent (can be merged with TUI Developer)
> **Total findings:** 1

### D.1 `QUICKSTART.md` — Ambiguous module reference

**Finding #11** (L48-51)

**Problem:** References `config_builder.py` ambiguously.

**Fix:** Change to `llama_manager/config/builder.py` or `llama_manager.config.builder`.

---

# Implementation Order

## Phase 1 — Critical Security/Data Fixes (Must Do First)

Fix these before any other changes. They affect system security and data correctness.

1. **process_manager.py** — Risk evaluation ignored (A.1)
2. **doctor.py** — Confirmation bypass (B.1.1)
3. **smoke.py** — Config divergence (B.1.2)

## Phase 2 — Major Bugs

4. **doctor.py** — RepairAction composite command (B.2.1)
5. **doctor.py** — Default subcommand returns None (B.2.2)
6. **setup.py** — Duplicate backend conversion (B.2.3)
7. **smoke.py** — Duplicate slot mappings (B.2.4)
8. **smoke.py** — Manual `__post_init__` (B.2.5)
9. **cli_parser.py** — Short flags as TUI mode (B.2.6)
10. **finalize.py** — Duplicate artifact serialization (A.2.1)
11. **profile_cache.py** — Symlink vulnerability (A.2.2)
12. **builder.py** — Port validation alignment (A.2.3)

## Phase 3 — Minor Fixes

13. **toolchain.py** — Empty string from `_extract_version` (A.3.1)
14. **security.py** — AUTH_HEADER redaction (A.3.2)
15. **risk_ack.py** — Mutation of input configs (A.3.3)
16. **file_ops.py** — Type mixing (A.3.4)
17. **_binary.py** — Unsigned int readers (A.3.5)
18. **profile.py** — `_resolve_bench_bin` path (B.3.1)
19. **profile_status.py** — Trailing newline + else (B.3.2)
20. **risk.py** — Hardcoded message + typing (B.3.3)
21. **system_health.py** — cpu_percent + dead code + caching (B.3.4, B.3.5)
22. **notices.py** — Render mutation (B.3.6)
23. **lock.py** — PID validation + bare except (A.4.7)
24. **builder.py** — Registry caching, Mapping check, deep_merge doc, override_dict, except handling (A.4.1)

## Phase 4 — Trivial Fixes

25. All remaining trivial findings across all modules (A.4.2 through B.4.9)

## Phase 5 — Test Fixes

26. All critical/major test fixes (C.1)
27. All minor test fixes (C.2)
28. All trivial test fixes (C.3)

## Phase 6 — Validation

29. Run `ruff check .` — must pass
30. Run `ruff format --check` — must pass
31. Run `pyright` — must pass
32. Run `pytest` — must pass
33. Run `pip-audit` — check for CVEs

---

# CI Quality Gates

Every agent must ensure their changes pass all four CI gates before submitting:

| Gate | Command | Description |
|------|---------|-------------|
| **Lint** | `uv run ruff check .` | No lint errors |
| **Format** | `uv run ruff format --check` | Code properly formatted |
| **Type Check** | `uv run pyright` | No type errors |
| **Tests** | `uv run pytest` | All tests pass |

---

# Important Reminders

1. **Verify before fixing:** Every CodeRabbit finding includes "Verify each finding against the current code and only fix it if needed." Confirm the finding is still valid in the current code state before applying any fix.

2. **One-way dependency:** `llama_manager` must never import from `llama_cli`. The dependency is strictly one-way.

3. **No subprocess in tests:** Test fixes must not introduce subprocess spawning or GPU dependency.

4. **Backwards incompatible:** No compatibility shims. Fix code directly.

5. **br workflow:** After completing fixes, run `br close <id> --reason "Fixed in CodeRabbit sweep"` for any related issues.

6. **Session end checklist:**
   ```bash
   git status
   git add <changed-files>
   br sync --flush-only
   git add .beads/
   git commit -m "fix: apply CodeRabbit review findings (sweep)"
   git push
   ```
