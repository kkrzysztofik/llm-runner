# Wave 2 Execution Checklist: Major Runtime/Code Behavior

**Scope:** 40+ major findings  
**Estimated Duration:** 6-8 hours  
**Dependencies:** Wave 1 (critical) must be complete first

---

## Pre-Flight: Wave 1 Verification

Before starting Wave 2, confirm Wave 1 commits are merged:

- [ ] `LockMetadata.started_at` uses `time.time()` (not `time.monotonic()`)
- [ ] `process_manager.py` `_atomic_write_json` uses atomic temp+rename
- [ ] `process_manager.py` lock creation uses O_CREAT|O_EXCL
- [ ] `gpu_stats.py` subprocess checks returncode before json.loads
- [ ] `dry_run.py` `build_dry_run_slot_payload` handles None validation_results
- [ ] `test_fr003_007_011_contracts.py` isinstance checks fixed

**Command to verify:**
```bash
git log --oneline | grep -E "(lock|monotonic|atomic|json.loads|validation_results|isinstance)" | head -10
```

---

## Python Backend Specialist Tasks

### Task 2.1: colors.py Migration to llama_cli (Findings 23, 28)
**Priority:** CRITICAL - Blocks all other tasks  
**Time:** 45 minutes

#### Steps:
1. [ ] Create `src/llama_cli/colors.py` with:
   - Copy entire `Colors` class from `src/llama_manager/colors.py`
   - Include `COLORS`, `get_code`, `is_enabled`, `set_enabled` methods
   - Remove `sys.stdout.isatty()` call from `is_enabled()`
   - Add module-level `Colors.enabled` flag (default: True)
   - Add `Colors.set_enabled(bool)` setter

2. [ ] Update `src/llama_manager/__init__.py`:
   - Remove `from .colors import Colors`
   - Remove `Colors` from `__all__`

3. [ ] Update all `llama_cli/` imports:
   - `from llama_manager.colors import Colors` → `from llama_cli.colors import Colors`
   - Verify all usages: `server_runner.py`, `tui_app.py`

4. [ ] Delete `src/llama_manager/colors.py`

5. [ ] Update `Colors.is_enabled()` usage in `server_runner.py:280`:
    ```python
    _ = Colors.is_enabled()  # Initialize color detection
    ```

6. [ ] Add `Colors.set_enabled(True)` in CLI entrypoints if needed

**Acceptance:**
- [ ] `llama_manager` has no TUI/presentation code
- [ ] `Colors.get_code()` works without Rich dependency in tests
- [ ] All imports resolve correctly

**Verification:**
```bash
uv run python -c "from llama_cli.colors import Colors; print(Colors.get_code('test'))"
uv run ruff check src/llama_manager/
uv run pyright src/llama_manager/
```

---

### Task 2.2: config.py Core Fixes (Findings 81, 116, 158, 159)
**Priority:** HIGH  
**Time:** 60 minutes

#### Steps:
1. [ ] **Move ModelSlot above functions (Finding 81):**
   - Cut `ModelSlot` dataclass definition
   - Paste before `detect_duplicate_slots()` function
   - Update forward references from `"ModelSlot"` to `ModelSlot`

2. [ ] **Validate ErrorCode naming consistency (Finding 116):**
   - Keep external serialized values stable (do not change wire/report values without migration plan)
   - Ensure enum member names remain clear and deterministic

3. [ ] **Fix derived paths (Finding 158):**
   - Add `__post_init__` to `Config` dataclass:
     ```python
     def __post_init__(self):
         self.llama_server_bin_intel = Path(self.llama_cpp_root) / "llama-server-sycl"
         self.llama_server_bin_nvidia = Path(self.llama_cpp_root) / "llama-server-cuda"
     ```
   - Remove class-level defaults for these fields

4. [ ] **Fix mutable aliasing (Finding 159):**
   - Add `import copy` at top of `config_builder.py`
   - Update `merge_config_overrides()`:
     ```python
     if isinstance(value, (dict, list)):
         result[key] = copy.deepcopy(value)
     else:
         result[key] = value
     ```

**Acceptance:**
- [ ] `ModelSlot` defined before `detect_duplicate_slots()`
- [ ] ErrorCode member names are consistent and serialized values remain stable
- [ ] `Config(llama_cpp_root="/custom")` computes derived paths correctly
- [ ] `merge_config_overrides()` doesn't mutate original dict

**Verification:**
```bash
uv run pytest src/tests/test_config.py -xvs -k "merge or detect or config"
uv run pyright src/llama_manager/config.py
```

---

### Task 2.3: server.py Improvements (Findings 167, 171, 172, 173)
**Priority:** MEDIUM  
**Time:** 30 minutes

#### Steps:
1. [ ] **Fix type ignore (Finding 167):**
   - Check `hardware_notes` type in `_build_hardware_notes()` and `DryRunSlotPayload`
   - If types align, remove `# type: ignore[assignment]` and add comment:
     ```python
     # mypy false positive: both are dict[str, str | None]
     ```

2. [ ] **Fix model_path validation (Finding 171):**
   - Update validation logic to handle 3 cases:
     ```python
     if not slot.model_path:
         # Skip validation (empty path allowed)
         pass
     elif os.path.isfile(slot.model_path):
         # Valid file
         pass
     elif os.path.isdir(slot.model_path):
         # Directory not allowed (or allowed if HF-style)
         validation_results.append(
             ValidationResult(
                 error_code=ErrorCode.FILE_NOT_FOUND,
                 message="model_path must be a file, not a directory"
             )
         )
     else:
         # Path doesn't exist
         validation_results.append(
             ValidationResult(
                 error_code=ErrorCode.FILE_NOT_FOUND,
                 message=f"model_path does not exist: {slot.model_path}"
             )
         )
     ```

3. [ ] **Precompile regex (Finding 172):**
   - Add module-level constant:
     ```python
     _SENSITIVE_KEY_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH)", re.IGNORECASE)
     ```
   - Update `redact_sensitive()` to use `_SENSITIVE_KEY_PATTERN.search(env_key)`

4. [ ] **Avoid Config() instantiation (Finding 173):**
   - Option A: Make `ServerConfig.server_bin` non-optional and require explicit value
   - Option B: Add `default_bin` parameter to `build_server_cmd()`:
     ```python
     def build_server_cmd(cfg: ServerConfig, default_bin: str) -> list[str]:
         server_bin = cfg.server_bin or default_bin
         ...
     ```
   - Update all call sites to pass `default_bin=Config().llama_server_bin_intel` once

**Acceptance:**
- [ ] Type ignore removed or documented
- [ ] model_path validation handles file/dir/nonexistent
- [ ] Regex precompiled at module level
- [ ] `Config()` called only once at setup

**Verification:**
```bash
uv run pytest src/tests/test_server.py -xvs
uv run pyright src/llama_manager/server.py
```

---

### Task 2.4: gpu_stats.py Fixes (Findings 134, 162)
**Priority:** HIGH  
**Time:** 25 minutes

#### Steps:
1. [ ] **Normalize schema (Finding 134):**
   - Update `_get_nvtop_stats()` psutil fallback to return same keys as nvtop branch:
     ```python
     return {
         "device": f"GPU {i}",
         "gpu_util": "N/A",
         "mem_util": "N/A",
         "temp": "N/A",
         "power": "N/A",
         "cpu": psutil.cpu_percent(),
         "mem": psutil.virtual_memory().percent,
     }
     ```

2. [ ] **Fix bare except (Finding 162):**
   - Replace:
     ```python
     except Exception:
         pass
     ```
   - With:
     ```python
     except (ValueError, OSError, subprocess.CalledProcessError) as e:
         logger.exception("Failed to parse GPU stats: %s", e)
         raise
     ```

**Acceptance:**
- [ ] Both nvtop and psutil paths return same keys
- [ ] Exceptions logged with stack trace
- [ ] Tests pass

**Verification:**
```bash
uv run pytest src/llama_manager/gpu_stats.py -xvs
uv run pyright src/llama_manager/gpu_stats.py
```

---

## TUI Developer Tasks

### Task 2.5: server_runner.py Fixes (Findings 27, 91, 118, 119, 132, 153)
**Priority:** HIGH  
**Time:** 45 minutes

#### Steps:
1. [ ] **Add return type (Finding 27):**
   - Update `main()` signature:
     ```python
     def main() -> None:
         args = parse_tui_args()
         ...
     ```

2. [ ] **Extract port defaulting (Finding 91):**
   - Add helper function:
     ```python
     def _resolve_port(ports: list[int], index: int, default: int) -> int:
         return ports[index] if len(ports) > index else default
     ```
   - Replace:
     ```python
     ports[0] if ports else cfg.summary_balanced_port
     ```
     with:
     ```python
     _resolve_port(ports, 0, cfg.summary_balanced_port)
     ```

3. [ ] **Fix is_enabled() usage (Finding 118):**
   - Update line 280:
     ```python
     _ = Color.is_enabled()  # Initialize color detection
     ```

4. [ ] **Extract error printing (Finding 119):**
   - Add helper:
     ```python
     def _print_backend_error_and_exit(backend_error: ErrorDetail) -> NoReturn:
         print(f"Error code: {backend_error.error_code}", file=sys.stderr)
         print(f"Failed check: {backend_error.failed_check}", file=sys.stderr)
         print(f"Why blocked: {backend_error.why_blocked}", file=sys.stderr)
         print(f"How to fix: {backend_error.how_to_fix}", file=sys.stderr)
         raise SystemExit(1)
     ```
   - Replace duplicated blocks with `_print_backend_error_and_exit(backend_error)`

5. [ ] **Add EOFError handling (Finding 132):**
   - Wrap `input(RISK_CONFIRM_PROMPT)`:
     ```python
     try:
         response = input(RISK_CONFIRM_PROMPT).strip().lower()
     except EOFError:
         _print_backend_error_and_exit(backend_error)  # Treat as non-confirmation
     ```

6. [ ] **Fix broad except (Finding 153):**
   - Update:
     ```python
     try:
         _run_mode(parsed.mode, parsed.ports, manager, cfg)
     except ValueError as e:
         logger.error(f"Invalid arguments: {e}")
         raise SystemExit(1)
     except IndexError as e:
         # Log and re-raise or handle explicitly
         logger.exception(f"Index error in _run_mode: {e}")
         raise
     ```

**Acceptance:**
- [ ] All return types annotated
- [ ] Port defaulting extracted to helper
- [ ] Error printing deduplicated
- [ ] EOFError handled gracefully
- [ ] No broad except blocks

**Verification:**
```bash
uv run pytest src/tests/test_tui*.py -xvs
uv run pyright src/llama_cli/server_runner.py
```

---

### Task 2.6: tui_app.py Fixes (Findings 147, 157, 160, 170)
**Priority:** HIGH  
**Time:** 40 minutes

#### Steps:
1. [ ] **Add type hint (Finding 147):**
   - Import: `from rich.console import ConsoleDimensions`
   - Update `on_resize`:
     ```python
     def on_resize(self, event: ConsoleDimensions) -> None:
         ...
     ```

2. [ ] **Fix Live loop signal handling (Finding 157):**
   - Add `stop()` method to `TUIApp`:
     ```python
     def stop(self) -> None:
         self.running = False
     ```
   - Update signal handlers:
     ```python
     def _handle_signal(signum, frame):
         self.stop()
     ```
   - Wire into `ServerManager.on_interrupt/on_terminate`:
     ```python
     manager.on_interrupt = lambda: self.stop()
     manager.on_terminate = lambda: self.stop()
     ```

3. [ ] **Cache Config instance (Finding 160):**
   - In `__init__`:
     ```python
     self.config = Config()
     ```
   - Update `_build_column_panel`:
     ```python
     # Before: Config().host
     # After: self.config.host
     ```

4. [ ] **Fix empty right column (Finding 170):**
   - In `render()`, add placeholder when `len(self.configs) == 1`:
     ```python
     if len(self.configs) == 1:
         layout["right"].update(self._build_placeholder_panel())
     ```
   - Add `_build_placeholder_panel()`:
     ```python
     def _build_placeholder_panel(self) -> Panel:
         return Panel(
             "[dim]No secondary config[/dim]",
             title="Status",
             border_style="dim",
         )
     ```

**Acceptance:**
- [ ] `on_resize` has type hint
- [ ] TUI exits cleanly on signal
- [ ] Config cached, no repeated allocations
- [ ] Right column shows placeholder

**Verification:**
```bash
uv run pytest src/tests/test_tui*.py -xvs
uv run pyright src/llama_cli/tui_app.py
```

---

### Task 2.7: dry_run.py Fixes (Findings 144, 146, 164)
**Priority:** MEDIUM  
**Time:** 20 minutes

#### Steps:
1. [ ] **Add NoReturn type (Finding 144):**
   - Import: `from typing import NoReturn`
   - Update:
     ```python
     def _print_acknowledgement_required_and_exit() -> NoReturn:
         ...
     ```

2. [ ] **Fix error message (Finding 146):**
   - Update:
     ```python
     allowed_modes = ", ".join(sorted(handlers.keys()))
     print(f"Invalid mode: {mode}. Valid modes: {allowed_modes}", file=sys.stderr)
     sys.exit(1)
     ```
   - Or explicitly add "llama32":
     ```python
     print(f"Invalid mode: {mode}. Valid modes: dry-run, both, summary-balanced, ...", file=sys.stderr)
     ```

3. [ ] **Move late import (Finding 164):**
   - Add to top imports:
     ```python
     from llama_manager import DryRunSlotPayload
     ```
   - Remove inline import in `dry_run()` function

**Acceptance:**
- [ ] `NoReturn` annotated
- [ ] Error message includes all modes
- [ ] `DryRunSlotPayload` imported at module level

**Verification:**
```bash
uv run pytest src/tests/test_us2_dry_run_schema.py -xvs
uv run pyright src/llama_cli/dry_run.py
```

---

## Python QA Tasks

### Task 2.8: Test File Improvements (Multiple Findings)
**Priority:** HIGH  
**Time:** 90 minutes

#### Group 1: Type Hints & Imports (Findings 9, 41, 42, 46, 82, 85, 92, 106, 108, 112, 126, 135, 136, 141, 142, 148, 151, 152, 166)
**Files:** `test_us3_precedence.py`, `conftest.py`, `test_regression_m1_contracts.py`, `test_sc006_performance.py`, `test_server.py`, `test_us1_launch_flow.py`, `__init__.py`, `test_us2_dry_run_schema.py`, `test_us1_lock_integrity.py`, `test_process_manager.py`, `test_foundation_contracts.py`

**Steps:**
1. [ ] Add `-> None` return types to all test functions
2. [ ] Move inline imports to module level (re, time, stat, ServerManager, LaunchResult)
3. [ ] Add type hints to fixtures (`artifact_writer`, `base_config`, `create_mock_proc`)
4. [ ] Fix line length > 100 chars (split parameters)
5. [ ] Update `__init__.py` imports to be alphabetically ordered

**Verification:**
```bash
uv run ruff check src/tests/
uv run pyright src/tests/
```

---

#### Group 2: Duplicated Helpers (Findings 44, 45, 61, 64, 69, 70, 108, 127)
**Files:** `test_regression_m1_contracts.py`, `test_t039_t040_risky_acknowledgement.py`, `test_us3_risk_acknowledgement.py`, `test_us2_artifacts.py`, `test_fr003_007_011_contracts.py`, `test_us1_degraded_vs_full_block.py`

**Steps:**
1. [ ] Extract `_cfg` helper to module-level fixture (test_regression_m1_contracts.py)
2. [ ] Extract `_FakeLive` class to module level (test_t039_t040_risky_acknowledgement.py)
3. [ ] Replace try/except with `pytest.importorskip()` (test_us3_risk_acknowledgement.py)
4. [ ] Extract `_valid_artifact_data` to fixture (test_us2_artifacts.py)
5. [ ] Extract `make_dry_run_payload` helper (test_fr003_007_011_contracts.py)
6. [ ] Extract `minimal_cfg` fixture (test_fr003_007_011_contracts.py)
7. [ ] Extract `mock_live_process_indeterminate` fixture (test_us1_degraded_vs_full_block.py)

**Verification:**
```bash
uv run pytest src/tests/ -xvs
```

---

#### Group 3: Mocks & Synchronization (Findings 88, 90, 93, 107, 129, 130, 133, 138, 161, 165)
**Files:** `test_process_manager.py`, `test_us1_degraded_vs_full_block.py`, `test_us1_launch_flow.py`

**Steps:**
1. [ ] Replace `time.sleep(0.1)` with `threading.Event.wait(timeout=1.0)` or `thread.join(timeout=5)`
2. [ ] Add `pytest.mark.skipif(sys.platform == "win32")` to chmod tests
3. [ ] Add `psutil.pid_exists` mocks where locks are created with fake PIDs
4. [ ] Fix test names to match behavior (e.g., `test_invalid_port_raises_validation_error`)

**Verification:**
```bash
uv run pytest src/tests/test_process_manager.py src/tests/test_us1_degraded_vs_full_block.py -xvs
```

---

#### Group 4: Test Logic Fixes (Findings 15, 16, 20, 24, 49, 55, 56, 62, 71, 72, 73, 78, 83, 84, 87, 97, 109, 111, 115, 124, 128, 131, 137, 139, 140, 145, 149, 150, 154, 155, 156, 166)
**Files:** Multiple test files

**Steps:**
1. [ ] Validate `expected_type` in `artifact_writer` (conftest.py)
2. [ ] Fix `to_dict` to use correct variable (test_us3_determinism.py)
3. [ ] Fix docstring accuracy (test_us3_risk_acknowledgement.py)
4. [ ] Replace float equality with `pytest.approx()` (test_us2_actionable_errors.py)
5. [ ] Remove unused fixture parameters (test_regression_m1_contracts.py)
6. [ ] Replace string concatenation with `os.path.join()` (test_sc006_performance.py)
7. [ ] Remove weak tests or update to use persisted artifacts (test_us2_artifacts.py)
8. [ ] Add assertions to lock tests (test_us1_lock_integrity.py)
9. [ ] Remove redundant assertions (test_fr003_007_011_contracts.py)
10. [ ] Fix timestamp comparison to check values, not just format (test_us2_dry_run_schema.py)
11. [ ] Remove unused `enumerate()` variable (test_us2_dry_run_schema.py)
12. [ ] Fix inconsistent tests (test_us1_launch_flow.py, test_us1_degraded_vs_full_block.py)
13. [ ] Replace `os.environ` manipulation with `monkeypatch` (test_us2_dry_run_schema.py)

**Verification:**
```bash
uv run pytest src/tests/ -x --tb=short
```

---

## Build Engineer Tasks

### Task 2.9: Shell Script Fixes (Findings 38, 59, 122)
**Priority:** HIGH  
**Time:** 15 minutes

#### Steps:
1. [ ] **Fix comment accuracy (Finding 38):**
   - Update comment above `DEFAULT_CACHE_TYPE_SUMMARY_K`:
     ```bash
     # Intel SYCL summary: q8_0 KV cache for 256k context (memory savings outweigh slight overhead vs f16)
     DEFAULT_CACHE_TYPE_SUMMARY_K=q8_0
     DEFAULT_CACHE_TYPE_SUMMARY_V=q8_0
     ```

2. [ ] **Add sampling flags (Finding 59):**
   - In `start_both_qwen35()`, after building `summary_balanced_cmd`:
     ```bash
     summary_balanced_cmd+=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0)
     summary_balanced_cmd+=(--presence-penalty 0.0 --repeat-penalty 1.0)
     ```

3. [ ] **Fix redaction pattern documentation (Finding 122):**
   - Update quickstart.md:
     ```markdown
     - Sensitive env values are redacted when the key contains any of: KEY, TOKEN, SECRET, PASSWORD, or AUTH (case-insensitive);
       filesystem paths remain visible
     ```

**Verification:**
```bash
bash -n run_opencode_models.sh
uv run llm-runner dry-run both 2>&1 | grep -E "temperature|top-p|top-k"
```

---

## Python Reviewer Tasks

### Task 2.10: Code Review & Refactoring Verification
**Priority:** CRITICAL  
**Time:** 60 minutes

#### Review Checklist:

**Backend Code:**
- [ ] `colors.py` migration complete, no `llama_manager.colors` imports remain
- [ ] `config.py` changes don't break existing tests
- [ ] `server.py` model_path validation handles all cases
- [ ] `gpu_stats.py` schema normalization doesn't break consumers
- [ ] All error handling uses specific exceptions, not bare `except`

**TUI Code:**
- [ ] Signal handlers properly stop TUI loop
- [ ] Config caching doesn't introduce stale state
- [ ] Placeholder panel renders correctly
- [ ] No `console.print()` calls while `Live` is active

**Test Code:**
- [ ] All test functions have `-> None` return type
- [ ] No inline imports in test functions
- [ ] Duplicated helpers extracted to fixtures
- [ ] `psutil` mocks used where appropriate
- [ ] No `time.sleep()` in tests

**Documentation:**
- [ ] All public APIs have docstrings
- [ ] Type hints are accurate and complete
- [ ] Error messages are clear and actionable

**Acceptance:**
- [ ] All CI gates pass (lint, format, typecheck, test)
- [ ] No new warnings introduced
- [ ] Code coverage maintained or improved

---

## CI Fixer Tasks

### Task 2.11: CI Integration Testing
**Priority:** MEDIUM  
**Time:** 30 minutes

#### Steps:
1. [ ] Run full test suite in CI environment:
   ```bash
   uv run pytest src/tests/ --cov --cov-report=xml
   ```

2. [ ] Verify type checking passes:
   ```bash
   uv run pyright
   ```

3. [ ] Verify linting passes:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```

4. [ ] Run pre-commit hooks:
   ```bash
   uv run pre-commit run --all-files
   ```

5. [ ] Update CI workflow if any new tools needed

**Acceptance:**
- [ ] All CI jobs pass
- [ ] No new warnings or failures
- [ ] Coverage report generated successfully

---

## Security Reviewer Tasks

### Task 2.12: Security Validation
**Priority:** HIGH  
**Time:** 30 minutes

#### Review Checklist:

**Input Validation:**
- [ ] `cli_parser.py` validates all ports (1-65535, unique, mode-specific)
- [ ] `server_runner.py` handles EOFError on user input
- [ ] `config_builder.py` deep-copies mutable objects

**Error Handling:**
- [ ] No bare `except Exception` blocks
- [ ] All exceptions logged with context
- [ ] Sensitive data never logged

**Resource Management:**
- [ ] File descriptors properly closed
- [ ] Process cleanup on exit
- [ ] Lock files cleaned up on error

**Acceptance:**
- [ ] Static analysis passes (if available)
- [ ] No security warnings from CI tools
- [ ] All error paths tested

---

## Wave 2 Completion Criteria

### Definition of Done:

- [ ] All 40+ major findings addressed
- [ ] No new lint/typecheck/test failures
- [ ] Code coverage maintained ≥ 80%
- [ ] All specialist tasks verified
- [ ] CI gates pass
- [ ] Security review approved
- [ ] Documentation updated

### Final Verification Commands:

```bash
# Full test suite
uv run pytest src/tests/ --cov --cov-report=term-missing -x

# Type checking
uv run pyright

# Linting
uv run ruff check .
uv run ruff format --check .

# Pre-commit
uv run pre-commit run --all-files

# Dry-run tests
uv run llm-runner dry-run both
uv run llm-runner dry-run summary-balanced
```

---

## Rollback Procedure

If Wave 2 fails:

1. **Identify failing task** - Check CI logs for specific error
2. **Revert affected files** - Use `git revert` or `git checkout`
3. **Isolate issue** - Test individual components
4. **Fix incrementally** - Address one finding at a time
5. **Re-integrate** - Gradually re-apply fixes

**Example rollback:**
```bash
git checkout HEAD~1 -- src/llama_manager/colors.py src/llama_manager/config.py
git commit -m "Revert Wave 2 color/config changes"
```

---

## Notes

- **Wave 1 findings already fixed:** All `time.monotonic()`, `pid=0`, `TOCTOU`, `json.loads`, `isinstance(None)`, and `validation_results=None` issues are excluded
- **Cross-wave dependencies:** Wave 3 (minor) may fix some issues in parallel, but Wave 2 must complete first for runtime stability
- **Specialist coordination:** Python Backend and TUI Developer should work in parallel on their respective files
- **QA integration:** Python QA should run tests after each task completion, not just at end
