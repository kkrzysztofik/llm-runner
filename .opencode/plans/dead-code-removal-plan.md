# Plan: Remove Deprecated / Obsolete / Unused Code

**Goal:** Clean up dead code, backward-compat aliases, and legacy modules across the llm-runner codebase.

**Verification gate (MANDATORY after every batch):**
```bash
uv run pre-commit run --all-files
uv run pytest
```

---

## Batch A — Dead Files (Zero Production Impact)

### A1. Delete `src/llama_cli/tui/components/slot_status.py`
- **Why:** Entire file is `# pragma: no cover`. `BACKEND_LABELS` is duplicated in `viewmodel.py`; `STATUS_COLORS` is never imported.
- **Action:** `rm src/llama_cli/tui/components/slot_status.py`
- **Risk:** None. No imports found anywhere.

### A2. Delete `src/llama_cli/gpu_collectors.py`
- **Why:** Marked `DEPRECATED` in docstring. `collect_nvtop_stats` is a pure re-export from `llama_manager.gpu_telemetry`. `_get_cpu_percent` / `_get_memory_percent` have zero production callers.
- **Action:** `rm src/llama_cli/gpu_collectors.py`
- **Risk:** None.

### A3. Delete `src/tests/system/test_gpu_collectors.py`
- **Why:** Only tests the dead module from A2.
- **Action:** `rm src/tests/system/test_gpu_collectors.py`
- **Risk:** None.

### A4. Delete `src/tests/helpers.py`
- **Why:** Compatibility re-export from `tests.support.helpers`. Zero imports across the entire codebase.
- **Action:** `rm src/tests/helpers.py`
- **Risk:** None.

### A5. Remove `BuildPanel` from `src/llama_cli/tui/components/build.py`
- **Why:** Commented as *"Legacy BuildPanel — kept for reference but no longer mounted"*. No imports, no CSS references, no instantiations.
- **Action:** Delete lines 58–85 (the header comment + `BuildPanel` class). Leave everything else intact.
- **Exact snippet to remove:**
  ```python
  # ---------------------------------------------------------------------------
  # Legacy BuildPanel — kept for reference but no longer mounted.
  # ---------------------------------------------------------------------------


  class BuildPanel(Container):
      """Build panel widget — always mounted, visibility toggled via CSS."""

      def __init__(self) -> None:  # pragma: no cover
          super().__init__(id="build-panel", classes="build-panel")
          self._title = Static("", id="build-title")
          self._message = Static("", id="build-message")
          self._progress = ProgressBar(id="build-progress", total=None, show_eta=False)
          self._result = Static("", id="build-result")
          self._retry_info = Static("", id="build-retry-info")
          self._error = Static("", id="build-error")
          self._target_prompt = Static("", id="build-target-prompt")

      def compose(self) -> ComposeResult:  # pragma: no cover
          with Container(id="build-content"):
              yield self._title
              yield self._message
              yield self._progress
              yield self._result
              yield self._retry_info
              yield self._error
              yield self._target_prompt
  ```
- **Risk:** None.

---

## Batch B — Backward-Compat Aliases (Per AGENTS.md: No Compat Shims)

### B1. Remove `ValidationResult.valid` alias
**File:** `src/llama_manager/config/errors.py`
- **Action:** Delete lines 18–21:
  ```python
      @property
      def valid(self) -> bool:
          """Alias for passed to maintain backward compatibility"""
          return self.passed
  ```

**File:** `src/tests/config/test_config_builders.py`
- **Action:** Update three test assertions to use `.passed` instead of `.valid`:
  1. Line 345: `assert result.valid is True` → `assert result.passed is True`
  2. Line 360: `assert result.valid is False` → `assert result.passed is False`
  3. Lines 365–370: Replace the entire `test_validation_result_valid_alias` test with nothing (it tests the removed alias).
- **Exact snippet to delete (lines 365–370):**
  ```python
      def test_validation_result_valid_alias(self) -> None:
          """ValidationResult.valid should alias passed property."""
          passed_result = ValidationResult(slot_id="a", passed=True)
          failed_result = ValidationResult(slot_id="b", passed=False)
          assert passed_result.valid is passed_result.passed
          assert failed_result.valid is failed_result.passed
  ```
- **Risk:** Low. Only test references; no production code uses `.valid`.

### B2. Remove `device_mapping` compat parameter from `gpu_index_for_config`
**File:** `src/llama_manager/slot_manager.py`
- **Action:** Remove the `device_mapping` parameter and its fallback logic.
- **Current signature (line 59–76):**
  ```python
  def gpu_index_for_config(
      cfg: ServerConfig,
      device_mapping: dict[str, int] | None = None,
  ) -> int:
      """Return telemetry ordinal for a configuration.

      Args:
          cfg: Server configuration.
          device_mapping: Optional override mapping from device class to GPU
              ordinal. Retained for compatibility.

      Returns:
          GPU ordinal for the device's dashboard panel.
      """
      device_class = device_class_for_config(cfg)
      if device_mapping is not None:
          return device_mapping.get(device_class, 0)
      return parse_gpu_telemetry_selector(cfg.device, cfg.main_gpu).ordinal
  ```
- **New signature:**
  ```python
  def gpu_index_for_config(cfg: ServerConfig) -> int:
      """Return telemetry ordinal for a configuration.

      Args:
          cfg: Server configuration.

      Returns:
          GPU ordinal for the device's dashboard panel.
      """
      return parse_gpu_telemetry_selector(cfg.device, cfg.main_gpu).ordinal
  ```

**File:** `src/tests/slot/test_slot_manager.py`
- **Action:** Delete the `test_custom_mapping` test (lines 142–145):
  ```python
      def test_custom_mapping(self) -> None:
          cfg = _make_config(device="SYCL0")
          mapping = {"sycl": 2, "cuda": 3}
          assert gpu_index_for_config(cfg, device_mapping=mapping) == 2
  ```
- **Risk:** Low. One test used the parameter; no production callers found.

---

## Batch C — Legacy `colors.py` -> `ui_output.py` Migration

**File:** `src/llama_cli/colors.py`
- **Why:** `ARCHITECTURE.md` marks it *"legacy; prefer ui_output"*. It exposes a large `Colors` class that duplicates what `_style()` in `ui_output.py` already does.
- **Strategy:** Migrate `doctor.py` and `setup.py` to use `_style()` from `ui_output.py`, then delete `colors.py`.

### C1. Understand the color mapping
`ui_output.py`'s `_COLORS` uses:
- `"green"` -> `\033[92m` (bright green)
- `"red"` -> `\033[91m` (bright red)
- `"yellow"` -> `\033[93m` (bright yellow)
- `"cyan"` -> `\033[96m` (bright cyan)

These are already the bright ANSI variants that `Colors.bright_green`, `Colors.bright_red`, etc. used. So:
- `Colors.bright_green(x)` -> `_style(x, "green")`
- `Colors.bright_red(x)` -> `_style(x, "red")`
- `Colors.bright_yellow(x)` -> `_style(x, "yellow")`
- `Colors.green(x)` -> `_style(x, "green")` (same bright variant)
- `Colors.cyan(x)` -> `_style(x, "cyan")`

### C2. Update `src/llama_cli/commands/doctor.py`
- **Action:**
  1. Replace import at line 17: `from llama_cli.colors import Colors` -> `from llama_cli.ui_output import _style`
  2. Replace all `Colors.*` calls:
     - Line 467: `Colors.bright_green("✓ YES")` -> `_style("✓ YES", "green")`
     - Line 468: `Colors.bright_red("✗ NO")` -> `_style("✗ NO", "red")`
     - Line 469: `Colors.bright_yellow("⚠ NO")` -> `_style("⚠ NO", "yellow")`
     - Line 479: `Colors.bright_red(str(result.profiles_stale))` -> `_style(str(result.profiles_stale), "red")`
     - Line 481: `Colors.bright_green(str(result.profiles_stale))` -> `_style(str(result.profiles_stale), "green")`
     - Line 814: `Colors.bright_yellow(" [CONFIRMATION REQUIRED]")` -> `_style(" [CONFIRMATION REQUIRED]", "yellow")`
     - Line 816: `Colors.cyan(str(i))` -> `_style(str(i), "cyan")`

### C3. Update `src/llama_cli/commands/setup.py`
- **Action:**
  1. Replace import at line 16: `from llama_cli.colors import Colors` -> `from llama_cli.ui_output import _style`
  2. Replace all `Colors.*` calls:
     - Line 67: `Colors.bright_green("✓ YES")` -> `_style("✓ YES", "green")`
     - Line 68: `Colors.bright_red("✗ NO")` -> `_style("✗ NO", "red")`
     - Line 69: `Colors.bright_red("MISSING")` -> `_style("MISSING", "red")`
     - Line 82: `Colors.green(value)` -> `_style(value, "green")`
     - Line 83: `Colors.cyan(name)` -> `_style(name, "cyan")`

### C4. Delete `src/llama_cli/colors.py`
- **Action:** `rm src/llama_cli/colors.py`
- **Risk:** Low. Only `doctor.py` and `setup.py` used it; they are being migrated above.

### C5. Update `docs/ARCHITECTURE.md`
- **Action:** Remove the "(legacy; prefer ui_output)" annotation from line 70, or replace the line to remove the file entry entirely since it will no longer exist.
- **Current line 70:** `│   ├── colors.py                   # ANSI colour constants (legacy; prefer ui_output)`
- **Change:** Delete that line.

---

## Batch D — `profile.py` `require_executable` Refactor

**Context:** `src/llama_cli/commands/profile.py` lines 83–93 defines a local `require_executable` that raises `FileNotFoundError` / `PermissionError`. `llama_manager.validation.validators` already has `require_executable(bin_path, name)` that returns `ErrorDetail | None`. We should unify on the manager version.

### D1. Remove local `require_executable` from `profile.py`
- **Action:** Delete lines 83–93:
  ```python
  def require_executable(path: str) -> None:
      """Validate that *path* exists and is executable.

      Raises:
          FileNotFoundError: If the path does not exist.
          PermissionError: If the path exists but is not executable.
      """
      if not os.path.exists(path):
          raise FileNotFoundError(f"file not found: {path}")
      if not os.access(path, os.X_OK):
          raise PermissionError(f"not executable: {path}")
  ```

### D2. Import the manager validator
- **Action:** In `profile.py`, find the `llama_manager` imports and add:
  ```python
  from llama_manager.validation.validators import require_executable
  ```
  (Check existing imports in `profile.py` first — there may already be a star or partial import from `llama_manager`.)

### D3. Update the call site (lines 167–171)
- **Current code:**
  ```python
      try:
          require_executable(bench_bin)
      except (FileNotFoundError, PermissionError) as exc:
          _emit(f"error: benchmark binary unavailable: {exc}", stderr=True)
          return 1
  ```
- **New code:**
  ```python
      exec_err = require_executable(bench_bin, name="benchmark binary")
      if exec_err is not None:
          _emit(f"error: {exec_err.why_blocked}", stderr=True)
          return 1
  ```
- **Risk:** Low. Behavior is identical (exits with code 1 and prints an error), but uses the canonical library validator.

---

## Execution Order for Smaller Model

1. **Run Batch A** (dead files + BuildPanel). Run gate.
2. **Run Batch B** (aliases). Run gate.
3. **Run Batch C** (colors migration). Run gate.
4. **Run Batch D** (profile.py refactor). Run gate.

Each batch is self-contained and safe to apply independently. The smaller model should apply one batch at a time, run the gate, fix any issues, then proceed.

## Edge Cases / Notes

- `_style` in `ui_output.py` uses `_tty()` to decide whether to emit ANSI codes. `Colors` used a module-level `enabled` flag. For CLI commands running in a terminal, behavior is identical. For non-TTY output, `_style` correctly strips colors while `Colors` would have too if `enabled=False` — but no code ever sets `enabled=False`.
- `profile.py` may have `from llama_manager import ...` or `from llama_manager.profile_orchestrator import ...` already. The smaller model should inspect the existing imports and add `require_executable` there without creating duplicate import blocks.
- `docs/ARCHITECTURE.md` is documentation; updating it is optional but recommended to keep it in sync.
