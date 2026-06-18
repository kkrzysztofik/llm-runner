# Fix TUI Built-In Profile Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make edited built-in run profiles survive a TUI restart when the TUI is launched with a built-in mode such as `summary-balanced`, `summary-fast`, `qwen35`, or `both`.

**Architecture:** Built-in profile edits are already saved as custom profile overrides in `slot_profiles.toml` via `upsert_custom_slot_profile()`. The bug is that TUI launch mode resolution still uses `create_default_profile_registry()`, which ignores those custom overrides. The fix is to make TUI startup resolve launch-mode profiles through `create_tui_profile_registry()` while leaving non-TUI smoke/profile/dry-run resolution unchanged.

**Tech Stack:** Python 3.12, pytest, Textual TUI controller, XDG config TOML profile store.

---

## Investigation Summary

**Observed failure path**
- User edits a built-in profile in the TUI Profiles screen.
- `DashboardController.update_slot_profile()` saves the edited profile through `upsert_custom_slot_profile(original_profile_id, spec)`.
- The profile store writes the override to `$XDG_CONFIG_HOME/llm-runner/slot_profiles.toml`.
- On TUI restart with a launch mode, `_run_tui()` calls `_build_tui_mode_configs()`.
- `_build_tui_mode_configs()` uses `create_default_profile_registry(cfg)`, so it ignores `slot_profiles.toml`.
- Result: initial slots use default built-in values after restart, even though the override exists on disk.

**Relevant code**
- `src/llama_cli/tui/controller.py:875-887`: edit path calls `upsert_custom_slot_profile()`.
- `src/llama_manager/config/builder.py:750-778`: `create_tui_profile_registry()` merges built-ins + custom overrides and lets custom profiles win.
- `src/llama_cli/server_runner.py:147`: `_build_tui_mode_configs()` incorrectly uses `create_default_profile_registry()`.
- `src/llama_cli/server_runner.py:168`: `_resolve_tui_mode_configs()` fallback incorrectly uses `create_default_profile_registry()`.

**Scope**
- Fix TUI launch and restart behavior only.
- Do not change smoke, profile, dry-run, or backend profiling behavior.
- Do not change profile edit UI layout.
- Do not delete files.
- Do not edit Speckit files.

---

### Task 1: Add Failing Test For TUI Mode Restart Using Built-In Override

**Files:**
- Modify: `src/tests/cli/test_server_runner.py`

**Step 1: Add this test to `TestBuildTuiModeConfigs`**

```python
    def test_build_tui_mode_configs_uses_custom_builtin_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TUI launch configs should use persisted custom overrides for built-ins."""
        from llama_cli.server_runner import _build_tui_mode_configs
        from llama_manager.config import Config, SlotProfileSpec
        from llama_manager.slot_profile_store import save_custom_slot_profile

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        override = SlotProfileSpec(
            profile_id="summary-balanced",
            model="/models/edited-summary.gguf",
            alias="summary-balanced",
            device="SYCL0",
            port=17777,
            ctx_size=12345,
            ubatch_size=256,
            threads=6,
            backend="llama_cpp",
        )
        save_custom_slot_profile(override)

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-balanced", port=None, port2=None)

        result = _build_tui_mode_configs(cfg, parsed)
        configs = result["summary-balanced"][2]

        assert len(configs) == 1
        assert configs[0].model == "/models/edited-summary.gguf"
        assert configs[0].port == 17777
        assert configs[0].ctx_size == 12345
        assert configs[0].threads == 6
```

**Step 2: Run the focused failing test**

Run:
```bash
rtk uv run pytest src/tests/cli/test_server_runner.py::TestBuildTuiModeConfigs::test_build_tui_mode_configs_uses_custom_builtin_override -q 2>&1 | distill "Did the focused regression test fail for the expected reason? Return FAIL_EXPECTED if model/port/ctx_size came from defaults, else return the failure summary."
```

Expected: `FAIL_EXPECTED`. The failure should show default built-in values instead of `/models/edited-summary.gguf` and `17777`.

---

### Task 2: Add Failing Test For `_resolve_tui_mode_configs()` Default Registry Fallback

**Files:**
- Modify: `src/tests/cli/test_server_runner.py`

**Step 1: Add this test to `TestResolveTuiModeConfigs`**

```python
    def test_resolve_mode_configs_uses_tui_registry_when_registry_not_supplied(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback registry should include custom overrides for TUI resolution."""
        from llama_cli.server_runner import _resolve_tui_mode_configs
        from llama_manager.config import Config, SlotProfileSpec
        from llama_manager.slot_profile_store import save_custom_slot_profile

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        save_custom_slot_profile(
            SlotProfileSpec(
                profile_id="summary-fast",
                model="/models/edited-fast.gguf",
                alias="summary-fast",
                device="SYCL0",
                port=18888,
                ctx_size=8192,
                ubatch_size=128,
                threads=3,
                backend="llama_cpp",
            )
        )

        cfg = Config()
        parsed = argparse.Namespace(mode="summary-fast", port=None, port2=None)

        configs = _resolve_tui_mode_configs("summary-fast", cfg, parsed)

        assert len(configs) == 1
        assert configs[0].model == "/models/edited-fast.gguf"
        assert configs[0].port == 18888
        assert configs[0].threads == 3
```

**Step 2: Run the focused failing test**

Run:
```bash
rtk uv run pytest src/tests/cli/test_server_runner.py::TestResolveTuiModeConfigs::test_resolve_mode_configs_uses_tui_registry_when_registry_not_supplied -q 2>&1 | distill "Did the focused fallback-registry test fail for the expected reason? Return FAIL_EXPECTED if values came from defaults, else return the failure summary."
```

Expected: `FAIL_EXPECTED`.

---

### Task 3: Implement Minimal TUI Registry Fix

**Files:**
- Modify: `src/llama_cli/server_runner.py:28`
- Modify: `src/llama_cli/server_runner.py:147`
- Modify: `src/llama_cli/server_runner.py:168`

**Step 1: Import the TUI registry builder**

Add this import near the existing config imports:

```python
from llama_manager.config.builder import create_tui_profile_registry
```

Do not remove `create_default_profile_registry`; `_build_target_configs()` still uses it.

**Step 2: Change `_build_tui_mode_configs()`**

Replace:

```python
    registry = create_default_profile_registry(cfg)
```

with:

```python
    registry = create_tui_profile_registry(cfg)
```

**Step 3: Change `_resolve_tui_mode_configs()` fallback**

Replace:

```python
    if registry is None:
        registry = create_default_profile_registry(cfg)
```

with:

```python
    if registry is None:
        registry = create_tui_profile_registry(cfg)
```

**Step 4: Do not change `_build_target_configs()` in this task**

Reason: `_build_target_configs()` is not used by `_run_tui()` and broader CLI mode semantics are outside this bug. If product wants CLI mode commands to honor TUI-edited profiles later, create a separate issue.

---

### Task 4: Verify Regression Tests Pass

**Files:**
- Test: `src/tests/cli/test_server_runner.py`

**Step 1: Run the two focused tests**

Run:
```bash
rtk uv run pytest \
  src/tests/cli/test_server_runner.py::TestBuildTuiModeConfigs::test_build_tui_mode_configs_uses_custom_builtin_override \
  src/tests/cli/test_server_runner.py::TestResolveTuiModeConfigs::test_resolve_mode_configs_uses_tui_registry_when_registry_not_supplied \
  -q 2>&1 | distill "Did both focused profile persistence tests pass? Return PASS or FAIL, followed by failing test names."
```

Expected: `PASS`.

**Step 2: Run nearby server runner tests**

Run:
```bash
rtk uv run pytest src/tests/cli/test_server_runner.py -q 2>&1 | distill "Did all server_runner tests pass? Return PASS or FAIL, followed by failing test names."
```

Expected: `PASS`.

---

### Task 5: Verify Existing Profile Store And Registry Tests Still Pass

**Files:**
- Test: `src/tests/config/test_slot_profile_store.py`
- Test: `src/tests/config/test_tui_profile_registry.py`
- Test: `src/tests/config/test_tui_profile_registry_hidden.py`

**Step 1: Run registry/store tests**

Run:
```bash
rtk uv run pytest \
  src/tests/config/test_slot_profile_store.py \
  src/tests/config/test_tui_profile_registry.py \
  src/tests/config/test_tui_profile_registry_hidden.py \
  -q 2>&1 | distill "Did profile store and TUI registry tests pass? Return PASS or FAIL, followed by failing test names."
```

Expected: `PASS`.

---

### Task 6: Run Required Local Gate Before Commit

**Files:**
- All changed code/test files.

**Step 1: Run pre-commit**

Run:
```bash
rtk uv run pre-commit run --all-files 2>&1 | distill "Did pre-commit pass? Return PASS or FAIL, followed by hook names that failed."
```

Expected: `PASS`.

**Step 2: Run full tests**

Run:
```bash
rtk uv run pytest 2>&1 | distill "Did the full pytest suite pass? Return PASS or FAIL, followed by failing test names."
```

Expected: `PASS`.

**Step 3: Inspect final diff**

Run:
```bash
rtk git diff -- src/llama_cli/server_runner.py src/tests/cli/test_server_runner.py 2>&1 | distill "Summarize final code changes. Return file paths and one bullet per behavior change."
```

Expected:
- `src/llama_cli/server_runner.py`: TUI launch mode registry now includes persisted custom overrides.
- `src/tests/cli/test_server_runner.py`: regression coverage for built-in overrides on TUI restart and fallback resolution.

---

### Task 7: Commit

**Files:**
- Modify: `src/llama_cli/server_runner.py`
- Modify: `src/tests/cli/test_server_runner.py`
- Do not include this plan file unless the user explicitly wants plan docs committed with the fix.

**Step 1: Stage code and tests**

Run:
```bash
rtk git add src/llama_cli/server_runner.py src/tests/cli/test_server_runner.py 2>&1 | distill "Did git add succeed? Return PASS or FAIL with error summary."
```

Expected: `PASS`.

**Step 2: Commit**

Run:
```bash
rtk git commit -m "fix: honor persisted TUI profile overrides on restart" 2>&1 | distill "Did git commit succeed? Return PASS or FAIL with commit hash or error summary."
```

Expected: `PASS`.

---

## Acceptance Criteria

- Editing built-in `summary-balanced` in the TUI creates or updates a custom override in `slot_profiles.toml`.
- Restarting the TUI with `llm-runner tui summary-balanced` or equivalent mode launch uses the edited profile values.
- `both` mode uses edited `summary-balanced` and `qwen35` overrides when present.
- Port overrides from CLI args still win over stored profile ports.
- Existing hidden built-in behavior still works.
- Full local gate passes: `uv run pre-commit run --all-files` and `uv run pytest`.

## Non-Goals

- Do not make smoke/profile/dry-run commands use TUI-edited overrides.
- Do not change profile TOML format.
- Do not rename profile concepts or create compatibility shims.
- Do not delete or migrate existing user profile files.
