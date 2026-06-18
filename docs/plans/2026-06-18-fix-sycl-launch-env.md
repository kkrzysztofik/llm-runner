# Fix SYCL Launch Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the TUI `summary-balanced` SYCL slot start under the same oneAPI runtime environment that makes the direct `llama-server` launch work.

**Architecture:** Keep command construction pure and keep process launching argv-based. Add a small launch-time SYCL wrapper only at the `ServerManager.start_servers()` boundary, where the code still has the `ServerConfig.device` value and can distinguish `SYCL0` from CUDA/non-SYCL slots.

**Tech Stack:** Python 3.12, pytest, Textual TUI process orchestration, Intel oneAPI `setvars.sh`, llama.cpp SYCL backend.

---

## Root Cause Summary

The screen error is not a TUI rendering bug and not a GDB wrapper configured by the app. The SYCL `llama-server` process starts, throws a `sycl::_V1::exception`, and llama.cpp/ggml prints a GDB backtrace. The decisive line in the persisted log is:

```text
terminate called after throwing an instance of 'sycl::_V1::exception'
what():  No device of requested type available.
```

Evidence gathered on 2026-06-18:

- TUI log: `/home/kmk/.local/state/llm-runner/logs/llm-runner-20260618-005532.log`
- Crash path:
  - `ggml_sycl_init()`
  - `dpct::dev_mgr::dev_mgr()`
  - `sycl::_V1::detail::select_device(...)`
  - `what(): No device of requested type available`
- Direct unsourced launch of the same summary command: `REPRO_CRASH`
- Direct launch after `source /opt/intel/oneapi/setvars.sh --force`: `STARTED`
- `sycl-ls` is not on PATH before `setvars.sh`
- `sycl-ls` after `setvars.sh` sees the B580:

```text
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Arc(TM) B580 Graphics
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) B580 Graphics
```

Selector-only attempts with `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and `SYCL_DEVICE_FILTER=level_zero:gpu` still crashed. The missing piece is the full oneAPI runtime setup from `setvars.sh`, not just a selector env var.

Important nearby fact: `summary-balanced` currently leaves `ServerConfig.server_bin == ""`, so `build_server_cmd()` falls back to `Config().paths.llama_server_bin_intel`. That is asymmetric with `qwen35`, which embeds the CUDA binary path in the built-in profile. This is worth preserving in notes, but it is not the primary failure proven here.

## Constraints

- Do not delete files.
- Do not edit Speckit plan/spec files for this fix.
- Do not use script-based code rewrites.
- Keep `build_server_cmd()` returning a plain `list[str]`.
- Preserve non-SYCL launch behavior exactly.
- Keep subprocess launch non-interactive.
- Per `AGENTS.md`, all shell commands must pipe through `distill` unless exact raw output is required.

## Proposed Fix

Add a helper that wraps only SYCL launch commands as:

```python
[
    "bash",
    "-c",
    (
        'if ! source "$1" --force >/dev/null 2>&1; then '
        'echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; '
        "exit 127; "
        "fi; "
        "shift; "
        'exec "$@"'
    ),
    "llm-runner-sycl-launch",
    "/opt/intel/oneapi/setvars.sh",
    *original_cmd,
]
```

Rationale:

- `bash -c` is needed because `source` is a shell builtin.
- `exec "$@"` replaces the shell with `llama-server`, so the tracked PID becomes the server process.
- The server argv remains separate args after the fixed shell snippet; no string-joined command is needed.
- `setvars.sh` stdout/stderr is suppressed on success so the TUI server log does not get oneAPI banner noise.
- On setvars failure, stderr gets one concise launch error.

## Task 1: Add Unit Tests For SYCL Wrapper Helper

**Files:**

- Modify: `src/tests/runtime/test_launcher.py`
- Modify later: `src/llama_manager/orchestration/launcher.py`

**Step 1: Write failing tests**

Add tests near `TestDefaultProcessLauncher` or a new `TestSyclLaunchWrapper` class:

```python
def test_wrap_sycl_launch_cmd_sources_setvars_when_device_is_sycl(tmp_path: Path) -> None:
    setvars = tmp_path / "setvars.sh"
    setvars.write_text("# test\n")
    cmd = ["/bin/echo", "hello world"]

    wrapped = wrap_sycl_launch_cmd(cmd, "SYCL0", setvars_path=setvars)

    assert wrapped[:3] == [
        "bash",
        "-c",
        (
            'if ! source "$1" --force >/dev/null 2>&1; then '
            'echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; '
            "exit 127; "
            "fi; "
            "shift; "
            'exec "$@"'
        ),
    ]
    assert wrapped[3] == "llm-runner-sycl-launch"
    assert wrapped[4] == str(setvars)
    assert wrapped[5:] == cmd
    assert cmd == ["/bin/echo", "hello world"]
```

```python
def test_wrap_sycl_launch_cmd_leaves_non_sycl_device_unchanged(tmp_path: Path) -> None:
    setvars = tmp_path / "setvars.sh"
    setvars.write_text("# test\n")
    cmd = ["/bin/echo", "cuda"]

    wrapped = wrap_sycl_launch_cmd(cmd, "CUDA0", setvars_path=setvars)

    assert wrapped is cmd
```

```python
def test_wrap_sycl_launch_cmd_leaves_command_unchanged_when_setvars_missing(tmp_path: Path) -> None:
    cmd = ["/bin/echo", "sycl"]

    wrapped = wrap_sycl_launch_cmd(cmd, "SYCL0", setvars_path=tmp_path / "missing.sh")

    assert wrapped is cmd
```

**Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest src/tests/runtime/test_launcher.py -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing test names and first missing symbol/error."
```

Expected: `FAIL` because `wrap_sycl_launch_cmd` does not exist.

## Task 2: Implement The Wrapper Helper

**Files:**

- Modify: `src/llama_manager/orchestration/launcher.py`

**Step 1: Add imports/constants**

Add `Path` import:

```python
from pathlib import Path
```

Add near module constants:

```python
_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")
_SYCL_LAUNCH_SCRIPT = (
    'if ! source "$1" --force >/dev/null 2>&1; then '
    'echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; '
    "exit 127; "
    "fi; "
    "shift; "
    'exec "$@"'
)
```

**Step 2: Add helper functions**

```python
def _is_sycl_device(device: str) -> bool:
    return device.upper().startswith("SYCL")
```

```python
def wrap_sycl_launch_cmd(
    cmd: list[str],
    device: str,
    setvars_path: Path = _INTEL_SETVARS_SH,
) -> list[str]:
    """Wrap SYCL server launches with Intel oneAPI runtime setup."""
    if not _is_sycl_device(device):
        return cmd
    if not setvars_path.exists():
        return cmd
    return [
        "bash",
        "-c",
        _SYCL_LAUNCH_SCRIPT,
        "llm-runner-sycl-launch",
        str(setvars_path),
        *cmd,
    ]
```

**Step 3: Run focused tests**

Run:

```bash
uv run pytest src/tests/runtime/test_launcher.py::TestSyclLaunchWrapper -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing assertions."
```

Expected: `PASS`.

## Task 3: Apply The Wrapper In ServerManager.start_servers

**Files:**

- Modify: `src/llama_manager/orchestration/manager.py`
- Modify: `src/tests/runtime/test_launcher.py`

**Step 1: Write failing integration tests**

Add tests under `TestProcessLauncherProtocol` because it already uses `MockProcessLauncher`:

```python
def test_start_servers_wraps_sycl_config_with_oneapi_setvars(
    self,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_manager.orchestration import ServerManager
    from llama_manager.orchestration import launcher as launcher_module
    from tests.support.helpers import make_server_config

    setvars = tmp_path / "setvars.sh"
    setvars.write_text("# test\n")
    monkeypatch.setattr(launcher_module, "_INTEL_SETVARS_SH", setvars)

    mock_launcher = MockProcessLauncher()
    manager = ServerManager(process_launcher=mock_launcher)  # pyright: ignore[arg-type]
    cfg = make_server_config(alias="summary-balanced", device="SYCL0", server_bin="/bin/echo")

    manager.start_servers([cfg], {})

    launched = mock_launcher.launch_calls[0]
    assert launched[0:2] == ["bash", "-c"]
    assert launched[3] == "llm-runner-sycl-launch"
    assert launched[4] == str(setvars)
    assert launched[5] == "/bin/echo"
```

```python
def test_start_servers_does_not_wrap_cuda_config(
    self,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_manager.orchestration import ServerManager
    from llama_manager.orchestration import launcher as launcher_module
    from tests.support.helpers import make_server_config

    setvars = tmp_path / "setvars.sh"
    setvars.write_text("# test\n")
    monkeypatch.setattr(launcher_module, "_INTEL_SETVARS_SH", setvars)

    mock_launcher = MockProcessLauncher()
    manager = ServerManager(process_launcher=mock_launcher)  # pyright: ignore[arg-type]
    cfg = make_server_config(alias="qwen35-coding", device="CUDA0", server_bin="/bin/echo")

    manager.start_servers([cfg], {})

    assert mock_launcher.launch_calls[0][0] == "/bin/echo"
```

**Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest src/tests/runtime/test_launcher.py -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing test names and first assertion."
```

Expected: `FAIL` for the SYCL integration test.

**Step 3: Implement integration**

In `src/llama_manager/orchestration/manager.py`, change `start_servers()`:

```python
from ..validation.commands import build_server_cmd
from .launcher import wrap_sycl_launch_cmd
```

Then inside the loop:

```python
cmd = build_server_cmd(cfg)
cmd = wrap_sycl_launch_cmd(cmd, cfg.device)
handler = log_handlers.get(cfg.alias) if log_handlers else None
proc = self.start_server_background(cfg.alias, cmd, handler)
```

**Step 4: Run focused tests**

Run:

```bash
uv run pytest src/tests/runtime/test_launcher.py -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing test names and first assertion."
```

Expected: `PASS`.

## Task 4: Add A Regression Test For TUI Add-Slot Launch Path

**Files:**

- Modify: `src/tests/tui/test_textual_app.py` or `src/tests/tui/test_controller.py`

**Step 1: Locate existing add-slot async launch tests**

Use ripgrep:

```bash
rg -n "_run_add_slot|start_servers|summary-balanced|SYCL0" src/tests/tui src/llama_cli/tui 2>&1 | distill "Return test names and source lines for TUI add-slot launch path."
```

**Step 2: Add or extend a test**

Preferred: extend the existing test that covers adding `summary-balanced` from the TUI. Assert that the mocked `start_servers()` receives a `ServerConfig` with `device == "SYCL0"`. Do not assert the bash wrapper in the TUI test; that belongs to `ServerManager` tests.

Example assertion pattern:

```python
cfgs_arg = controller.server_manager.start_servers.call_args.args[0]
assert cfgs_arg[0].alias == "summary-balanced"
assert cfgs_arg[0].device == "SYCL0"
```

If no suitable mocked TUI test exists, skip this task and document why in the final implementation notes. Do not create a broad slow Textual integration test just for this wrapper.

**Step 3: Run focused TUI test**

Run:

```bash
uv run pytest src/tests/tui/test_textual_app.py -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing test names and first assertion."
```

Expected: `PASS`.

## Task 5: Manual Runtime Verification

**Files:**

- No code changes.

**Step 1: Verify direct crash still describes the root cause**

Run only if manual runtime validation is acceptable on the workstation:

```bash
timeout 18 /home/kmk/.cache/llm-runner/llama.cpp/build/bin/llama-server --model /home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf --alias summary-balanced --n-gpu-layers 99 --split-mode layer --ctx-size 16144 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 256 --threads 4 --poll 500 --n-predict -1 --parallel 4 --host 127.0.0.1 --port 18080 --no-webui --device SYCL0 --reasoning off --reasoning-format deepseek --jinja --no-mmproj-offload --mmap 2>&1 | distill "Return REPRO_CRASH/STARTED/TIMEOUT/OTHER, first fatal exception, SYCL device error, and whether GDB backtrace appears."
```

Expected before fix or without wrapper: `REPRO_CRASH` with `No device of requested type available`.

**Step 2: Verify wrapper command starts**

After implementation, run a bounded command equivalent to the wrapper:

```bash
timeout 18 bash -c 'if ! source "$1" --force >/dev/null 2>&1; then echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; exit 127; fi; shift; exec "$@"' llm-runner-sycl-launch /opt/intel/oneapi/setvars.sh /home/kmk/.cache/llm-runner/llama.cpp/build/bin/llama-server --model /home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf --alias summary-balanced --n-gpu-layers 99 --split-mode layer --ctx-size 16144 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 256 --threads 4 --poll 500 --n-predict -1 --parallel 4 --host 127.0.0.1 --port 18081 --no-webui --device SYCL0 --reasoning off --reasoning-format deepseek --jinja --no-mmproj-offload --mmap 2>&1 | distill "Return STARTED if server reaches listening/model-ready output before timeout; return REPRO_CRASH if SYCL exception appears; include first readiness or exception line."
```

Expected: `STARTED` or timeout after successful startup progress, with no `sycl::_V1::exception`.

**Step 3: Verify TUI path**

Run:

```bash
uv run llm-runner 2>&1 | distill "This is a TUI command; if distill breaks interactivity, stop and rerun manually. Pass criteria: add summary-balanced slot and verify no SYCL exception appears in log."
```

If `distill` breaks the TUI, this is one of the allowed exceptions. Run interactively:

```bash
uv run llm-runner
```

Pass criteria:

- Add `summary-balanced`.
- Slot remains running.
- Log does not show `No device of requested type available`.
- Log does not show the GDB backtrace shown in the original screenshot.

## Task 6: Run Gates

**Files:**

- No code changes unless failures require fixes.

**Step 1: Run focused tests**

```bash
uv run pytest src/tests/runtime/test_launcher.py src/tests/tui/test_textual_app.py -q 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing test names and first assertion."
```

Expected: `PASS`.

**Step 2: Run mandatory local gate before commit/push**

```bash
uv run pre-commit run --all-files 2>&1 | distill "Return PASS or FAIL. If FAIL, list hook names and files."
```

Expected: `PASS`.

```bash
uv run pytest 2>&1 | distill "Return PASS or FAIL. If FAIL, list failing tests and error summary."
```

Expected: `PASS`.

## Pass Criteria

- `summary-balanced` launch command is wrapped with oneAPI `setvars.sh` only when `cfg.device` starts with `SYCL`.
- CUDA and non-SYCL commands are unchanged.
- Unit tests prove the wrapper preserves original argv after the fixed shell prefix.
- Integration tests prove `ServerManager.start_servers()` applies the wrapper for SYCL configs.
- Bounded manual launch through the wrapper starts instead of throwing `No device of requested type available`.
- Mandatory local gate passes:
  - `uv run pre-commit run --all-files`
  - `uv run pytest`

## Do Not Do

- Do not add a compatibility shim.
- Do not move this into the TUI layer only; CLI/TUI share `ServerManager.start_servers()`.
- Do not add global environment mutations in Python with `os.environ`.
- Do not rely only on `ONEAPI_DEVICE_SELECTOR` or `SYCL_DEVICE_FILTER`; selector-only probes still crashed.
- Do not suppress server stderr broadly; only suppress successful `setvars.sh` banner noise.
