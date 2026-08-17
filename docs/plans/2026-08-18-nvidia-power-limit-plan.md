# NVIDIA GPU Default Power Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply a default 290 W power cap to every NVIDIA GPU used by a launched profile, set once per launch before the llama-server process starts.

**Architecture:** A `nvidia_power_limit_watts` global default on `ServerDefaultsConfig` (0 = disabled). A small pure module `orchestration/power_limit.py` parses CUDA ordinals from a device string and runs `sudo -n nvidia-smi -i <idx> -pl <watts>` best-effort. The hook fires in `ServerManager.start_servers` before `build_server_cmd`, so CLI launch, TUI boot, and TUI add/replace slot all get the cap in one place. Dry-run reports the planned cap in `hardware_notes`. The Config modal edits the value.

**Tech Stack:** Python 3.14, subprocess, dataclasses, Textual (modal), pytest (unit only — no GPU, no real nvidia-smi).

## Global Constraints

- Python ≥ 3.14, type hints on all new functions, line length 100 (ruff).
- No GPU hardware, no subprocess llama-server, no real nvidia-smi in tests — mock `subprocess.run` and `apply_nvidia_power_limit`.
- `llama_manager` stays a pure library — no argparse, no Rich; the hook takes a `warn: Callable[[str], None]` callback so the manager stays decoupled from `Config` (watts passed as a param).
- Spec: `docs/plans/2026-08-18-nvidia-power-limit-design.md` — read it first.
- Must not break existing `start_servers(configs, log_handlers)` calls — new param is optional with default `0`.

---

### Task 1: `nvidia_power_limit_watts` on `ServerDefaultsConfig`

**Files:**
- Modify: `src/llama_manager/config/defaults.py` (add field to `ServerDefaultsConfig`, ~line 113)
- Test: `src/tests/config/test_config_builders.py:48` (`TestConfig.test_defaults_are_set`)

**Interfaces:**
- Produces: `ServerDefaultsConfig.nvidia_power_limit_watts: int = 290` — read by Task 3 (launch hook), Task 4 (dry-run), Task 5 (Config modal).

- [ ] **Step 1: Write the failing test**

Add to `test_defaults_are_set` in `src/tests/config/test_config_builders.py`:

```python
    def test_nvidia_power_limit_watts_default(self) -> None:
        cfg = Config()
        assert cfg.server_defaults.nvidia_power_limit_watts == 290
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/config/test_config_builders.py::TestConfig::test_nvidia_power_limit_watts_default -v`
Expected: FAIL with `AttributeError: 'ServerDefaultsConfig' object has no attribute 'nvidia_power_limit_watts'`

- [ ] **Step 3: Write minimal implementation**

In `src/llama_manager/config/defaults.py`, inside `class ServerDefaultsConfig`, add a field with the other flat defaults (e.g. right after `n_gpu_layers_profile: str = "all"`):

```python
    # 0 disables the per-launch power cap; 290 W is below the RTX 3090's 350 W stock limit.
    nvidia_power_limit_watts: int = 290
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/tests/config/test_config_builders.py::TestConfig::test_nvidia_power_limit_watts_default -v`
Expected: PASS

- [ ] **Step 5: Run the persistence-field auto-registration guard**

`_UPDATE_FIELDS` / `_COERCION_FIELDS` in `persistence.py` are derived automatically from dataclass fields via `_SECTION_CLASSES` + `fields()`, so `server_defaults.nvidia_power_limit_watts` is auto-registered as an int-coerced, persisted field. Verify with:

Run: `uv run pytest src/tests/config/test_config_persistence.py -v`
Expected: PASS (no existing test breaks; the new field is simply one more int-coerced field)

- [ ] **Step 6: Commit**

```bash
git add src/llama_manager/config/defaults.py src/tests/config/test_config_builders.py
git commit -m "feat: add nvidia_power_limit_watts server default (290W)"
```

---

### Task 2: `power_limit` module — ordinal parsing + best-effort apply

**Files:**
- Create: `src/llama_manager/orchestration/power_limit.py`
- Test: `src/tests/runtime/test_power_limit.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `cuda_ordinals(device: str) -> list[int]` — `"CUDA0,CUDA1"` → `[0, 1]`; `"cuda:0"` → `[0]`; `""`/`"auto"` → `[0]`; `"SYCL0"`/garbage → `[]`.
  - `apply_nvidia_power_limit(device: str, watts: int, warn: Callable[[str], None]) -> None` — no-op when `watts <= 0`; otherwise runs `sudo -n nvidia-smi -i <idx> -pl <watts>` per ordinal with a 2 s timeout, calling `warn(msg)` on failure and continuing.

- [ ] **Step 1: Write the failing tests**

Create `src/tests/runtime/test_power_limit.py`:

```python
"""Unit tests for orchestration.power_limit — mocked subprocess, no GPU."""

from __future__ import annotations

from unittest.mock import Mock, patch

from llama_manager.orchestration.power_limit import (
    apply_nvidia_power_limit,
    cuda_ordinals,
)


class TestCudaOrdinals:
    def test_parses_multi_device(self) -> None:
        assert cuda_ordinals("CUDA0,CUDA1") == [0, 1]

    def test_parses_dotted_cuda(self) -> None:
        assert cuda_ordinals("cuda:0") == [0]

    def test_empty_device_defaults_to_zero(self) -> None:
        assert cuda_ordinals("") == [0]

    def test_auto_defaults_to_zero(self) -> None:
        assert cuda_ordinals("auto") == [0]

    def test_parses_nonzero_indices(self) -> None:
        assert cuda_ordinals("cuda:1,2") == [1, 2]

    def test_sycl_returns_empty(self) -> None:
        assert cuda_ordinals("SYCL0") == []

    def test_garbage_returns_empty(self) -> None:
        assert cuda_ordinals("not-a-device") == []


class TestApplyNvidiaPowerLimit:
    def test_zero_watts_is_noop(self) -> None:
        warn = Mock()
        with patch("subprocess.run") as run:
            apply_nvidia_power_limit("cuda:0", 0, warn)
        run.assert_not_called()
        warn.assert_not_called()

    def test_applies_to_each_ordinal(self) -> None:
        warn = Mock()
        with patch("subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")) as run:
            apply_nvidia_power_limit("CUDA0,CUDA1", 290, warn)
        assert run.call_count == 2
        assert run.call_args_list[0].args[0] == [
            "sudo", "-n", "nvidia-smi", "-i", "0", "-pl", "290",
        ]
        assert run.call_args_list[1].args[0] == [
            "sudo", "-n", "nvidia-smi", "-i", "1", "-pl", "290",
        ]
        warn.assert_not_called()

    def test_nonzero_exit_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", return_value=Mock(returncode=1, stdout="", stderr="denied")):
            apply_nvidia_power_limit("CUDA0,CUDA1", 290, warn)
        assert warn.call_count == 2

    def test_oserror_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", side_effect=OSError("no sudo")):
            apply_nvidia_power_limit("cuda:0", 290, warn)
        warn.assert_called_once()

    def test_timeout_warns_and_continues(self) -> None:
        warn = Mock()
        with patch("subprocess.run", side_effect=TimeoutError("hung")):
            apply_nvidia_power_limit("cuda:0", 290, warn)
        warn.assert_called_once()

    def test_sycl_device_noop(self) -> None:
        warn = Mock()
        with patch("subprocess.run") as run:
            apply_nvidia_power_limit("SYCL0", 290, warn)
        run.assert_not_called()
        warn.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/tests/runtime/test_power_limit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llama_manager.orchestration.power_limit'`

- [ ] **Step 3: Write minimal implementation**

Create `src/llama_manager/orchestration/power_limit.py`:

```python
"""NVIDIA GPU power-cap side effects (best-effort, per-launch)."""

from collections.abc import Callable
import subprocess

_TIMEOUT_S = 2


def cuda_ordinals(device: str) -> list[int]:
    """Return the CUDA ordinals referenced by a device string.

    Empty/auto devices default to ``[0]`` (CUDA auto-detect). Non-CUDA
    devices and unparseable input return ``[]`` (no cap applied).
    """
    stripped = device.strip().upper()
    if not stripped or stripped == "AUTO":
        return [0]
    if not stripped.startswith("CUDA"):
        return []
    rest = stripped[4:].lstrip(":")
    if not rest:
        return [0]
    ordinals: list[int] = []
    for part in rest.split(","):
        part = part.strip()
        if part.isdigit():
            ordinals.append(int(part))
    return ordinals


def apply_nvidia_power_limit(
    device: str, watts: int, warn: Callable[[str], None]
) -> None:
    """Apply a power cap to every CUDA device in *device* (best-effort).

    ``watts <= 0`` disables the cap. Failures are reported through *warn*
    and never raised, so a missing driver or missing sudo permission cannot
    block a server launch.
    """
    if watts <= 0:
        return
    for ordinal in cuda_ordinals(device):
        try:
            result = subprocess.run(
                ["sudo", "-n", "nvidia-smi", "-i", str(ordinal), "-pl", str(watts)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            warn(f"failed to set NVIDIA power limit {watts}W on GPU {ordinal}")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            warn(
                f"failed to set NVIDIA power limit {watts}W on GPU {ordinal}"
                + (f": {detail}" if detail else "")
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/tests/runtime/test_power_limit.py -v`
Expected: PASS (all 14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llama_manager/orchestration/power_limit.py src/tests/runtime/test_power_limit.py
git commit -m "feat: nvidia power-limit apply helper (best-effort sudo nvidia-smi)"
```

---

### Task 3: Hook into `ServerManager.start_servers` + wire all launch call sites

**Files:**
- Modify: `src/llama_manager/orchestration/manager.py:188-216` (`start_servers`)
- Modify: `src/llama_manager/orchestration/launch.py:124-137` (`_start_and_map_servers`) and `:193-239` (`launch_orchestrate`)
- Modify: `src/llama_cli/tui/textual_app.py:680` (TUI add/replace slot direct call)
- Modify: `src/llama_manager/slot_manager.py:118-149` (`register_and_start_slot`)
- Test: `src/tests/runtime/test_launcher.py` (start_servers tests), `src/tests/runtime/test_launch_flow.py`

**Interfaces:**
- Consumes: `apply_nvidia_power_limit(device, watts, warn)` from Task 2.
- Produces: `ServerManager.start_servers(configs, log_handlers=None, power_limit_watts: int = 0)`; `_start_and_map_servers(..., power_limit_watts: int = 0)`; `register_and_start_slot(..., power_limit_watts: int = 0)`.

- [ ] **Step 1: Write the failing tests**

In `src/tests/runtime/test_launcher.py`, add a test class after the existing `start_servers` tests:

```python
class TestStartServersPowerLimit:
    def _cuda_cfg(self) -> ServerConfig:
        return ServerConfig(
            model="/models/qwen.gguf",
            alias="qwen",
            device="CUDA0,CUDA1",
            port=8081,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )

    def _start(self, watts: int) -> Mock:
        """Run start_servers with launch internals patched out."""
        manager = ServerManager(process_launcher=Mock())
        with (
            patch("llama_manager.orchestration.manager.apply_nvidia_power_limit") as apply_patch,
            patch.object(manager, "_reserve_slot_lock"),
            patch.object(manager, "_record_started_slot_lock"),
            patch.object(manager, "start_server_background", return_value=Mock()),
        ):
            manager.start_servers([self._cuda_cfg()], {}, power_limit_watts=watts)
        return apply_patch

    def test_applies_power_limit_for_cuda_when_watts_set(self) -> None:
        apply_patch = self._start(290)
        apply_patch.assert_called_once()
        args = apply_patch.call_args.args
        assert args[0] == "CUDA0,CUDA1"
        assert args[1] == 290

    def test_skips_power_limit_when_zero(self) -> None:
        apply_patch = self._start(0)
        apply_patch.assert_not_called()
```

In `src/tests/runtime/test_launch_flow.py`, add a test that `launch_orchestrate` forwards the configured watts:

```python
    def test_launch_orchestrate_forwards_power_limit_watts(self) -> None:
        cfg = ServerConfig(
            model="/models/qwen.gguf",
            alias="qwen35",
            device="CUDA0",
            port=8081,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        base = Config()
        base.server_defaults.nvidia_power_limit_watts = 290
        mock_sm = Mock()
        mock_sm.begin_launch_attempt.return_value = "launch-1"
        mock_sm.launch_all_slots.return_value = LaunchResult(status="success", launched=["qwen35"])
        mock_sm.start_servers.return_value = [Mock()]
        result = launch_orchestrate(
            [cfg], base, mock_sm, log_buffers={}, get_driver_version=lambda _: "v1",
        )
        assert result.launch_result.status == "success"
        mock_sm.start_servers.assert_called_once()
        assert mock_sm.start_servers.call_args.kwargs["power_limit_watts"] == 290
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/tests/runtime/test_launcher.py::TestStartServersPowerLimit src/tests/runtime/test_launch_flow.py::TestLaunchOrchestrate::test_launch_orchestrate_forwards_power_limit_watts -v`

Expected: FAIL — `start_servers()` got an unexpected keyword argument `power_limit_watts` (and `apply_nvidia_power_limit` not importable at `llama_manager.orchestration.manager`).

- [ ] **Step 3: Write minimal implementation**

In `src/llama_manager/orchestration/manager.py`:

1. Add `import logging` to the stdlib imports, and after the imports block: `logger = logging.getLogger(__name__)`.
2. Add the module-level import with the other `.launcher` imports at the top:

```python
from .launcher import (
    ProcessHandle,
    ProcessLauncher,
    ProcessTimeoutError,
    filter_owned_running_pids,
    send_signals_to_pids,
    stream_pipe,
    wait_for_processes,
)
from .power_limit import apply_nvidia_power_limit
```

Importing at module scope (not lazily) matters: the tests patch `llama_manager.orchestration.manager.apply_nvidia_power_limit`, and a lazy `from .power_limit import ...` inside the function would re-bind the module attribute and clobber the patch. `power_limit` only imports stdlib, so there is no import-cycle risk.

Then the method:

```python
    def start_servers(
        self,
        configs: list[ServerConfig],
        log_handlers: dict[str, Callable[[str], None]] | None = None,
        power_limit_watts: int = 0,
    ) -> list[ProcessHandle]:
        """Start multiple servers and return their processes.

        When ``power_limit_watts > 0``, applies an NVIDIA power cap to every
        CUDA device of each config before launching (best-effort; failures
        only log a warning and never block the launch).
        """
        from ..validation.commands import build_server_cmd
        from .launcher import wrap_sycl_launch_cmd

        log_handlers = log_handlers or {}
        processes = []
        for cfg in configs:
            try:
                if isinstance(power_limit_watts, int) and power_limit_watts > 0:
                    apply_nvidia_power_limit(cfg.device, power_limit_watts, logger.warning)
                self._reserve_slot_lock(cfg)
                cmd = build_server_cmd(cfg)
                cmd = wrap_sycl_launch_cmd(cmd, cfg.device)
                handler = log_handlers.get(cfg.alias) if log_handlers else None
                proc = self.start_server_background(cfg.alias, cmd, handler)
            except Exception:
                self.release_lock(cfg.alias)
                raise
            try:
                self._record_started_slot_lock(cfg, proc.pid)
            except Exception:
                self._shutdown_process_handle(cfg.alias, proc, timeout=1.0)
                raise
            self.slot_processes[cfg.alias] = proc
            processes.append(proc)
        return processes
```

The `isinstance(power_limit_watts, int)` guard keeps characterization tests that pass a `Mock` base_config (whose `server_defaults.nvidia_power_limit_watts` is a truthy Mock) from invoking the helper at all.

In `src/llama_manager/orchestration/launch.py`:

```python
def _start_and_map_servers(
    launched_configs: list[ServerConfig],
    log_handlers: dict[str, Callable[[str], None]],
    server_manager: ServerManager,
    power_limit_watts: int = 0,
) -> dict[str, Any]:
    """Start servers and map processes by alias."""
    processes: dict[str, Any] = {}
    try:
        processes_list = server_manager.start_servers(
            launched_configs, log_handlers, power_limit_watts=power_limit_watts
        )
    except Exception:
        server_manager.cleanup_servers()
        raise

    for cfg, proc in zip(launched_configs, processes_list, strict=True):
        processes[cfg.alias] = proc

    return processes
```

And in `launch_orchestrate`, change the call at line ~239:

```python
    processes = _start_and_map_servers(
        launched_configs,
        log_handlers,
        server_manager,
        power_limit_watts=base_config.server_defaults.nvidia_power_limit_watts,
    )
```

In `src/llama_cli/tui/textual_app.py` (line ~680):

```python
            procs = self.controller.server_manager.start_servers(
                [new_cfg],
                {stage.alias: log_handler},
                power_limit_watts=self.controller.config.server_defaults.nvidia_power_limit_watts,
            )
```

In `src/llama_manager/slot_manager.py` `register_and_start_slot`:

```python
def register_and_start_slot(
    cfg: ServerConfig,
    server_manager: ServerManager,
    state: dict[str, Any],
    startup_callback: Callable[[], None] | None = None,
    power_limit_watts: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    ...
    procs = server_manager.start_servers(
        [cfg], {alias: log_handler}, power_limit_watts=power_limit_watts
    )
```

`add_slot_from_form` / `_upsert_profile_slot` keep their calls unchanged (they default to `power_limit_watts=0` — the sync add-slot path is superseded by the TUI async path at `textual_app.py:680`). Mark that intent with a `# ponytail: sync add-slot path is test/legacy-only; TUI uses the async start_servers call` comment on the `register_and_start_slot` default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/tests/runtime/test_launcher.py src/tests/runtime/test_launch_flow.py -v`
Expected: PASS (new tests + all existing start_servers/launch tests — existing callers use positional args so the new optional param is compatible)

- [ ] **Step 5: Run the wider orchestration suite**

Run: `uv run pytest src/tests/runtime src/tests/slot -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/llama_manager/orchestration/manager.py src/llama_manager/orchestration/launch.py src/llama_cli/tui/textual_app.py src/llama_manager/slot_manager.py src/tests/runtime/test_launcher.py src/tests/runtime/test_launch_flow.py
git commit -m "feat: apply nvidia power cap before server launch (CLI + TUI)"
```

---

### Task 4: Dry-run `hardware_notes.power_limit_watts` + CLI output line

**Files:**
- Modify: `src/llama_manager/validation/commands/builder.py:245-287` (`build_dry_run_slot_payload`), `:330-342` (`_build_hardware_notes`)
- Modify: `src/llama_manager/dry_run.py:145-184` (`_build_dry_run_result`), `:102` (call site)
- Modify: `src/llama_cli/commands/dry_run.py:75-97` (`_print_resolved_slot`)
- Test: `src/tests/server/test_dry_run_schema.py` (hardware_notes tests), `src/tests/cli/test_server_runner.py` (output line test)

**Interfaces:**
- Consumes: `build_dry_run_slot_payload` gains optional kwarg `power_limit_watts: int = 0`.
- Produces: `hardware_notes["power_limit_watts"]` = int or `None` (None when 0); dry-run CLI prints `Power limit: <N> W`.

- [ ] **Step 1: Write the failing tests**

In `src/tests/server/test_dry_run_schema.py`, add:

```python
    def test_hardware_notes_power_limit_watts(self) -> None:
        """FR-003: hardware_notes should carry the configured power cap."""
        from tests.support.helpers import make_server_config

        cfg = make_server_config(alias="qwen35", model="/m.gguf", port=8081, device="CUDA0")
        payload = build_dry_run_slot_payload(
            cfg, slot_id="qwen35", power_limit_watts=290
        )
        assert payload.hardware_notes["power_limit_watts"] == 290

    def test_hardware_notes_power_limit_disabled_when_zero(self) -> None:
        from tests.support.helpers import make_server_config

        cfg = make_server_config(alias="qwen35", model="/m.gguf", port=8081, device="CUDA0")
        payload = build_dry_run_slot_payload(cfg, slot_id="qwen35", power_limit_watts=0)
        assert payload.hardware_notes["power_limit_watts"] is None
```

In `src/tests/cli/test_server_runner.py`, after `test_print_resolved_slot_includes_thinking_level`:

```python
def test_print_resolved_slot_includes_power_limit(capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run slot print shows the planned NVIDIA power cap."""
    from llama_cli.commands.dry_run import _print_resolved_slot
    from llama_manager.validation import build_dry_run_slot_payload
    from tests.support.helpers import make_server_config

    server_cfg = make_server_config(
        alias="qwen35", model="/models/qwen.gguf", port=8081, device="CUDA0",
    )
    payload = build_dry_run_slot_payload(
        server_cfg, slot_id="qwen35", power_limit_watts=290
    )
    _print_resolved_slot("qwen35", server_cfg, payload)

    captured = capsys.readouterr()
    assert "Power limit: 290 W" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/tests/server/test_dry_run_schema.py::TestDryRunSchema::test_hardware_notes_power_limit_watts src/tests/server/test_dry_run_schema.py::TestDryRunSchema::test_hardware_notes_power_limit_disabled_when_zero src/tests/cli/test_server_runner.py::test_print_resolved_slot_includes_power_limit -v`
Expected: FAIL — `power_limit_watts` unexpected kwarg and/or `'power_limit_watts'` key missing.

- [ ] **Step 3: Write minimal implementation**

In `src/llama_manager/validation/commands/builder.py`:

1. Widen the payload field type (line ~50) from `dict[str, str | None]` to `dict[str, str | int | None]`:

```python
    hardware_notes: dict[str, str | int | None]
```

2. Update the builder signature and helper:

```python
def build_dry_run_slot_payload(
    cfg: ServerConfig,
    slot_id: str,
    validation_results: DryRunValidationSummary | None = None,
    warnings: list[str] | None = None,
    power_limit_watts: int = 0,
) -> DryRunSlotPayload:
    ...
    hardware_notes = _build_hardware_notes(cfg, power_limit_watts)
    ...

def _build_hardware_notes(
    cfg: ServerConfig, power_limit_watts: int = 0
) -> dict[str, str | int | None]:
    """Build hardware notes dict describing backend and hardware."""
    backend = cfg.backend or "llama_cpp"
    device = cfg.device or "auto"
    device_id, device_name = _parse_device_details(device)

    return {
        "backend": backend,
        "device_id": device_id,
        "device_name": device_name,
        "power_limit_watts": power_limit_watts if power_limit_watts > 0 else None,
        "driver_version": None,
        "runtime_version": None,
    }
```

Existing schema tests that assert each `hardware_notes` value is `str | None` iterate only the specific keys `backend`/`device_id`/`device_name` (see `test_dry_run_schema.py:964`), so widening the dict type does not break them.
```

In `src/llama_manager/dry_run.py`:

```python
def _build_dry_run_result(
    mode: str,
    configs: list,
    profile_ids: tuple[str, ...],
    acknowledged: bool,
    power_limit_watts: int = 0,
) -> DryRunResult:
    ...
        slot_payloads.append(
            build_dry_run_slot_payload(
                server_cfg,
                slot_id=slot_id,
                validation_results=DryRunValidationSummary(passed=True, checks=[]),
                warnings=[],
                power_limit_watts=power_limit_watts,
            )
        )
    ...
```

And at the call site in `run_dry_run` (line ~102):

```python
    return _build_dry_run_result(
        mode,
        configs,
        profile_ids,
        acknowledged,
        power_limit_watts=config.server_defaults.nvidia_power_limit_watts,
    )
```

In `src/llama_cli/commands/dry_run.py` `_print_resolved_slot`, after the KV-cache line:

```python
    power_limit = payload.hardware_notes.get("power_limit_watts")
    if power_limit:
        emit_info(f"  Power limit: {power_limit} W")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/tests/server/test_dry_run_schema.py src/tests/cli/test_server_runner.py::test_print_resolved_slot_includes_power_limit -v`
Expected: PASS

- [ ] **Step 5: Run the dry-run suites**

Run: `uv run pytest src/tests/server src/tests/cli -q`
Expected: PASS (existing callers of `build_dry_run_slot_payload` don't pass the new kwarg and get `None` for `power_limit_watts`)

- [ ] **Step 6: Commit**

```bash
git add src/llama_manager/validation/commands/builder.py src/llama_manager/dry_run.py src/llama_cli/commands/dry_run.py src/tests/server/test_dry_run_schema.py src/tests/cli/test_server_runner.py
git commit -m "feat: report planned nvidia power cap in dry-run"
```

---

### Task 5: Config modal field + validation

**Files:**
- Modify: `src/llama_cli/tui/components/config_modal.py` (payload field, `to_config_updates`, validation, `_collect_values`)
- Modify: `src/llama_cli/tui/components/form_widgets.py:384-433` (`build_config_profile_defaults_collapsible`)
- Test: `src/tests/tui/test_config_modal.py`

**Interfaces:**
- Consumes: `ServerDefaultsConfig.nvidia_power_limit_watts` (Task 1), ConfigPayload pattern in `config_modal.py`.
- Produces: `ConfigPayload.default_nvidia_power_limit_watts: str`; modal Input id `#cfg-default_nvidia_power_limit_watts`; `to_config_updates()["server_defaults.nvidia_power_limit_watts"]` (raw string, coerced to int by `apply_config_updates`).

- [ ] **Step 1: Write the failing tests**

In `src/tests/tui/test_config_modal.py`:

1. Add to the `_DEFAULT_TO_SERVER_DEFAULTS` dict: `"default_nvidia_power_limit_watts": "nvidia_power_limit_watts",`
2. In `test_extended_server_defaults_collected`, add `default_nvidia_power_limit_watts=290,` to the `_make_config(...)` call and after line 417:

```python
        assert payload.default_nvidia_power_limit_watts == "290"
```

3. Add a new test:

```python
    @pytest.mark.anyio
    async def test_nvidia_power_limit_watts_roundtrip(self) -> None:
        """Config modal collects and maps the NVIDIA power limit."""
        config = _make_config(default_nvidia_power_limit_watts=0)
        modal = ConfigModal(config)
        app = ConfigModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            payload = modal._collect_values()

        assert payload.default_nvidia_power_limit_watts == "0"
        updates = payload.to_config_updates()
        assert updates["server_defaults.nvidia_power_limit_watts"] == "0"

    def test_invalid_nvidia_power_limit_watts_rejected(self) -> None:
        payload = ConfigPayload(default_nvidia_power_limit_watts="-5")
        errors = _validate_config_payload(payload)
        assert any("nvidia power limit" in err for err in errors)

        empty = ConfigPayload(default_nvidia_power_limit_watts="")
        errors = _validate_config_payload(empty)
        assert any("nvidia power limit" in err for err in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/tests/tui/test_config_modal.py -v`
Expected: FAIL — `AttributeError: 'ServerDefaultsConfig' object has no attribute 'nvidia_power_limit_watts'` during `compose()` / `_collect_values` (`#cfg-default_nvidia_power_limit_watts` Input not found), and `default_nvidia_power_limit_watts` not a field of `ConfigPayload`.

- [ ] **Step 3: Write minimal implementation**

In `src/llama_cli/tui/components/config_modal.py`:

1. Add the field to `ConfigPayload` (with the other `default_*` fields):

```python
    default_split_mode: str = "layer"
    default_nvidia_power_limit_watts: str = "290"
```

2. Add to `to_config_updates()` (after the `split_mode` line):

```python
            "server_defaults.nvidia_power_limit_watts": self.default_nvidia_power_limit_watts,
```

3. In `_validate_config_payload`, add to the `numeric_fields` tuple:

```python
        ("nvidia power limit", payload.default_nvidia_power_limit_watts, True, True),
```

4. In `_collect_values`, after the `default_split_mode` line:

```python
            default_nvidia_power_limit_watts=self.query_one(
                "#cfg-default_nvidia_power_limit_watts", Input
            ).value.strip(),
```

In `src/llama_cli/tui/components/form_widgets.py` `build_config_profile_defaults_collapsible`, add a `field_row` after the "Default GPU layers" field (after line ~433):

```python
        field_row(
            "NVIDIA power limit (W)",
            "default_nvidia_power_limit_watts",
            str(defaults.nvidia_power_limit_watts),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
```

- [ ] **Step 4: Handle the empty-value case in `_validate_config_payload`**

The existing `_validate_optional_number_field` treats an empty string as valid (returns `None`). The power-limit field is required (never empty — defaults to `"290"`). Make the `numeric_fields` loop treat it as required by special-casing it before the loop, or extend the loop. Add this block right after the `numeric_fields` tuple is built:

```python
    if not payload.default_nvidia_power_limit_watts.strip():
        errors.append("Invalid nvidia power limit: must be a number (0 = disabled)")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/tests/tui/test_config_modal.py -v`
Expected: PASS (all existing + new tests; `test_all_fields_collected` now also reads the new Input via `_collect_values`)

- [ ] **Step 6: Run the TUI suite**

Run: `uv run pytest src/tests/tui -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/llama_cli/tui/components/config_modal.py src/llama_cli/tui/components/form_widgets.py src/tests/tui/test_config_modal.py
git commit -m "feat: config modal field for nvidia power limit default"
```

---

### Task 6: Full gate

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `uv run ruff check .`
Expected: PASS

- [ ] **Step 2: Format check**

Run: `uv run ruff format --check .`
Expected: PASS

- [ ] **Step 3: Type check**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Full test suite**

Run: `uv run pytest`
Expected: PASS

- [ ] **Step 5: Pre-commit gate**

Run: `uv run pre-commit run --all-files`
Expected: PASS

- [ ] **Step 6: Manual smoke (optional, requires hardware)**

If the machine has an NVIDIA GPU and passwordless sudo, launch a dry run to see the cap:

Run: `uv run llm-runner dry-run qwen35`
Expected: output includes `Power limit: 290 W`

And a real launch applies the cap before the server boots:

Run: `sudo nvidia-smi -q -d POWER | rg "Power Limit"` before and after `uv run llm-runner qwen35`.