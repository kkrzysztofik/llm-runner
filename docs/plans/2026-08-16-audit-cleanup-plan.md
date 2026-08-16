# Audit Cleanup Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the 8-batch over-engineering cleanup approved in
`docs/plans/2026-08-16-audit-cleanup-design.md`, cutting ~3,400 source lines
from `int` with every batch a gate-green commit + push.

**Architecture:** Deletions first (batches 1–4), config-cluster dedup (5),
TUI dead code (6), telemetry/benchmark micro-cuts (7), behavior changes +
AGENTS.md (8). Each batch is atomic: verify → delete/rewrite → prune tests →
gate → commit → push. No half-applied deletion ever lands on `int`.

**Tech Stack:** Python 3.14, textual, rich, loguru, pytest, ruff, pyright, uv.

## Global Constraints

- Work directly on branch `int`. Atomicity: if maintainer WIP lands mid-task,
  finish-or-abort the task first — never push a half-deleted tree.
- No script-based code changes (AGENTS.md). Subagents edit by hand; the lead
  reviews every subagent diff before the gate.
- Every code deletion removes its tests in the same commit. No test may
  reference a deleted symbol (ruff/pyright/pytest all fail otherwise).
- Gate before EVERY commit (AGENTS.md agent guardrail):
  ```bash
  uv run pre-commit run --all-files
  uv run pytest
  ```
  Only if both green: commit `chore(cleanup): batch-N — <theme>` and `git push`.
- No br commands at any point in this campaign (user instruction). If
  `.beads/` appears in `git status`, flag and ask — do not touch it.
- No new dependencies. No new source files. File deletions for this
  campaign are EXACTLY this list (all covered by spec approval) — nothing
  else may be deleted:
  1. `src/llama_manager/dashboard_controller.py` (Task 5)
  2. `src/llama_manager/dashboard_view_model.py` (Task 5)
  3. `src/tests/config/test_dashboard_controller_save_profile.py` (Task 5)
  4. `src/tests/config/test_dashboard_view_model.py` (Task 5)
  5. `src/llama_manager/reports/redaction.py` (Task 4)
  6. `src/llama_cli/tui/components/system_status.py` (Task 6)
  7. `src/llama_cli/tui/components/gpu_telemetry.py` (Task 6)
  8. `src/llama_manager/gpu_telemetry/level_zero.py` (Task 7)
- Line numbers cited from the audit are HEAD-relative starting points —
  locate by symbol name, not line number.
- Do not touch: `docs/ARCHITECTURE.md`, `docs/PRD.md`, `specs/`, GPU driver /
  toolchain environment logic, lockfile atomicity, cancel/process-group
  handling beyond the listed items.
- `except A, B:` multi-exception syntax is valid on this project's Python —
  never "fix" it in batches that touch it.

---

## Task 1: Orchestration / slot / probe dead code

**Files:**
- Modify: `src/llama_manager/orchestration/manager.py`, `audit.py`,
  `launcher.py`, `slot_state.py`, `smoke.py`, `risk_ack.py`,
  `slot_lockfile.py`, `types.py`, `src/llama_manager/slot_state.py`,
  `src/llama_manager/smoke.py`
- Test: `src/tests/test_launcher.py`, `src/tests/test_probe_config_models.py`, `src/tests/test_slot_lockfile.py`, `src/tests/orchestration/*`, `src/tests/test_launch_flow.py` (whichever exist — `rtk ls src/tests` first)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ServerManager` with only `stream_pipe`, `wait_for_processes`,
  `close_log_pipe`, `launch_slot` surface; `risk_ack.issue_risk_prompt_if_needed`
  (tokenless ack); `SlotRuntime`/`ArtifactMetadata`/`ProcessMetadata` gone.

- [ ] **Step 1: Verify deadness** (each grep must return zero production
  callers outside the listed owners before deleting that symbol):

```bash
cd /home/kmk/llm-runner
for s in on_interrupt on_terminate run_server_foreground acquire_lock check_lock_stale \
         ConsecutiveFailureCounter SlotRuntime ArtifactMetadata ProcessMetadata \
         is_success launch_count issue_ack_token validate_ack_token \
         _build_launch_status_messages risk_result log_path; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
```

- [ ] **Step 2: Delete ServerManager test-only API** from
  `orchestration/manager.py` + `orchestration/__init__.py` re-exports:
  `on_interrupt`, `on_terminate`, `_stream_pipe`, `_wait_for_processes`,
  `_format_output`, `run_server_foreground`, `acquire_lock`,
  `check_lock_stale`; and `slot_lockfile.check_lock_stale`.

- [ ] **Step 3: Delete the file-based audit log** from `orchestration/audit.py`:
  file open/rotate/append/fchmod machinery and the `log_path` parameter
  threaded through `audit.py` → `manager.py`. Keep the in-memory
  `record_event` list + `get_events`.

- [ ] **Step 4: Delete probe dead surface** from `orchestration/slot_state.py`
  and `orchestration/smoke.py`: `ConsecutiveFailureCounter`,
  `SlotRuntime`, `ArtifactMetadata`, `ProcessMetadata`, `LaunchResult`
  dead fields (`is_success`, `launch_count`), the 5 duplicated permission
  constants in `types.py` (artifact.py keeps its own copies),
  `_build_launch_status_messages`, the `SmokeTarget.backend` field (keep the
  other SmokeTarget fields), the `SmokeTarget.aliases` list→str rename is NOT
  in scope.

- [ ] **Step 5: Cut the ack-token ceremony.** In `risk_ack.py` delete
  `issue_ack_token` and `validate_ack_token`; in `orchestration/manager.py`
  (or wherever `evaluate_risks`/`acknowledge_risk` live — grep
  `issue_ack_token` for the call chain), drop the `token` param/return and
  the `f"ack:{attempt_id}"` generation. Keep attempt_id-scoped ack state and
  the `RISK_ACK` constants. Remove the constant-None `risk_result` parameter
  from `_evaluate_and_handle_risks`.

- [ ] **Step 6: Fix remaining cross-file bits.**
  - `slot_lockfile.py`: replace `from .launch import _lockfile_error` with
    `from .lockfile import _lockfile_error`.
  - `resolve_slot_runtime_status`: delete the dead
    `getattr`/`pid_exists` fallback branch (keep the live path).
  - Confirm `risk_ack.py` is the single owner of the
    `RISK_ACK_LABEL = "warning_bypass"` literal. The dead copy in
    `tui/constants.py` is deleted in Task 6 Step 9 (design assigns it to
    batch 6) — do NOT delete it here.

- [ ] **Step 7: Rewrite `_SubprocessHandle`** in
  `orchestration/launcher.py` (replaces the class at ~line 93 and the wrap at
  ~line 80):

```python
class _ServerProc(subprocess.Popen[str]):
    """Popen whose timed ``wait`` raises :class:`ProcessTimeoutError`."""

    def wait(self, timeout: float | None = None) -> int:
        try:
            return super().wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ProcessTimeoutError(
                f"process {self.pid} did not exit within {timeout}s",
            ) from None
```

  In the spawn site, replace `_SubprocessHandle(subprocess.Popen(...))` with
  `_ServerProc(...)` keeping the existing `cmd` args, `stdout/stderr=PIPE,
  text=True, bufsize=1`, the `# noqa: S603` and the safe-argv comment. Remove
  `type: ignore` lines that vanish with the wrapper. `ProcessHandle` protocol
  conformance: `pid`/`stdout`/`stderr`/`poll` come from `Popen` itself.

- [ ] **Step 8: Shrink `resolve_smoke_targets`** (smoke.py): after resolving
  the profile-id list, build targets with one comprehension instead of the
  both/slot special branches (behavior identical — both branches already
  produce the same `(profile_id, aliases)` list, aliases singular).

- [ ] **Step 9: Prune tests.** In the test files named under Files: delete
  tests for every symbol removed in Steps 2–6 (`run_server_foreground`,
  lock acquire/stale, `_ConsecutiveFailureCounter` sections of
  `test_probe_config_models.py`, `SlotRuntime`/`ArtifactMetadata`
  constructor tests, ack-token tests, `_build_launch_status_messages`).
  Tests that build `ServerManager` purely to exercise removed methods go
  with them. Keep tests for `stream_pipe`, `wait_for_processes`,
  `launch_slot`, lockfile `acquire`/`release` (production API).

- [ ] **Step 10: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 1 — orchestration/slot/probe dead code" && git push
```

---

## Task 2: Build pipeline + toolchain

**Files:**
- Modify: `src/llama_manager/toolchain/detector.py`, `toolchain/__init__.py`,
  `build_pipeline/models.py`, `build_pipeline/pipeline.py`,
  `build_pipeline/orchestration.py`, `build_pipeline/utils.py`,
  `build_pipeline/clone.py`, `build_pipeline/status.py`,
  `build_pipeline/_context.py` (only if it imports deleted helpers),
  `src/llama_cli/commands/build.py`, `src/llama_manager/setup_venv.py`
- Test: `src/tests/build/test_toolchain.py`, `test_pipeline_orchestration.py`,
  `test_pipeline_clone_sources.py`, `test_build_cli.py`,
  `test_build_config.py`, `test_setup_toolchain.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `BuildPipeline.run_build_for_backend` (only entry),
  `tools_present` (no `ToolchainHint`), `VenvResult` without accessor methods,
  `BuildBackend` = SYCL|CUDA only.

- [ ] **Step 1: Verify deadness**

```bash
for s in parse_version version_at_least CMAKE_MINIMUM_VERSION ToolchainHint \
         get_python_path get_pip_path format_hint is_url_available required_for \
         run_both_backends BuildBackend.BOTH get_lock_error_message is_success \
         binary_size_mb CMAKE_C_COMPILER_SYCL MSG_SOURCES_NOT_GIT_REPO \
         source_existed_before_clone clone_timeout \
         _get_detect_tool _get_oneapi_bin; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
```

- [ ] **Step 2: Toolchain.** Delete `parse_version`, `version_at_least`,
  `CMAKE_MINIMUM_VERSION` from `toolchain/detector.py` (keep
  `TOOL_VERSION_MISMATCH` handling if it only *emits* — grep first). Delete
  the 7 `*_HINT` re-exports + `SYCL/CUDA_REQUIRED_TOOLS` from
  `toolchain/__init__.py` (sole consumer already imports from
  `.constants`). Delete `_get_detect_tool()` / `_get_oneapi_bin()` indirection
  in `detector.py` — call `detect_tool` / `_ONEAPI_BIN` directly. Delete
  `_COMMON_MISSING_TOOLS` comprehension in the detector (reconstruct the set
  inline where tests previously patched it — see Step 7 for the test-side
  move).

- [ ] **Step 3: Venv + models.** In `setup_venv.py`: delete
  `VenvResult.is_valid` / `get_python_path` / `get_pip_path`; delete the
  Windows `Scripts` fallback branches (Linux-only project). In
  `build_pipeline/models.py`: delete `ToolchainHint.format_hint` /
  `is_url_available` / `required_for` + the initializer list args; delete
  `BuildArtifact.is_success` / `binary_size_mb`; delete
  `BuildConfig.CMAKE_C_COMPILER_SYCL` / `CMAKE_CXX_COMPILER_SYCL` ClassVars
  (configure() hardcodes `icx`/`icpx` already); delete `MSG_SOURCES_NOT_GIT_REPO`;
  delete `BuildBackend.BOTH`.

- [ ] **Step 4: Pipeline.** Delete `BuildPipeline.run_both_backends` (callers
  already loop `run_build_for_backend`); delete `get_lock_error_message` and
  its callers' indirection (the lock-contention path returns the static
  message directly — keep the message string).

- [ ] **Step 5: clone.py + status.py.** Delete the
  `source_existed_before_clone` parameter chain (always False end-to-end);
  replace `getattr(config, "clone_timeout", 120)` with a module constant
  `CLONE_TIMEOUT_S = 120` in clone.py (verify no real `config.clone_timeout`
  exists first); replace the `build_shallow_clone` `getattr(config,
  "shallow_clone", ...)` with the actual `config.shallow_clone` (and the same
  getattr in `tui/components/build.py:629`); delete the unreachable
  `if not parts` guard in `status.py`.

- [ ] **Step 6: Rewrites (exact code).**

  a) `build_pipeline/orchestration.py` — replace `_merge_config_overrides`
  (lines ~17–63) with:

```python
import dataclasses

_DERIVED_FIELDS: frozenset[str] = frozenset(
    ("backend", "source_dir", "build_dir", "output_dir")
)


def _merge_config_overrides(base: BuildConfig, overrides: BuildConfig) -> BuildConfig:
    """Merge non-None, non-empty-string fields from *overrides* onto *base*.

    Derived fields (``_DERIVED_FIELDS``) are never overwritten — they are
    always taken from *base* so flavor-resolved URLs survive empty-string
    overrides.
    """
    merged = dataclasses.asdict(base)
    for field_name, value in dataclasses.asdict(overrides).items():
        if field_name in _DERIVED_FIELDS:
            continue
        if value is not None and value != "":
            merged[field_name] = value
    return BuildConfig(**merged)
```

  b) `build_pipeline/utils.py` — replace `_send_termination_signal` +
  `_send_kill_signal` (lines ~120–143) with:

```python
def _signal_proc(
    proc: subprocess.Popen[str],
    process_group_id: int | None,
    sig: int,
    fallback: Callable[[], None],
) -> None:
    try:
        if process_group_id is not None:
            os.killpg(process_group_id, sig)
        else:
            fallback()
    except ProcessLookupError:
        pass
    except OSError:
        fallback()


def _send_termination_signal(proc: subprocess.Popen[str], process_group_id: int | None) -> None:
    _signal_proc(proc, process_group_id, signal.SIGTERM, proc.terminate)


def _send_kill_signal(proc: subprocess.Popen[str], process_group_id: int | None) -> None:
    _signal_proc(proc, process_group_id, signal.SIGKILL, proc.kill)
```

  (Add `Callable` to the `collections.abc` import if missing.)

  c) `src/llama_cli/commands/build.py`:
  - delete `_format_duration` — `from llama_manager.build_pipeline.utils
    import _format_duration` (private but same-repo import is the established
    pattern here; if the module guard complains, move `_format_duration`
    public to `build_pipeline/utils.py` as `format_duration` and update both
    importers).
  - replace the manual `_format_success_json` body with
    `BuildArtifact(**...).to_dict()` serialization of the artifact the
    function already builds (verify `to_dict` exists on `BuildArtifact` —
    `rtk grep -n "def to_dict" src/llama_manager/build_pipeline/models.py`).
  - `_get_backends`: collapse the empty-arg default branch (it re-lists the
    same backends in two places).
  - `_default_build_dir`: single expression.

- [ ] **Step 7: Prune + move tests.**
  - `test_toolchain.py`: delete all version-parsing suites; move the
    `_get_detect_tool` patch targets to
    `monkeypatch.setattr(detector, "detect_tool", ...)` and
    `monkeypatch.setattr(detector, "_ONEAPI_BIN", ...)` in the remaining
    detector tests.
  - `test_pipeline_orchestration.py`: delete `run_both_backends` tests;
    `test_pipeline_clone_sources.py`: delete tests exercising
    `source_existed_before_clone` / the `clone_timeout` getattr (replace with
    the constant if they test the timeout behavior).
  - `test_build_cli.py` / `test_build_config.py`: delete tests for
    `CMAKE_*_COMPILER_SYCL`, `is_success`, `binary_size_mb`,
    `MSG_SOURCES_NOT_GIT_REPO`.
  - `test_setup_toolchain.py`: delete `is_valid`/`get_*_path` tests and the
    Windows-branch tests.

- [ ] **Step 8: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 2 — build pipeline + toolchain" && git push
```

---

## Task 3: Validation clusters

**Files:**
- Modify: `src/llama_manager/validation/commands/builder.py`,
  `src/llama_manager/validation/validators.py`,
  `src/llama_manager/config/errors.py`
- Test: `src/tests/system/test_foundation_contracts.py`,
  `src/tests/system/test_toolchain.py` (version sections),
  `src/tests/system/test_server.py` (validator tests),
  `src/tests/test_validation*` (whatever exists)

**Interfaces:**
- Consumes: nothing.
- Produces: `validation/validators.py` keeps ONLY `require_model`,
  `validate_port`, `validate_ports`, `validate_server_config` (+ their
  helpers). `errors.py` keeps the dataclasses + `MultiValidationError`
  without `sort_errors`/`error_count`.

- [ ] **Step 1: Verify deadness**

```bash
for s in DoctorCheckResult DoctorReport sort_validation_errors
         compute_machine_fingerprint _get_lspci_output _sycl_device_details
         _sycl_dotted_device_details check_hardware_allowlist assess_vram_risk
         VRamRecommendation validate_slots _validate_slot
         _validate_duplicate_slots _convert_results_to_errors validate_threads
         validate_backend_eligibility sort_errors error_count; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
```

- [ ] **Step 2: Delete the doctor/fingerprint/VRAM cluster** from
  `validation/commands/builder.py` (~370 lines): `DoctorCheckResult`,
  `DoctorReport`, `sort_validation_errors` + its `sort_key`,
  `compute_machine_fingerprint` + `_get_cpu_model` + `_get_os_name`
  (these shell out to `cat` for `Path.read_text`-able data),
  `_get_lspci_output`, `_sycl_device_details`,
  `_sycl_dotted_device_details`, `check_hardware_allowlist`,
  `assess_vram_risk` + its `VRamRecommendation` usage. After the cut,
  confirm the `VRamRecommendation` enum (`config/enums.py`) has zero
  consumers repo-wide and delete the enum + its `config/__init__.py`
  re-export NOW — batch 3 owns it (GgufParseError and DoctorCheckStatus
  also live in enums.py but are deleted in Task 5).
  The CLI `doctor` command has its OWN equivalents in
  `llama_cli/commands/doctor.py` — never touch that file.

- [ ] **Step 3: Delete the dead validator chain** from
  `validation/validators.py`: `validate_slots`, `_validate_slot`,
  `_validate_duplicate_slots`, `_convert_results_to_errors`,
  `validate_threads`, `validate_backend_eligibility`. Keep
  `require_model`, `validate_port`, `validate_ports`,
  `validate_server_config` and their live helpers (`require_executable` and
  friends are used by `commands/profile.py` — grep before touching).

- [ ] **Step 4: Delete `errors.py` test-only API**:
  `MultiValidationError.sort_errors` and `.error_count` (plus any now-unused
  imports like the error-code sorting helper they pulled in).

- [ ] **Step 5: Prune tests.** `test_foundation_contracts.py`: delete the
  sections covering the symbols above (the file mixes live + dead coverage —
  keep everything that imports the survivors). `test_toolchain.py`: delete
  any remaining `assess_vram_risk`/fingerprint tests (version suites already
  handled in Task 2 if they were there). `test_server.py`: delete
  `validate_threads` / slot-validation tests.

- [ ] **Step 6: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 3 — validation clusters" && git push
```

---

## Task 4: Reports, logging, profile_orchestrator copy

**Files:**
- Modify/delete: `src/llama_manager/reports/` (shrink, not delete module),
  `src/llama_manager/reports/redaction.py` (DELETE — folded into
  `common/security.py`), `src/llama_manager/logging_setup.py`,
  `src/llama_manager/profile_orchestrator.py`,
  `src/llama_manager/common/security.py`,
  `src/llama_cli/server_runner.py` (configure_logging_split caller)
- Test: `src/tests/system/test_reports.py`, `test_logging_setup.py`,
  `src/tests/cli/test_profile_cli.py`, `test_security*` (redaction tests)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `reports.write_failure_report` (kept); single
  `configure_logging(stderr_level=..., file_level=..., log_file=...,
  json_logs=...)`; `profile_orchestrator` keeps ONLY
  `resolve_profile_slot`, `resolve_benchmark_config`,
  `resolve_benchmark_binary`, `get_driver_version` + their private helpers;
  `redact_sensitive` gone (all importers switch to
  `llama_manager.common.security.redact_log_line`).

- [ ] **Step 1: Verify deadness**

```bash
for s in MutatingActionLogEntry log_mutating_action _rotate_mutating_log
         rotate_reports redact_sensitive run_profile create_profile_record
         _default_subprocess_runner detect_backend _stream_to_text
         DriverVersionProvider; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
rtk grep -rn "from .reports" src/llama_manager/reports/ --type py
rtk grep -rn "from .redaction\|from ..reports.redaction\|reports import" src/ --include="*.py" 2>/dev/null | grep -v __pycache__ | grep -v tests
```

- [ ] **Step 2: Reports.** Delete the mutating-action cluster from
  `reports/` (find it — likely `reports/mutating.py` or in
  `reports/__init__.py`: `MutatingActionLogEntry`, `log_mutating_action`,
  `_rotate_mutating_log`, `rotate_reports` + their log-file paths/rotation
  helpers). Keep `write_failure_report` and its helpers (live from
  `build_pipeline/_context.py:120`).

- [ ] **Step 3: Fold redaction — VERBATIM move, do NOT re-target at
  `redact_log_line`.** DEVIATION NOTE (from the design doc wording "fold
  its URL patterns into redact_log_line"): verification showed
  `reports/redaction.py` contains NO URL pattern, and `redact_sensitive`
  and `redact_log_line` have DIFFERENT output formats — `redact_sensitive`
  renders `KEY: [REDACTED]` (handles `=` and `:`, quotes, and standalone
  sensitive words) while `redact_log_line` renders `KEY=[REDACTED]` and
  leaves standalone words alone. Re-targeting callers at `redact_log_line`
  would change build-output redaction. Instead:
  - Move the `redact_sensitive` function body verbatim into
    `common/security.py` (next to `redact_log_line`; it already imports
    nothing from reports — use the `re` + `REDACTED_VALUE` constants that
    file already has).
  - DELETE `reports/redaction.py` (file deletion #5 above, spec-approved).
  - Retarget the 3 production import sites to
    `from ..common.security import redact_sensitive`:
    `build_pipeline/utils.py:19`, `reports/failure.py:10` (failure.py holds
    the kept `write_failure_report` — it keeps redacting, from the new
    home), and `reports/__init__.py:9,13` (remove the re-export +
    `__all__` entry).
  - `LogBuffer` is unaffected (it already uses `redact_log_line`).
  - The `redact_sensitive` test class in `test_reports.py` moves to
    `src/tests/test_security.py` (or the existing security test module)
    with UNCHANGED assertions.

- [ ] **Step 4: Merge logging config.** In `logging_setup.py`:
  - Delete `_JsonLogEnvelope`, `_format_json`, `_json_default` (json path
    uses loguru `serialize=True` already).
  - Promote the two duplicated filter closures to module level (identical in
    both functions):

```python
def _redact_only_filter(record: LoguruRecord) -> bool:
    record["message"] = _redact_log_message(record["message"])
    return True


def _stderr_sink_filter(record: LoguruRecord) -> bool:
    rec_name = record["name"] or ""
    if _SUPPRESS_BUILD_PIPELINE_ON_STDERR.get() and rec_name.startswith(
        _BUILD_PIPELINE_LOG_PREFIX
    ):
        return False
    record["message"] = _redact_log_message(record["message"])
    return True
```

  - Delete `configure_logging_split` and collapse `configure_logging` to:

```python
def configure_logging(
    *,
    stderr_level: str | None = "INFO",
    file_level: str | None = None,
    log_file: str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure the logging subsystem.

    *stderr_level* ``None`` disables the stderr sink. *file_level* ``None``
    follows *stderr_level* (or INFO when stderr is disabled).
    """
    logger.remove()

    stderr_norm: str | None = _validate_log_level(stderr_level)
    if log_file is not None:
        if file_level is None:
            file_norm = stderr_norm if stderr_norm is not None else "INFO"
        else:
            file_norm = _validate_log_level(file_level)
    else:
        file_norm = "INFO"

    text_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}"
    fmt = text_format if not json_logs else ""

    if stderr_norm is not None:
        logger.add(
            sys.stderr,
            level=stderr_norm,
            format=fmt,
            colorize=True,
            filter=_stderr_sink_filter,
            serialize=json_logs,
        )
    if log_file is not None:
        logger.add(
            log_file,
            level=file_norm,
            format=fmt,
            colorize=False,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            filter=_redact_only_filter,
            serialize=json_logs,
        )

    # --- Install stdlib → Loguru bridge (unchanged tail) ---
    root_logger = logging.getLogger()
    if not any(isinstance(h, _InterceptHandler) for h in root_logger.handlers):
        root_logger.addHandler(_InterceptHandler())
    for target_name in ("llama_manager", "llama_cli"):
        logging.getLogger(target_name).setLevel(logging.DEBUG)
        logging.getLogger(target_name).handlers = []
```

    with `_validate_log_level(level)` =
    `normalized = level.upper(); raise ValueError(f"unknown log level
    '{level}' — must be one of {list(_LEVEL_MAP)}") if normalized not in
    _LEVEL_MAP else normalized`.
  - Retarget callers: `server_runner.py:418`
    `configure_logging_split(stderr_level=..., file_level=..., log_file=...)`
    → `configure_logging(...)` (same kwargs). `commands/build.py:461`
    `configure_logging()` — unchanged.

- [ ] **Step 5: Profile orchestrator copy.** Delete from
  `profile_orchestrator.py`: `run_profile`, `create_profile_record`,
  `_default_subprocess_runner`, `detect_backend`, `_stream_to_text`,
  `DriverVersionProvider`, and the benchmark-timeout module constants
  (grep each — the CLI `cmd_profile` re-implements all of them with
  cancellation support). KEEP (imported by `commands/profile.py`):
  `resolve_profile_slot`, `resolve_benchmark_config`,
  `resolve_benchmark_binary`, `get_driver_version` and the private
  helpers those four need (`_BENCHMARK_*` config plumbing,
  `BenchmarkConfig` import). Also delete the pipeline-patch targets in
  `test_profile_cli.py` (tests that `monkeypatch.setattr(orchestrator,
  "run_profile"...)` — switch those tests to patch
  `llama_cli.commands.profile` internals instead, or delete if they only
  exercised the deleted copy).

- [ ] **Step 6: Prune tests.** `test_reports.py`: delete the
  mutating/rotate/redaction sections; keep the `write_failure_report` tests.
  `test_logging_setup.py`: rewire `configure_logging(level=X, ...)` calls to
  `configure_logging(stderr_level=X, ...)` (~25 call sites); rewire the four
  `configure_logging_split(...)` tests (lines ~452–492) to
  `configure_logging(...)`; delete tests for the JSON-envelope helpers.
  `test_security*`: redact_sensitive URL tests → `redact_log_line`.

- [ ] **Step 7: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 4 — reports/logging/orchestrator copy" && git push
```

---

## Task 5: Config cluster

**Files:**
- Modify: `src/llama_manager/dashboard_controller.py` (DELETE whole file —
  see Step 9), `src/llama_manager/dashboard_view_model.py` (DELETE whole
  file), `src/llama_manager/config/builder.py`, `config/persistence.py`,
  `config/defaults.py`, `config/enums.py` (GgufParseError only),
  `config/spec_decode.py`, `config/profile_cache.py` (not — flavor is
  Task 8), `src/llama_manager/common/profile_io.py`, `common/file_ops.py`,
  `common/validators.py`, `common/security.py` (safe_log),
  `common/errors.py` (error_message), `config/launch_runtime.py`,
  `src/llama_manager/metadata/` (leftovers), `src/llama_manager/slot_profile_store.py`
- Delete (WHOLE, spec-approved): `src/tests/config/test_dashboard_controller_save_profile.py`,
  `src/tests/config/test_dashboard_view_model.py`
- Test: `src/tests/config/test_config_persistence.py`,
  `test_config_builders.py`, `test_profile_cache.py`, `test_slot_profile_store.py`,
  `test_spec_decode.py`, `test_metadata.py`, `test_model_index.py`,
  `test_launch_runtime.py` (if exists)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `persistence.save_config_to_file`/`load_config_overrides_from_file`/
  `apply_config_updates` with type-derived field sets; generic
  `write_profile_toml`; plain-dataclass `SpeculativeDecodingConfig`;
  `config/__init__.py` without the three `create_*_cfg` factories.

- [ ] **Step 1: Verify deadness**

```bash
for s in create_summary_balanced_cfg create_summary_fast_cfg create_qwen35_cfg
         venv_path tui_launch_timeout_s probe_latency_threshold_s
         extract_gguf_metadata tokenizer_type GgufParseError is_valid_port
         profile_dir_path safe_log LaunchRuntimeOverrides launch_runtime_as_dict
         error_message; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
```

- [ ] **Step 2: Delete the legacy factory trio** from
  `config/builder.py` (`create_summary_balanced_cfg`,
  `create_summary_fast_cfg`, `create_qwen35_cfg`) and their re-exports in
  `config/__init__.py` (+ `llama_manager/__init__.py` if it re-exports them —
  grep). Their test section in `test_config_builders.py` goes too.

- [ ] **Step 3: Persistence rewrite.** In `config/persistence.py`:

  a) Replace the per-field `_PERSISTED_SECTIONS` value tuples with a
  section→class map and type-derived field sets:

```python
from dataclasses import fields
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints, NoneType

_SECTION_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "deployment": DeploymentConfig,
    "build": BuildPipelineConfig,
    "smoke": SmokeConfig,
    "server_defaults": ServerDefaultsConfig,
}


def _section_field_names() -> dict[str, set[str]]:
    return {
        section: {f.name for f in fields(cls)}
        for section, cls in _SECTION_CLASSES.items()
    }


def _field_type(cls: type, name: str):
    return get_type_hints(cls)[name]


def _is_nullable(t) -> bool:
    origin = get_origin(t)
    if origin is Union or origin is UnionType:
        return NoneType in get_args(t)
    return False


def _coercion_kind(t) -> str | None:
    if _is_nullable(t):
        return _coercion_kind(next(a for a in get_args(t) if a is not NoneType))
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is bool:
        return "bool"
    return None
```

  b) Build the qualified-name sets once (replaces `_INT_FIELDS`,
  `_FLOAT_FIELDS`, `_BOOL_FIELDS`, `_NULLABLE_OPTIONAL_FIELDS` — delete
  those four frozensets, ~75 lines):

```python
_COERCION_FIELDS: dict[str, set[str]] = {"int": set(), "float": set(), "bool": set()}
_NULLABLE_FIELDS: set[str] = set()
for _section, _cls in _SECTION_CLASSES.items():
    for _f in fields(_cls):
        _t = _field_type(_cls, _f.name)
        if _is_nullable(_t):
            _NULLABLE_FIELDS.add(f"{_section}.{_f.name}")
        _kind = _coercion_kind(_t)
        if _kind:
            _COERCION_FIELDS[_kind].add(f"{_section}.{_f.name}")
```

  c) `save_config_to_file`: replace the `_PERSISTED_SECTIONS.items()` loop
  with an asdict walk (keep top-level fields + `_LEGACY_SERVER_DEFAULTS_KEYS`
  logic untouched):

```python
def save_config_to_file(config: Config, path: Path) -> None:
    """Write the modal-exposed config sections to *path* as TOML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"{field} = {_toml_value(getattr(config, field))}" for field in _TOP_LEVEL_FIELDS]
    for section, cls in _SECTION_CLASSES.items():
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in dataclasses.asdict(getattr(config, section)).items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

  d) `load_config_overrides_from_file`: replace the per-section field
  enumeration with the derived names (unknown keys still ignored, legacy
  `mmap`/`mlock` handling stays):

```python
    field_names = _section_field_names()
    for section, cls in _SECTION_CLASSES.items():
        section_data = raw.get(section)
        if not isinstance(section_data, dict):
            continue
        values = {k: v for k, v in section_data.items() if k in field_names[section]}
        if section == "server_defaults":
            for legacy_key in _LEGACY_SERVER_DEFAULTS_KEYS:
                if legacy_key in section_data:
                    values[legacy_key] = section_data[legacy_key]
        if values:
            overrides[section] = values
```

  e) `apply_config_updates` / `_coerce_config_field_value`: reference
  `_COERCION_FIELDS["int"]` / `["float"]` / `["bool"]` and `_NULLABLE_FIELDS`
  instead of the deleted frozensets. `_UPDATE_FIELDS` derives from
  `_section_field_names()` + `_TOP_LEVEL_FIELDS`.
  NOTE the behavior guard: a field annotated `Union[int, str]` (e.g.
  `n_gpu_layers_profile`) must NOT be int-coerced — `_coercion_kind`
  falls through to `None` for such unions, matching today's
  `return raw_value, None`.

- [ ] **Step 4: Generic profile TOML writer.** In
  `common/profile_io.py`, replace `write_profile_toml` body +
  `_profile_lines`/`_append_optional_int`/`_append_optional_float` (~120
  lines) with:

```python
def write_profile_toml(path: Path, data: dict[str, Any]) -> None:
    """Write the shared profile TOML shape used by slot/run profile stores."""
    lines: list[str] = []
    hidden = sorted(data.get("hidden_builtin_profiles", []))
    if hidden:
        lines.append(f"hidden_builtin_profiles = {json.dumps(list(hidden))}")
        lines.append("")
    for index, profile in enumerate(data.get("profiles", [])):
        if index > 0:
            lines.append("")
        lines.append("[[profiles]]")
        for key, value in profile.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_scalar(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        # chat_template_kwargs and friends are stored as JSON-encoded strings
        return json.dumps(json.dumps(value))
    if isinstance(value, list):
        return json.dumps(value)
    raise TypeError(f"unsupported profile value type: {type(value).__name__}")
```

  The reader (`slot_profile_store._profile_from_dict`) fills defaults for
  missing keys — verified by `test_profile_from_dict_applies_defaults` — so
  writing only present keys is safe. Also delete `profile_dir_path`
  (zero callers) from this file. If a test asserts exact TOML text for a
  minimal dict (missing keys written as defaults), switch it to
  parse-and-compare.

- [ ] **Step 5: Spec decode cleanup.** In `config/spec_decode.py`: delete
  `SpeculativeDecodingFieldsMixin` (12 flat pass-through properties —
  production always reads `cfg.spec_decode.<field>`) and drop the
  `dict` base from `SpeculativeDecodingConfig` (dict-style access
  `cfg.spec_decode["key"]` is tests-only — update those tests to attribute
  access). In `config/builder.py`: delete `_SPEC_DECODE_FIELDS` — import
  `SPECULATIVE_DECODING_FIELD_NAMES` from `config.spec_decode` instead.
  `config/enums.py`: delete `GgufParseError`.

- [ ] **Step 6: Dead Config surface + metadata.** `config/defaults.py`:
  delete `PathsConfig.venv_path`, `ServerDefaultsConfig.spec_decode`
  property+setter, `model_qwen35_both`, `summary_balanced_port`?? — NO,
  ports stay (persistence writes them); delete only what Step 1 proved dead:
  `venv_path`, the `spec_decode` property/setter pair,   `ctx_size_both_*`-style
  fields, and `tui_launch_timeout_s` / `probe_latency_threshold_s` (the
  Step-3 derived field sets handle persistence coverage automatically —
  no manual list edits needed).
  `metadata/`: delete `_GGUF_V2/V3/V4_MAGIC`, `_GENERAL_NAME_PATTERN`,
  `tokenizer_type` + `_detect_tokenizer_type_from_reader`, the
  `model_name` parameter of `extract_gguf_metadata`, and the unused record
  fields (verify via Step 1 greps — `tokenizer_type` etc. must show zero
  production callers).

- [ ] **Step 7: Small config/common cuts.**
  - `builder.py::_profile_to_config_data`: replace the ~42 explicit
    assignments with `dataclasses.asdict` + flatten the
    `server_defaults`/`spec_decode` subdicts into the ServerConfig kwargs
    (keep the explicit overrides for fields whose names differ between
    layers, e.g. `n_gpu_layers_profile` → `n_gpu_layers`).
  - `model_index.py`: drop the `config` parameter from
    `model_index_path(config)` (callers pass nothing useful — update them);
    collapse the double isolation (per-file `multiprocessing.Process` around
    a thread-with-timeout) to ONE timeout layer with a
    `# ponytail: single OS-thread timeout; per-file processes if extraction
    hangs the whole scan` comment.
  - `common/validators.py`: delete `is_valid_port`.
  - `common/profile_io.py`: (done in Step 4 — `profile_dir_path`).
  - `common/security.py`: delete `safe_log`.
  - `config/launch_runtime.py`: delete `LaunchRuntimeOverrides` +
    `launch_runtime_as_dict` if Step 1 shows zero callers (the
    `split_mode` field in `LaunchRuntime` dataclass + TypedDict STAYS —
    it's live from commit `f70e54e`).
  - `common/errors.py`: delete `ErrorDetail.error_message`.
  - `common/file_ops.py`: merge `atomic_write`/`atomic_write_json` into one
    helper with an optional serializer callback; update the two JSON
    callers.

- [ ] **Step 8: Gate + commit + push** (dashboard deletion is Step 9 —
  include it in this same commit):

```bash
uv run pre-commit run --all-files && uv run pytest
```

- [ ] **Step 9: Delete the dashboard pair.** Verify zero non-test importers:

```bash
rtk grep -rn "dashboard_controller\|dashboard_view_model" src/llama_cli src/llama_manager --include="*.py" | grep -v __pycache__ | grep -v tests
```

Then DELETE `src/llama_manager/dashboard_controller.py` and
`src/llama_manager/dashboard_view_model.py` (spec-approved) and DELETE the
two whole test files named in the Files block. Then:

```bash
git add -A -- src/
git commit -m "chore(cleanup): batch 5 — config cluster" && git push
```

---

## Task 6: TUI dead code

**Files:**
- Modify: `src/llama_cli/tui/model.py`, `controller.py`, `viewmodel.py`,
  `textual_app.py`, `components/build.py`, `components/gpu_stats.py`,
  `components/system_health.py`,   `components/server_column_panel.py`,
  `components/server_log.py`, `components/__init__.py`, `constants.py`,
  all modals under `tui/modals/` (BINDINGS dedup step)
- Delete (WHOLE, spec-approved — see Global Constraints list):
  `src/llama_cli/tui/components/system_status.py` (Step 7),
  `src/llama_cli/tui/components/gpu_telemetry.py` (Step 6)
- Test: `src/tests/tui/test_tui.py`, `test_controller.py`,
  `test_viewmodel.py`, `test_build_component.py`, `test_textual_app.py`,
  `test_gpu_stats.py` (panel tests), `test_system_health*`

**Interfaces:**
- Consumes: Task 5's `SplitMode`/launch-runtime surface only if `build.py`
  reads it (it does not — the build wizard is independent). No cross-task
  dependency.
- Produces: `DashboardModel` without the dead build-request/slot-ops
  surface; `SystemHealthWidget` taking an `id` (SystemStatusWidget gone);
  modals sharing `MODAL_CANCEL_BINDINGS`.

- [ ] **Step 1: Verify deadness.** For each symbol in Steps 2–8 below, grep
  production callers (exclude `tests/`) and confirm zero or only the listed
  keepers:
  `cpu_percentages`, `memory_usage_rows`, `system_info_snapshot`,
  `collect_memory_usage_rows_now`, `collect_system_info_snapshot_now`,
  `set_cached_slot_stats`, `make_collector`, `build_is_retrying`,
  `build_retries_remaining`, `log_lines`, `build_request`,
  `request_build`, `_build_request`, `handle_slot_transition`,
  `get_stale_warning`, `is_model_index_refreshing`, `apply_add_slot_from_form`,
  `add_slot_from_form`, `remove_live_slot`, `build_llama_cpp`,
  `_signal_handler_build`, `_original_sigint_handler`,
  `action_create_profile`,
    `_refresh_add_slot_startup`, `_deferred_resolve`, `GPUTelemetryWidget`,
  `gpu_telemetry_lines`, `_format_gpu_stats_text`, `SystemHealthProvider`,
  `_EmptySystemHealthProvider`, `SystemStatusWidget`,
  `action_interrupt_dashboard`, `_STATUS_MESSAGE_LIFETIME_S`,
  `_status_messages`, `_status_lock`, `build_selected_backends`,
  `build_in_progress`, `build_result`, `build_error`,
  `build_selected_backends_options`, `build_stage`,
  `build_progress_percent`, `navigate_wizard_step`, `_clear_mounted`,
  `_collect_options`, `_read_build_form_fields`,
  `RISK_CONFIRM_PROMPT`, `STATUS_PREFIX`, `STYLE_BOLD_YELLOW`,
  `SystemHealthRenderer`.

- [ ] **Step 2: Model dead surface** (`tui/model.py`):
  delete `cpu_percentages`, `memory_usage_rows`, `system_info_snapshot`,
  `collect_memory_usage_rows_now`, `collect_system_info_snapshot_now`
  (dupes of `collect_system_health_snapshot`), `set_cached_slot_stats`,
  `stop`, `make_collector` (+ the controller pass-through — Step 3),
  `build_is_retrying`, `build_retries_remaining` properties,
  `ServerColumnState.log_lines` (always `()`),
  `types.BuildViewState` (zero references).
  Keep `build_stage` / `build_progress_percent` ONLY if Step 1 shows a live
  reader — the audit found build.py reads `build_progress` directly, so
  expect to delete both.

- [ ] **Step 3: Controller dead surface** (`tui/controller.py`):
  delete the legacy slot-ops trio
  (`apply_add_slot_from_form`, `add_slot_from_form`, `remove_live_slot` —
  superseded by the async compute→prepare/stage/complete path),
  `build_llama_cpp` + `_signal_handler_build` + `_original_sigint_handler`
  (production builds via `begin_build`/`run_build_loop`),
  `handle_slot_transition`, `get_stale_warning`,
  `is_model_index_refreshing`, `model_index_path`,
  `_STATUS_MESSAGE_LIFETIME_S` alias, `_status_messages` / `_status_lock`,
  and the one-line model pass-through props the app bypasses
  (`controller.model.X` — keep only the externally used few: verify each
  with a grep of `controller.X` inside `textual_app.py` + `components/`).

- [ ] **Step 4: Build-request console flow** (`model.py` + `controller.py`
  + `viewmodel.py` + `CommandMenuState`):
  delete `request_build`, the `_build_request` prop on model/viewmodel, the
  build branch in `cancel_pending_prompt` (keep the Escape/ctrl-c branch
  and the binding), the build branch in `check_action`,
  `viewmodel.can_select_build_target`'s build-request clause,
  `CommandMenuState.build_request`. Nothing sets the flag — BuildModalScreen
  replaced it.

- [ ] **Step 5: View-model build pass-throughs** (`tui/viewmodel.py`):
  delete `build_selected_backends`, `build_in_progress`, `build_result`,
  `build_error`, `build_selected_backends_options`, `build_stage`,
  `build_progress_percent` properties — build.py reads the model/wizard
  state directly (verify per property with the Step 1 grep before deleting;
  if any has a live reader, keep it and note why).

- [ ] **Step 6: GPU telemetry widget + system health chain.**
  - Delete `GPUTelemetryWidget` + `_flatten_gpu_lines` — i.e. DELETE the
    whole `components/gpu_telemetry.py` file (it contains nothing else;
    file deletion #7 above), plus `viewmodel.gpu_telemetry_lines`,
    `_format_gpu_stats_text`, and the `.gpu-telemetry*` CSS from
    `textual_app.py`'s CSS block; remove the `GPUTelemetryWidget` import +
    export from `components/__init__.py` (lines 6, 29). Test cleanup: the
    3 test methods at `test_tui.py:292–334` (widget compose/visibility)
    are deleted, and the assertion at `test_tui.py:2339`
    (`assert not list(app.query(GPUTelemetryWidget))`) goes with the class.
  - Delete the `SystemHealthRenderer` string-builder chain
    (`render_cpu_usage`, `render_memory_swap_usage`, `render_system_info`,
    `_format_core_grid_lines`, `_format_memory_row`, `_usage_bar`,
    `_usage_color`, `_format_uptime`, `_task_summary`, `_load_summary`,
    `_content_width`, `_memory_bar_width`, `_build_core_grid_rows`,
    `MIN/MAX_CONTENT_WIDTH`, `CPU_CORE_*` constants) from
    `components/system_health.py` — the widgets compose the snapshots
    directly. Keep the surviving `SystemHealthWidget` and
    `SystemStatusWidget`-fold (Step 7).

- [ ] **Step 7: SystemStatusWidget fold.** `SystemStatusWidget`
  (`components/system_status.py`, 17 lines) exists only to set
  `id="alerts"` + `classes="system-status"` around one
  `SystemHealthWidget` yield. Exact steps:
  - Extend `SystemHealthWidget.__init__` in `components/system_health.py`
    to forward `**kwargs` to `super().__init__` (standard Textual
    pattern; keep its existing `provider` parameter and its current
    default behavior).
  - In `textual_app.py:134` replace
    `yield SystemStatusWidget(self.view_model)` with
    `yield SystemHealthWidget(self.view_model, id="alerts",
    classes="system-status")`; fix the import (textual_app.py:39) — merge
    with the existing `system_health` import in that file.
  - DELETE `components/system_status.py` (file deletion #6 above); remove
    its import + export from `components/__init__.py` (lines 18, 36).
  - Keep the `.system-status` CSS rule in textual_app.py (the class now
    lives on `SystemHealthWidget` itself).

- [ ] **Step 8: build.py dead helpers** (`components/build.py`):
  delete the module-level `_read_build_form_fields` (verbatim copy of the
  production method), `_collect_options`, `navigate_wizard_step` (point its
  tests at `_handle_next` which inlines the step map), `_clear_mounted`,
  and the dead `_result_content` / `build_result_content` pair (point tests
  at `_build_result_panel`).

- [ ] **Step 9: App + constants + modals.**
  - Delete `action_interrupt_dashboard`, `action_create_profile`,
    `_refresh_add_slot_startup` → collapse to
    `viewmodel.mark_slot_launching` + the `_deferred_resolve` + deferred
    branch in `_resolve_slot_status`.
  - `tui/constants.py`: delete `RISK_ACK_LABEL` (ownership confirmed in
    Task 1 Step 6 — verify zero importers before deleting),
    `RISK_CONFIRM_PROMPT`, `STATUS_PREFIX`, `STYLE_BOLD_YELLOW`.
  - Five modals re-declaring escape/ctrl-c cancel `BINDINGS` → import and
    reuse `form_widgets.MODAL_CANCEL_BINDINGS` (2 modals already do — copy
    the pattern).

- [ ] **Step 10: Protocol + misc.**
  - `SystemHealthProvider` Protocol + `_EmptySystemHealthProvider` → type
    the widgets against the viewmodel with a `TYPE_CHECKING` import
    (existing pattern in `gpu_telemetry.py`).
  - `ServerColumnPanel.on_mount` docstring-only no-op → delete the method.
  - `server_log.py`: delete the `perf_counter` + dual `logger.debug`
    instrumentation around the once-per-mount compose.

- [ ] **Step 11: Shrinks folded in** (`model.py` + viewmodel):
  - Bounded status buffer → `collections.deque(maxlen=5)`:

```python
from collections import deque

self.status_messages: deque[tuple[float, str]] = deque(maxlen=5)

def push_status_message(self, message: str) -> None:
    with self.status_lock:
        self.status_messages.append((time.monotonic(), message))

def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
    cutoff = time.monotonic() - self.STATUS_MESSAGE_LIFETIME_S
    with self.status_lock:
        return [(ts, m) for ts, m in self.status_messages if ts > since_ts and ts >= cutoff]
```

  - `AsyncSlotPlan` (controller.py) reduced to the fields it actually
    carries: `(alias, profile_id, old_alias)` — drop
    `success`/`messages` (always `True`/`[]`). Update its instantiation at
    `controller.py:606` and any destructure sites.
  - Core-grid layout math: the renderer (deleted in Step 6) owns
    `MIN_CONTENT_WIDTH` / `MAX_CONTENT_WIDTH` / `CPU_CORE_BAR_WIDTH` /
    `CPU_CORE_CELL_WIDTH` — `SystemHealthWidget` needs that clamp, so
    move the constants onto `SystemHealthWidget` (or as module constants
    in `system_health.py`) and delete the duplicate width math in
    `viewmodel.cpu_usage_rows` / `_content_width` if present.
  - `_format_model_line` → call `_find_model_index_entry` (no re-implement
    the index lookup).
  - `_optional_numeric_display` → import from `form_widgets`.

- [ ] **Step 12: Prune tests.** In the named test files delete coverage for
  every symbol removed; `test_controller.py`/`test_viewmodel.py`/`test_tui.py`
  have the largest sections. Keep tests that drive the live build path
  (`begin_build`/`run_build_loop`), the async slot path, and the surviving
  widgets. Where a test constructed a widget with
  `_EmptySystemHealthProvider`, switch it to a viewmodel stub matching the
  TYPE_CHECKING type.

- [ ] **Step 13: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 6 — TUI dead code" && git push
```

---

## Task 7: gpu_telemetry + benchmark

**Files:**
- Modify: `src/llama_manager/gpu_telemetry/level_zero.py` (DELETE — shim),
  `gpu_telemetry/stats.py`, `gpu_telemetry/__init__.py`,
  `gpu_telemetry/level_zero_telemetry.py`, `src/llama_manager/gpu_stats.py`,
  `src/llama_manager/system_stats.py`, `src/llama_manager/benchmark/`,
  `src/llama_cli/tui/components/gpu_stats.py`, `components/system_health.py`
- Test: `src/tests/test_gpu_stats.py`, `test_gpu_telemetry_stats.py`,
  `test_benchmark.py`, `test_system_stats.py`, `test_foundation.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–6 (TUI meter helper lands after Task 6).
- Produces: `level_zero_telemetry.collect_level_zero_stats` as the direct
  import target; `usage_fill(percent, width)` shared helper; benchmark
  returns `CalledProcessResult`.

- [ ] **Step 1: Verify deadness**

```bash
for s in collect_nvtop_stats make_gpu_collector gpu_util memory_util
         format_stats_text _safe_read_text _unique_paths SubprocessResult
         _split_contiguous_blocks _get_task_stats _format_uptime; do
  echo "== $s =="; rtk grep -rn "$s" src/llama_cli src/llama_manager --type py | grep -v tests
done
```

- [ ] **Step 2: level_zero shim.** DELETE `gpu_telemetry/level_zero.py`
  (~72-line re-export shim — production uses only
  `collect_level_zero_stats`). Point importers (`stats.py`,
  `__init__.py`, 2 test imports) at `level_zero_telemetry`.

- [ ] **Step 3: Dead stats API.** Delete from `gpu_telemetry/stats.py` and
  `gpu_stats.py`: `collect_nvtop_stats` (legacy non-selector aggregate —
  production uses `collect_nvtop_stats_for_selector`), `make_gpu_collector`,
  `GPUStats.gpu_util` / `memory_util` / `format_stats_text`
  (`test_viewmodel.py` asserts `format_stats_text` is never called — that
  assertion + its callers go with the methods).

- [ ] **Step 4: Parsing dups.** In `gpu_telemetry/level_zero_telemetry.py`
  and adjacent: dedup `_safe_read_text` (sysfs + fdinfo copies → one
  helper), `_unique_paths` ≈ `_unique_existing_dirs` → one, and the two
  identical prefix-scan blocks → one loop.

- [ ] **Step 5: system_stats TTL + uptime.**
  Replace the 8× `type: ignore` function-attribute TTL cache in
  `_get_task_stats` with 3 lines of module state:

```python
_task_stats_cache: tuple[float, dict] | None = None
_TASK_STATS_TTL_S = 1.0

def _get_task_stats() -> dict:
    global _task_stats_cache
    now = time.monotonic()
    if _task_stats_cache and now - _task_stats_cache[0] < _TASK_STATS_TTL_S:
        return _task_stats_cache[1]
    stats = _read_task_stats()
    _task_stats_cache = (now, stats)
    return stats
```

  (`_read_task_stats` = the body that was cached — split the read from the
  cache.) Replace `_format_uptime` with
  `str(datetime.timedelta(seconds=int(seconds)))`.

- [ ] **Step 6: Benchmark.** Delete the `SubprocessResult` wrapper — call
  `subprocess.run(..., check=True)` and use `CalledProcessResult` at the 2
  call sites. Delete `_split_contiguous_blocks` (unreachable — input
  pre-filtered to `|` lines) and call `_parse_table_block` directly.

- [ ] **Step 7: Shared usage-meter helper.** In
  `components/gpu_stats.py` add:

```python
def usage_fill(percent: float | None, width: int) -> str:
    """Fill a usage-bar string; ``None`` renders all ``?``."""
    if percent is None:
        return "?" * width
    filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
    return "|" * filled + " " * (width - filled)
```

  `GPUStatsPanel._usage_meter` → `return usage_fill(percent, 10)`. In
  `components/system_health.py` (Task 6 deleted the renderer's
  `_usage_bar`; the surviving widget builds its own) →
  `from .gpu_stats import usage_fill` and call `usage_fill(percent, width)`.
  Keep the two display mappings (`_usage_level_class` vs `_usage_color`) —
  they differ.

- [ ] **Step 8: Prune tests.** `test_gpu_stats.py` /
  `test_gpu_telemetry_stats.py`: delete `collect_nvtop_stats`
  (non-selector), `make_gpu_collector`, `gpu_util`/`memory_util`/
  `format_stats_text` tests. `test_benchmark.py`: delete
  `SubprocessResult` + `_split_contiguous_blocks` tests.
  `test_foundation.py`: delete any GPU stats API tests now orphaned.

- [ ] **Step 9: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 7 — gpu_telemetry + benchmark" && git push
```

---

## Task 8: Behavior changes + AGENTS.md

**Files:**
- Modify: `src/llama_cli/ui_output.py`, `src/llama_manager/config/profile_cache.py:49`,
  `src/llama_manager/profile_orchestrator.py` (quality branch),
  `src/llama_cli/commands/profile.py` (choices + docs),
  `src/llama_cli/tui/components/digital_clock.py`,
  `src/llama_manager/probe/provenance.py`, `AGENTS.md`
- Test: `src/tests/test_ui_output.py`, `test_profile_cli.py`,
  `test_profile_orchestrator.py`, `test_cli_parser.py`,
  `test_tui.py` (logo), `test_provenance*`

**Interfaces:**
- Produces: `emit_*` on rich Console (signatures unchanged);
  `ProfileFlavor` = BALANCED|FAST; `LLM_RUNNER_LOGO` static;
  `_resolve_sha` via `git rev-parse` only.

- [ ] **Step 1: ui_output → rich Console.** Replace `src/llama_cli/ui_output.py`
  (72 lines, currently ANSI + `_tty`) with:

```python
"""User-facing output helpers for llama_cli — separate from diagnostic logging."""

from rich.console import Console
from rich.text import Text

_STDOUT = Console(highlight=False)
_STDERR = Console(stderr=True, highlight=False)


def _emit(console: Console, prefix: str, color: str, msg: str) -> None:
    console.print(Text(f"{prefix} ", color), msg, markup=False)


def emit_info(msg: str) -> None:
    """Print an informational message to stdout."""
    _emit(_STDOUT, "info:", "cyan", msg)


def emit_success(msg: str) -> None:
    """Print a success/status message to stdout."""
    _emit(_STDOUT, "ok:", "green", msg)


def emit_warn(msg: str) -> None:
    """Print a warning message to stderr."""
    _emit(_STDERR, "warn:", "yellow", msg)


def emit_error(msg: str) -> None:
    """Print an error message to stderr."""
    _emit(_STDERR, "error:", "red", msg)


def emit_plain(msg: str, *, err: bool = False) -> None:
    """Print raw text without prefix or coloring."""
    (_STDERR if err else _STDOUT).print(msg, markup=False)


def emit_heading(msg: str, *, level: int = 1) -> None:
    """Print a section heading (level 1 = #, 2 = ##, etc.) dimmed."""
    _STDOUT.print(Text(f"{'#' * level} ", style="dim"), msg, markup=False)
```

  **Critical:** `markup=False` on `msg` — messages may contain `[...]`
  (profile/bracketed strings) that Rich would otherwise parse as tags.
  `Console` auto-disables color on non-TTY, replacing `_tty()`.

- [ ] **Step 2: Digital clock logo.** In `digital_clock.py`, delete the
  LLM-side width machinery (`_LLM_WIDTH`, `_LOGO_ROWS`, the row-extension
  lists in `_build_logo_rows`) — every `_LLM_BLOCK` row is already 32
  visible chars (pre-aligned art, so LLM padding is a proven no-op).
  KEEP robot-side padding: robot rows are NOT aligned (12–16 visible
  chars), and dropping it would shift the rendered mascot. Replace the
  machinery with:

```python
import re
from itertools import zip_longest


def _pad_markup_line(s: str, width: int) -> str:
    needed = width - len(re.sub(r"\[[^\]]*\]", "", s))
    return s + " " * max(0, needed)


_ROBOT_WIDTH = max(len(re.sub(r"\[[^\]]*\]", "", r)) for r in _ROBOT_BLOCK)

LLM_RUNNER_LOGO = "\n".join(
    llm + _LOGO_GAP + _pad_markup_line(robot, _ROBOT_WIDTH)
    for llm, robot in zip_longest(_LLM_BLOCK, _ROBOT_BLOCK, fillvalue="")
)
```

  (deletes `_clean_markup`, `_LLM_WIDTH`, `_build_logo_rows`; keeps
  `_LOGO_GAP`, both block constants, the `re` import). `DigitalClockWidget`
  (Digits, 1s tick, date-right) and the rainbow wordmark + robot art are
  UNCHANGED. The logo test must still see 7 rows.

- [ ] **Step 3: Remove ProfileFlavor.QUALITY.**
  - `config/profile_cache.py:49`: delete `QUALITY = "quality"`.
  - `profile_orchestrator.py:201–208`: delete the
    `# quality — use balanced as base` fallback return (BALANCED/FAST
    branches already cover the remaining enum).
  - `commands/profile.py:428`: `choices=["balanced", "fast"]` (drop
    `"quality"`). Update the epilog example at `:417` (drop the
    `quality` line).
  - `commands/profile.py:166` `ProfileFlavor(flavor)` will now raise on
    `"quality"` → the surrounding error path already handles invalid
    flavors (verify it emits `emit_error` + exits, not a traceback).
  - On-disk profile-cache entries with `flavor="quality"`
    (`profile_cache.py:158` `ProfileFlavor(data["flavor"])`) will raise on
    load — verify the load path surfaces a clean error (does NOT silently
    map quality→balanced; no compat shim).

- [ ] **Step 4: _resolve_sha → git only.** In `probe/provenance.py:38–83`,
  delete the manual `.git/HEAD` + ref-file parsing (lines 49–65). Keep ONLY
  the `git -C <root> rev-parse HEAD` path as the single resolution:

```python
def _resolve_sha() -> str:
    """Resolve the git SHA via ``git rev-parse HEAD``, or 'unknown'."""
    cfg = Config()
    llama_cpp_root = str(cfg.paths.llama_cpp_root)
    from subprocess import TimeoutExpired, run

    try:
        result = run(
            ["git", "-C", llama_cpp_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"
```

  (`subprocess.run` never raises `CalledProcessError` without
  `check=True` — the old code's catch of it was dead.)

- [ ] **Step 5: AGENTS.md refresh (targeted edits only).**
  - Repo-layout block: replace the stale `llama_manager/` file list
    (`config.py`, `config_builder.py`, `server.py`, `process_manager.py`,
    `gpu_stats.py`, `log_buffer.py`, `colors.py` — none exist) with the
    current package layout. Match the file tree to what
    `rtk ls src/llama_manager/` actually returns.
  - Python `3.12` → `3.14` in every spot (`.python-version`,
    `requires-python`, ruff target, pyright).
  - Architecture example: `create_summary_balanced_cfg(port=8080, threads=4)`
    → the profile-registry pattern (factory deleted in Task 5).
  - **Remove the Issue-Tracking (br) section** (the `br CLI Commands`
    table, `Workflow Pattern`, `Issue Types`, `Agent Usage — Always Use --json`,
    `Session End Checklist`, `Best Practices`) — per the approved design,
    br is no longer part of the workflow. Leave `.beads/` on disk untouched.
  - Re-verify any pitfall references against the final code state (do not
    introduce new claims).
  - Do NOT restructure/merge other AGENTS.md sections (no-deletion rule).

- [ ] **Step 6: Prune + fix tests.**
  - `test_ui_output.py`: color assertions — Console emits different ANSI
    (no bare `\033[96m`). Rewire to assert visible prefix + stripped text
    (e.g. `re.sub(r"\033\[[0-9;]*m", "", captured)`), or construct
    `Console(file=..., force_terminal=True)`. Keep the semantic assertions
    (which stream, prefix, no prefix for `emit_plain`).
  - `test_profile_cli.py`: lines 270/289/294/864/878 reference `quality` —
    switch to `balanced`/`fast`, and ADD one test that `--flavor quality`
    exits non-zero (argparse `choices` rejection).
  - `test_profile_orchestrator.py:233`: `resolve_benchmark_config(cfg,
    ProfileFlavor.QUALITY, config)` → delete that test (no QUALITY member).
  - `test_cli_parser.py:510–513` (`flavor="quality"`): switch to a valid
    flavor or assert the parser now rejects `quality`.
  - `test_tui.py`: logo tests — `LLM_RUNNER_LOGO` content string changes
    (no padding). Update the literal or assert structural invariants
    (row count = 7, contains `LLM` block).
  - `test_provenance*`: `_resolve_sha` tests — delete the `.git/HEAD`
    manual-parse cases; keep the `git rev-parse` success + failure cases
    (mock `subprocess.run`).

- [ ] **Step 7: Final gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest && uv run pip-audit
git add -A -- src/ AGENTS.md && git commit -m "chore(cleanup): batch 8 — behavior changes + AGENTS.md" && git push
```

---

## Final verification (after Task 8)

- [ ] Full green: `uv run pre-commit run --all-files && uv run pytest`
- [ ] `uv run pip-audit`: the ONLY reported CVEs are the two documented
  transitive-dev `pip` ones (AGENTS.md: `CVE-2026-3219` and
  `PYSEC-2026-196`, dev deps of pip-audit itself). Anything else must be
  fixed before the final push.
- [ ] Line-count sanity: `rtk wc -l` on the major cut files confirms the
  ~3,400 net; total test count in the ~2,700–2,800 band.
- [ ] No half-deleted tree: `git log --oneline -9` shows the 8 batch
  commits (`dba780c` design doc excluded) each gate-green.
- [ ] Do NOT run any `br` command. If `.beads/` is in `git status`, flag it.

## Net removable (from the design)

- ~3,400 source lines across 8 batches.
- **0 dependencies** removed (all of loguru/psutil/httpx/gguf/rich/textual
  have live callers after cleanup).
- 2 test files deleted whole (`test_dashboard_controller_save_profile.py`,
  `test_dashboard_view_model.py`).