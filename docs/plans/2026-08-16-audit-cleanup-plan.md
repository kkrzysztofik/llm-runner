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

> Rewritten from the verified inventory in
> `.superpowers/sdd/2026-08-16-audit-cleanup-plan/task-1-report.md` — the
> original brief had 12 confirmed discrepancies (wrong file paths,
> nonexistent symbols, missing affected files/tests). All file:line
> references below were re-verified against the tree on 2026-08-16.

**Files:**
- Modify (production): `src/llama_manager/orchestration/manager.py`,
  `orchestration/audit.py`, `orchestration/launcher.py`,
  `orchestration/slot_lockfile.py`, `orchestration/types.py`,
  `orchestration/risk.py`, `orchestration/launch.py`,
  `orchestration/artifact.py`, `orchestration/__init__.py`,
  `src/llama_manager/slot_state.py`, `src/llama_manager/smoke.py`,
  `src/llama_manager/risk_ack.py`, `src/llama_manager/dry_run.py`,
  `src/llama_manager/probe/smoke.py`, `src/llama_manager/probe/__init__.py`
- Test (delete/fix tests in): `src/tests/runtime/test_launcher.py`,
  `src/tests/runtime/test_slot_lockfile.py`,
  `src/tests/runtime/test_launch_flow.py`,
  `src/tests/runtime/test_audit_redaction.py`,
  `src/tests/smoke/test_probe_config_models.py`,
  `src/tests/smoke/test_smoke_lifecycle.py`,
  `src/tests/smoke/test_smoke_manager.py`,
  `src/tests/system/test_foundation_contracts.py`,
  `src/tests/tui/test_tui.py`, `src/tests/cli/test_server_runner.py`,
  `src/tests/slot/test_slot_state.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ServerManager` with the test-only API removed (live surface
  kept: `start_server_background` / `start_servers` / `launch_all_slots` /
  `shutdown_slot` / `cleanup_servers`, lock helpers, and the module-level
  `stream_pipe` / `wait_for_processes` in `launcher.py`); tokenless risk
  ack — `acknowledge_risk` without an `ack_token` param (the
  `issue_ack_token` / `validate_ack_token` chain is gone);
  `SlotRuntime` / `ArtifactMetadata` / `ProcessMetadata` /
  `ConsecutiveFailureCounter` gone.

- [ ] **Step 1: Verify deadness** (each grep must return zero production
  callers outside the listed owners before deleting that symbol):

```bash
cd /home/kmk/llm-runner
# Expected: zero production callers (fully dead)
for s in on_interrupt on_terminate _stream_pipe _wait_for_processes \
         _format_output run_server_foreground _build_launch_status_messages \
         ConsecutiveFailureCounter SlotRuntime ArtifactMetadata ProcessMetadata; do
  echo "== $s =="; rtk grep -rnw "$s" src/llama_cli src/llama_manager --include='*.py' | grep -v tests
done
# Expected: only the hits listed here (all in scope)
#   acquire_lock     -> ServerManager.acquire_lock (manager.py:384) is dead;
#                       build_pipeline/lock.py:16 acquire_lock is LIVE — do not touch
#   check_lock_stale -> manager.py:396 + slot_lockfile.py:48 (both dead)
#   issue_ack_token / validate_ack_token -> the ack chain in Step 5
#   log_path         -> AuditLogger file machinery (audit.py) + manager.py:49,59
for s in acquire_lock check_lock_stale issue_ack_token validate_ack_token log_path; do
  echo "== $s =="; rtk grep -rnw "$s" src/llama_cli src/llama_manager --include='*.py' | grep -v tests
done
# Expected: LIVE — do NOT delete: is_success, launch_count (LaunchResult
# methods, types.py:43-46,56-58), risk_result (LaunchOrchestrationResult
# field, types.py:70, consumed by tui/controller.py:1379-1381),
# lifecycle_audit (AuditLogger accessor property, audit.py:91-94 — there is
# no get_events method)
```

- [ ] **Step 2: Delete ServerManager test-only API** from
  `orchestration/manager.py`: `on_interrupt` (158-161), `on_terminate`
  (163-166), `_stream_pipe` (172-180), `_wait_for_processes` (182-184),
  `_format_output` (186-190), `run_server_foreground` (231-234),
  `acquire_lock` (384-388), `check_lock_stale` (396-400); and
  `slot_lockfile.check_lock_stale` (slot_lockfile.py:48-64). These are
  methods — there are no `orchestration/__init__.py` re-exports to remove
  for them. Keep `build_pipeline/lock.py::acquire_lock` (live — used by
  build_pipeline/pipeline.py:356).

- [ ] **Step 3: Delete the file-based audit log** from
  `orchestration/audit.py`: `_AUDIT_LOG_MAX_BYTES` / `_AUDIT_LOG_MAX_FILES`
  (12-14), `_rotate_audit_log` (17-40), `_append_audit_log` (43-64), the
  `log_path` parameter of `AuditLogger.__init__` (70-71), and the file
  append in `record_event` (84-89); drop the `audit_log_path` parameter
  threaded through `manager.py` (49, 59 — no caller passes it). Keep the
  in-memory `record_event` list + the `lifecycle_audit` property
  (audit.py:91-94).

- [ ] **Step 4: Delete dead dataclasses + probe surface.**
  - `orchestration/types.py`: `ProcessMetadata` (26-31), `SlotRuntime`
    (74-99), and the 5 duplicated permission constants (12-15, 17-22, 23:
    `ARTIFACT_CHECK_NAME`, `OWNER_ONLY_PERMISSIONS_FAILURE`,
    `PERMISSION_SUPPORT_HINT`, `PERMISSION_WRITABILITY_HINT`,
    `MAX_COLLISION_RETRIES`) — `artifact.py` keeps its own copies
    (artifact.py:15-25). `LOCKFILE_FIX_SUGGESTION` (types.py:16) is LIVE
    (manager.py:39, slot_lockfile.py:20) — keep. `LaunchResult.is_success`
    (56-58) and `LaunchResult.launch_count` (43-46) are LIVE — keep.
  - `orchestration/artifact.py`: `ArtifactMetadata` (39-54).
  - `orchestration/__init__.py`: drop the `ArtifactMetadata` (6, 52),
    `SlotRuntime` (33, 60), `ProcessMetadata` (32, 61) re-exports.
  - `probe/smoke.py`: `ConsecutiveFailureCounter` (194-227);
    `probe/__init__.py`: drop its import (5) + `__all__` entry (22).
  - `orchestration/launch.py`: `_build_launch_status_messages` (98-120) —
    zero call sites.
  - `src/llama_manager/smoke.py`: the `SmokeTarget.backend` field (50) +
    its two construction sites (86, 103) + docstring line (43) — no
    production readers (`run_smoke_probes` uses host/port/model only).
    Keep the other `SmokeTarget` fields (`slot_id`, `model`, `host`,
    `port`).
  - **D11 follow-up:** deleting the 5 constants from types.py breaks
    `launch.py:20` (`from .types import ARTIFACT_CHECK_NAME, ...`), which
    is used only by launch.py's dead `_artifact_error` (52-58 — a duplicate
    of artifact.py's own `_artifact_error`, zero call sites). Delete that
    dead function and trim the import to
    `from .types import LaunchOrchestrationResult, LaunchResult`.

- [ ] **Step 5: Cut the ack-token ceremony (full chain).** `issue_ack_token`
  has LIVE production callers — the whole chain is in scope (verified):

  ```
  launch_orchestrate            launch.py:253        ack_token = server_manager.issue_ack_token(id)
    -> _evaluate_and_handle_risks  launch.py:65 (param), 82, 259
           -> risk_ack.evaluate_risks       risk_ack.py:86 (param)
                -> _collect_risky_details   risk_ack.py:39 (param)
                     -> manager.acknowledge_risk(..., ack_token=)  manager.py:89,91
                          -> RiskAckManager.acknowledge_risk       risk.py:36, 43-44
  dry_run._build_dry_run_result dry_run.py:158 -> _risk_warnings (dry_run.py:192 param,
                                   159, 199) -> risk_ack.evaluate_risks
  ```

  Exact edits:
  - `orchestration/risk.py`: delete `issue_ack_token` (20-23) and
    `validate_ack_token` (25-29); drop the `ack_token` param + the
    ValueError check (36, 43-44) from `acknowledge_risk`. Keep
    `begin_launch_attempt`, the attempt-scoped `_risky_acknowledged_cache`,
    `is_risk_acknowledged`, `clear_all`.
  - `orchestration/manager.py`: delete delegates `issue_ack_token` (78-79)
    and `validate_ack_token` (81-82); drop the `ack_token` param from
    `acknowledge_risk` (84-91).
  - `src/llama_manager/risk_ack.py`: drop the `ack_token` param from
    `evaluate_risks` (86) and `_collect_risky_details` (39) + its
    pass-through (120, 76) and docstring (48).
  - `orchestration/launch.py`: delete line 253 (the `issue_ack_token`
    call); drop the `ack_token` param (65), the arg (82), and the call-site
    arg (259). Also drop the constant-None `risk_result` param (68) — the
    only caller (255-262) relies on the default.
  - `src/llama_manager/dry_run.py`: delete line 158; drop `ack_token` from
    `_risk_warnings` (192) and the call (159, 199).
  - Keep (verified live): `LaunchOrchestrationResult.risk_result` field
    (consumed by `llama_cli/tui/controller.py:1379-1381`); `RISK_ACK_LABEL`
    (`risk_ack.py:15`, passed as a risk_type by
    `cli/commands/dry_run.py:175`); all `begin_launch_attempt` /
    `is_risk_acknowledged` / tokenless `acknowledge_risk` call sites.

- [ ] **Step 6: Fix remaining cross-file bits.**
  - `slot_lockfile.py`: replace `from .launch import _lockfile_error` with
    `from .lockfile import _lockfile_error` (verified behaviorally
    identical: launch.py:45-49 vs lockfile.py:57-60 — same signature, same
    `ErrorCode.LOCKFILE_INTEGRITY_FAILURE` + `LOCKFILE_CHECK_NAME`).
  - `resolve_slot_runtime_status` (src/llama_manager/slot_state.py): delete
    the dead `pid_exists` fallback branch (97-100) + the `pid_exists`
    parameter (69) — the sole production caller
    (`llama_cli/tui/viewmodel.py:229`) passes either `None` (handled at
    90-91) or a `ProcessHandle` that always has `.poll`
    (protocol, launcher.py:57-63).
  - Confirm `risk_ack.py` is the single owner of the
    `RISK_ACK_LABEL = "warning_bypass"` literal. The dead copy in
    `tui/constants.py` is deleted in Task 6 Step 9 (design assigns it to
    batch 6) — do NOT delete it here.

- [ ] **Step 7: Rewrite `_SubprocessHandle`** in
  `orchestration/launcher.py` (replaces the class at line 93-120 and the
  wrap at line 80):

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
  text=True, bufsize=1`, the `# noqa: S603` and the safe-argv comment.
  `ProcessHandle` protocol conformance: `pid`/`stdout`/`stderr`/`poll` come
  from `Popen` itself. NOTE: pyright may still flag the `Popen ->
  ProcessHandle` structural return (Popen declares `stdout: TextIO | None`,
  the protocol wants `TextIOWrapper`) — if so, KEEP the existing
  `# type: ignore[return-value]` on that line; delete other ignores that
  truly vanish. `from io import TextIOWrapper` in launcher.py stays (used
  by the `ProcessHandle` protocol).

  **Test updates (D10):** `src/tests/runtime/test_launcher.py` references
  the wrapper class directly — update: import at line 14
  (`_SubprocessHandle` → `_ServerProc`), `isinstance` checks at 122 and 218,
  class `TestSubprocessHandleWaitTimeout` (143), and
  `handle._proc.terminate()` / `handle._proc.wait()` at 139-140 and 153-154
  — `_ServerProc` *is* the Popen, so `._proc` access disappears; call
  `handle.terminate()` / `handle.wait()` directly (drop the
  `# type: ignore[reportAttributeAccessIssue]` comments on those lines).

- [ ] **Step 8: Shrink `resolve_smoke_targets`**
  (src/llama_manager/smoke.py): after resolving the profile-id list, build
  targets with one comprehension instead of the both/slot special branches
  (behavior identical — both branches already produce equivalent
  `SmokeTarget` lists; verified by reading both branches at smoke.py:75-110).
  Note: `SmokeTarget` has fields `slot_id, model, host, port, backend` —
  there is no `profile_id` or `aliases` field.

- [ ] **Step 9: Prune tests.** In the test files named under Files: delete
  tests for every symbol removed in Steps 2–6:
  - `runtime/test_launcher.py`: `on_interrupt`/`on_terminate` tests
    (707-730), `_wait_for_processes` (236-254), `_format_output` (635-674),
    `run_server_foreground` (676-705), the `validate_ack_token` class
    (532-558), the invalid-token ValueError test (602-609), and the
    Step-7 `_SubprocessHandle` references (D10 above).
  - `tui/test_tui.py`: `on_interrupt`/`on_terminate` tests (823-857).
  - `runtime/test_audit_redaction.py`: `_stream_pipe` tests (143-212),
    `SlotRuntime` top import (21) + class (~968-1180), audit
    rotate/append/fchmod tests (1716-1787). Keep the rest (2474 lines).
  - `smoke/test_probe_config_models.py`: `ConsecutiveFailureCounter`
    import (17) + T032 section (861-~965).
  - `smoke/test_smoke_lifecycle.py`: `SlotRuntime` top import (21) + class
    `TestStateMachineLifecycle` (25-439). Keep `TestTuiVsCliSmokeParity`
    (441+) and `TestDryRunSmokeFlagBundleOutput` (674+).
  - `system/test_foundation_contracts.py`: `ArtifactMetadata` top import
    (30, remove one name) + tests (571-603).
  - `cli/test_server_runner.py`: delete
    `test_ack_token_validation_is_attempt_scoped` (914-920).
  - `smoke/test_smoke_manager.py`: delete `test_both_targets_have_backend`
    (38-44), fix `TestSmokeTarget` (435-458, incl. 452), strip `backend=`
    kwargs from ~16 `SmokeTarget(...)` constructions (168-408).
  - `slot/test_slot_state.py`: delete `_NoPollProcess` (8-12) and the 3
    fallback tests (43-59, incl. the 2 `pid_exists=` tests). Keep the
    poll-based tests.
  - `runtime/test_slot_lockfile.py`: delete
    `TestCheckLockStale.test_no_lockfile_returns_false` (53-61) and whole
    `TestCheckLockStaleErrorDetail` (158-173). Note: `TestCheckLockStale`
    also *contains* `shutdown_slot` tests (63-155) that must stay
    (misplaced under that class name) — rename or move when cutting.
  - `runtime/test_launch_flow.py`: drop the `ack_token` mock-setup lines
    (383, 425, 469, 526); line 320 is a stale comment — remove.
  Tests that build `ServerManager` purely to exercise removed methods go
  with them. Keep tests for `start_server_background` / `start_servers` /
  `launch_all_slots`, lockfile `acquire`/`release` (production API).

- [ ] **Step 10: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 1 — orchestration/slot/probe dead code" && git push
```

---

## Task 2: Build pipeline + toolchain

**Files:**
- Modify: `src/llama_manager/toolchain/detector.py`, `toolchain/__init__.py`,
  `toolchain/constants.py`,
  `build_pipeline/models.py`, `build_pipeline/pipeline.py`,
  `build_pipeline/orchestration.py`, `build_pipeline/utils.py`,
  `build_pipeline/lock.py`, `build_pipeline/__init__.py`,
  `build_pipeline/stages/clone.py`, `build_pipeline/status.py`,
  `build_pipeline/_context.py` (only if it imports deleted helpers),
  `src/llama_cli/commands/build.py`, `src/llama_manager/setup_venv.py`,
  `src/llama_cli/tui/components/build.py`
- Test: `src/tests/system/test_toolchain.py`, `test_setup_toolchain.py`,
  `test_foundation_contracts.py` (toolchain sections),
  `src/tests/build/test_build_pipeline_orchestration.py`
  (`run_both_backends` tests), `test_build_config.py`
  (`is_success`/`binary_size_mb` tests)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `BuildPipeline.run_build_for_backend` (only entry). The
  tool-presence surface stays `detect_tool` / `detect_toolchain` /
  `ToolchainStatus` (toolchain/detector.py:237); `VenvResult`
  (setup_venv.py) keeps no accessor methods; `BuildBackend` = SYCL|CUDA only.

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

- [ ] **Step 2: Toolchain.** Delete `parse_version` / `version_at_least`
  from `toolchain/detector.py` (:198/:223) + their re-exports
  (`toolchain/__init__.py:22-23,51-52`) and `CMAKE_MINIMUM_VERSION` from
  `toolchain/constants.py:11` (re-exported in `toolchain/__init__.py:5,45`)
  (keep `TOOL_VERSION_MISMATCH` handling if it only *emits* — grep first).
  Delete the 7 `*_HINT` re-exports + `SYCL/CUDA_REQUIRED_TOOLS` from
  `toolchain/__init__.py` (sole consumer already imports from
  `.constants`). Delete `_get_detect_tool()` / `_get_oneapi_bin()` indirection
  in `detector.py` (:39/:30) — call `detect_tool` / `_INTEL_ONEAPI_BIN`
  directly (move the `_INTEL_ONEAPI_BIN` constant from
  `toolchain/__init__.py:27` to a module-level constant in `detector.py`).
  Delete the `_COMMON_MISSING_TOOLS` tuple literal in the detector (:80-85)
  and inline it into `detect_toolchain` (:301 — its only consumer; no test
  references it).

- [ ] **Step 3: Venv + models.** In `setup_venv.py`: delete
  `VenvResult.is_valid` / `get_python_path` / `get_pip_path` (:33/:43/:55);
  delete the Windows `Scripts` fallback branches (Linux-only project). In
  `toolchain/constants.py`: delete `ToolchainHint.format_hint` /
  `is_url_available` / `required_for` (:33/:29/:26) + the initializer list
  args. In `build_pipeline/models.py`: delete
  `BuildArtifact.is_success` / `binary_size_mb` (:101/:106); delete
  `BuildConfig.CMAKE_C_COMPILER_SYCL` / `CMAKE_CXX_COMPILER_SYCL` ClassVars
  (:43/:44 — configure() hardcodes `icx`/`icpx` already at
  `stages/configure.py:150-151`); delete `MSG_SOURCES_NOT_GIT_REPO`
  (`build_pipeline/utils.py:24`); delete `BuildBackend.BOTH` (models.py:23).

- [ ] **Step 4: Pipeline.** Delete `BuildPipeline.run_both_backends`
  (pipeline.py:321 — callers already loop `run_build_for_backend` in
  `tui/controller.py:1266,1332`); also delete the `BuildBackend.BOTH` guard
  in `run()` (pipeline.py:202) since BOTH no longer exists. Delete
  `get_lock_error_message` (lock.py:117) + its re-export
  (`build_pipeline/__init__.py:8,37`) + the import (pipeline.py:11) + the
  dead `_get_lock_error_message` delegator (pipeline.py:372-373, never
  called).

- [ ] **Step 5: stages/clone.py + status.py.** Delete the
  `source_existed_before_clone` parameter chain (stages/clone.py:49-50,120,
  180,189 — always False end-to-end; keep the `source_exists()` check at
  :192); replace `getattr(ctx.config, "clone_timeout", 120)`
  (stages/clone.py:142) with a module constant `CLONE_TIMEOUT_S = 120`
  (verified: `BuildConfig` has no `clone_timeout` field); replace
  `getattr(config, "build_shallow_clone", True)` with the actual
  `config.shallow_clone` (models.py:54) at `tui/components/build.py:629,631`
  and `build_pipeline/orchestration.py:119`; delete the unreachable
  `if not parts` guard in `status.py:220`.

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
  - `test_toolchain.py`: delete the version-parsing suites
    (`TestParseVersion` :110-183, `TestVersionAtLeast` :186-233) and the
    `parse_version` / `version_at_least` imports (:21-22); move the
    `_get_detect_tool` / `_get_oneapi_bin` patch targets (currently
    `patch("llama_manager.toolchain.detect_tool")` ×12 at :410-549 and
    `patch("llama_manager.toolchain._INTEL_ONEAPI_BIN")` at :359) to
    `monkeypatch.setattr(detector, "detect_tool", ...)` and
    `monkeypatch.setattr(detector, "_INTEL_ONEAPI_BIN", ...)` in the
    remaining detector tests.
  - `test_build_pipeline_orchestration.py`: delete the `run_both_backends`
    tests (:531-574). `test_pipeline_orchestration.py` covers the kept
    `run_build_for_backend` / `_merge_config_overrides` — no change.
  - `test_pipeline_clone_sources.py`: no change — no test references
    `source_existed_before_clone` or `clone_timeout` (verified); the
    offline-continue tests (:1311+, :1759+) keep working via the
    `source_exists()` check at `stages/clone.py:192`.
  - `test_build_config.py`: delete the `is_success` tests (:161-197), the
    `binary_size_mb` tests (:199-235) and the `:583-585` assertions.
    `CMAKE_*_COMPILER_SYCL` / `MSG_SOURCES_NOT_GIT_REPO` have no test
    references (verified); `test_build_cli.py` has none either — no change.
  - `test_setup_toolchain.py`: delete the version-parsing tests (:285-310)
    and the `parse_version` / `version_at_least` imports (:32-33). No
    `is_valid`/`get_*_path`/Windows-branch tests exist (verified).
  - `test_foundation_contracts.py` (toolchain sections): delete
    `test_version_parsing_and_comparison` (:1609-1620) and
    `test_toolchain_hint_structure` (:1641-1653); drop the
    `parse_version` / `version_at_least` / `ToolchainHint` names from the
    import (:1265-1271); move the 7 `patch("llama_manager.toolchain.detect_tool")`
    targets (:1281-1382, :1668) to the detector module like above.

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
- Test: `src/tests/server/test_server.py` (validator +
  MultiValidationError tests), `src/tests/system/test_foundation_contracts.py`
  (validate_slots + sort_errors/error_count sections),
  `src/tests/server/test_dry_run_schema.py` (sort_errors calls +
  validate_backend_eligibility tests),
  `src/tests/runtime/test_launch_flow.py` +
  `src/tests/runtime/test_audit_redaction.py` (error_count assertions)

**Interfaces:**
- Consumes: nothing.
- Produces: `validation/validators.py` keeps `require_model`,
  `validate_port`, `validate_ports`, `validate_server_config`,
  `require_executable` (live: `commands/profile.py:154`,
  `server_runner.py:251`) and `detect_risky_operations` (live:
  `risk_ack.py:53-58`), plus their helpers. `errors.py` keeps the dataclasses
  + `MultiValidationError` without `sort_errors`/`error_count`.

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
  `llama_cli/commands/doctor.py` (its local `DoctorCheckResult` at
  doctor.py:66 — never touch that file). Also drop the deleted names from
  the re-export lists in `validation/__init__.py` (:4-14, :47-55) and
  `validation/commands/__init__.py` (:3-15, :23-28).

- [ ] **Step 3: Delete the dead validator chain** from
  `validation/validators.py`: `validate_slots` (:167), `_validate_slot`
  (:100), `_validate_duplicate_slots` (:81), `_convert_results_to_errors`
  (:157 — its only production caller of `sort_validation_errors`, making
  that builder.py function dead too), `validate_threads` (:28).
  `validate_backend_eligibility` (:64) is **live** — its only production
  caller is the kept `validate_server_config` (validators.py:78; used by
  `dry_run.py:163`): inline its body into `validate_server_config` and drop
  the standalone function + its `validation/__init__.py` re-exports
  (:20, :34). Keep `require_model`, `validate_port`, `validate_ports`,
  `validate_server_config`, `require_executable` (live:
  `commands/profile.py:154`, `server_runner.py:251`) and
  `detect_risky_operations` (live: `risk_ack.py:53-58`).

- [ ] **Step 4: Delete `errors.py` test-only API**:
  `MultiValidationError.sort_errors` and `.error_count` (plus any now-unused
  imports like the error-code sorting helper they pulled in).

- [ ] **Step 5: Prune tests.**
  - `src/tests/server/test_server.py`: delete `TestValidateThreads` (:85),
    `TestSortValidationErrors` (:503), `TestComputeMachineFingerprint`
    (:606), `TestCheckHardwareAllowlist` (:738), `TestAssessVramRisk`
    (:826), `TestFR005ErrorOrdering` (:1054), `TestSC002DenominatorCounting`
    (:1168), `TestFR005DeterministicOrdering` (:1302); drop the
    `error_count` tests inside `TestFR005MultiValidationErrorSchema`
    (:1003-1051, keep the `has_errors_field` test at :990); remove the
    `sort_validation_errors` / `validate_threads` imports (:12, :15).
    Keep `TestValidatePort` (:20), `TestValidatePorts` (:72),
    `TestBuildServerCmd` (:117), `TestFR005SingleErrorSchema` (:921),
    `TestMultiValidationErrorFieldTypes` (:1273).
  - `src/tests/system/test_foundation_contracts.py`: delete the
    `validate_slots` section (:290+) and the
    `sort_errors`/`error_count` coverage (:148-212, :231, :317, :329, :377,
    :703, :752, :816, :856, :946, :999, :1060, :1110, :1147, :1195); drop
    the `validate_slots` import (:41). Keep everything that imports the
    survivors.
  - `src/tests/server/test_dry_run_schema.py`: drop the 6
    `mve.sort_errors()` calls (:593, :648, :711, :763, :802, :852); delete
    the `validate_backend_eligibility` tests (:1485+) + import (:942) if
    Step 3 inlines the function (keep `validate_server_config` tests at
    :1534+).
  - `src/tests/runtime/test_launch_flow.py`: replace the `.error_count`
    assertions with `len(errors)` (:730, :774, :854, :981-1009).
  - `src/tests/runtime/test_audit_redaction.py`: replace the `.error_count`
    assertion (:81) with `len(errors)`.
  - `test_toolchain.py`: no `assess_vram_risk`/fingerprint tests exist
    (verified) — no change here.

- [ ] **Step 6: Gate + commit + push**

```bash
uv run pre-commit run --all-files && uv run pytest
git add -A -- src/ && git commit -m "chore(cleanup): batch 3 — validation clusters" && git push
```

---

## Task 4: Reports, logging, profile_orchestrator copy

**Files:**
- Modify/delete: `src/llama_manager/reports/` (shrink, not delete module —
  `reports/rotation.py` becomes a docstring-only stub after Step 2; it is
  NOT on the approved deletion list),
  `src/llama_manager/reports/redaction.py` (DELETE — folded into
  `common/security.py`), `src/llama_manager/logging_setup.py`,
  `src/llama_manager/profile_orchestrator.py`,
  `src/llama_manager/common/security.py`,
  `src/llama_cli/server_runner.py` (configure_logging_split caller)
- Test: `src/tests/system/test_reports.py`,
  `src/tests/test_logging_setup.py` (top-level tests dir, not system/),
  `src/tests/cli/test_profile_cli.py`,
  `src/tests/runtime/test_security_helpers.py` (existing common.security
  test module — receives the moved redact_sensitive tests)

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

- [ ] **Step 2: Reports.** Delete the mutating-action cluster:
  `reports/failure.py` — `MutatingActionLogEntry` (:70),
  `log_mutating_action` (:215) + the `from .rotation import
  _rotate_mutating_log` import (:11); `reports/rotation.py` —
  `rotate_reports` (:15) and `_rotate_mutating_log` (:57) — leaving
  `rotation.py` as a docstring-only stub (not on the approved deletion
  list). Drop the deleted names from `reports/__init__.py` (import block
  :3-10, `__all__` :12-18). Keep `FailureReport` (:15) and
  `write_failure_report` (:132) and its helpers (live from
  `build_pipeline/_context.py:110`).

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
   - The `TestRedactSensitive` class in `test_reports.py` (:375) moves to
     `src/tests/runtime/test_security_helpers.py` (the existing
     `common.security` test module) with UNCHANGED assertions, importing
     `redact_sensitive` from `llama_manager.common.security`.

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
  `profile_orchestrator.py`: `run_profile` (:364), `create_profile_record`
  (:319), `_default_subprocess_runner` (:481), `detect_backend` (:132 —
  the CLI `commands/profile.py` has its own local `_detect_backend` at
  :53), `_stream_to_text` (:521 — CLI local copy at profile.py:373),
  `DriverVersionProvider` (:49), `BENCHMARK_RUN_TIMEOUT_SECONDS` (:41) and
  `BENCHMARK_PROMPT_TOKENS` (:42) (both used only by the deleted
  `run_profile`/`_default_subprocess_runner`), and the now-unused imports
  `compute_driver_version_hash` / `write_profile` (:34) and
  `get_gpu_identifier` (:35). KEEP (imported by `commands/profile.py:44-48`):
  `resolve_profile_slot` (:83), `resolve_benchmark_config` (:154),
  `resolve_benchmark_binary` (:216), `get_driver_version` (:301), the
  `BenchmarkConfig` dataclass (:58 — returned by `resolve_benchmark_config`)
  and the private helpers those need (`_query_nvidia_driver` :249,
  `_query_sycl_driver` :274). `test_profile_cli.py`: NO change — it has
  zero `run_profile` references; its patch targets
  (`llama_manager.profile_orchestrator.create_default_profile_registry`
  ×9, `.resolve_benchmark_binary` ×1) point at kept names.

- [ ] **Step 6: Prune tests.** `test_reports.py`: delete the
  mutating/rotate/redaction sections; keep the `write_failure_report` tests.
  `test_logging_setup.py`: rewire `configure_logging(level=X, ...)` calls to
  `configure_logging(stderr_level=X, ...)` (~25 call sites); rewire the four
  `configure_logging_split(...)` tests (lines ~452–492) to
  `configure_logging(...)`; delete tests for the JSON-envelope helpers.
   `src/tests/runtime/test_security_helpers.py`: receives the moved
   `TestRedactSensitive` class (per Step 3 — UNCHANGED assertions; there
   are no URL tests to re-target, `redact_sensitive` has no URL pattern).

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
  `config/errors.py` (ErrorDetail.error_message — NOT common/errors.py,
  which does not exist), `config/launch_runtime.py`,
  `src/llama_manager/metadata/` (leftovers), `src/llama_manager/slot_profile_store.py`
- Delete (WHOLE, spec-approved): `src/tests/config/test_dashboard_controller_save_profile.py`,
  `src/tests/config/test_dashboard_view_model.py`
- Test: `src/tests/config/test_config_persistence.py`,
  `test_config_builders.py`, `test_profile_cache.py`, `test_slot_profile_store.py`,
  `test_spec_decode.py`, `src/tests/system/test_metadata.py`,
  `src/tests/system/test_gguf_reader.py`, `src/tests/config/test_model_index.py`,
  `src/tests/server/test_dry_run_artifacts.py` (create_summary_balanced_cfg
  fixture at :85,88), `src/tests/tui/test_config_modal.py` (expected field
  sets include the deleted defaults fields),
  `test_launch_runtime.py` (does not exist — verified)

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
  `config/__init__.py` (:7,10,11 + `__all__` :146,149,150;
  `llama_manager/__init__.py` does NOT re-export them — verified). Their
  test section in `test_config_builders.py` (:233+) goes too, plus the
  `create_summary_balanced_cfg` import + call in
  `src/tests/server/test_dry_run_artifacts.py` (:85, :88 — replace with a
  direct `ServerConfig(...)` construction).

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
  production always reads `cfg.spec_decode.<field>`) and remove it from the
  base classes of `SlotProfileSpec` (`config/profiles.py:26`) and
  `ServerConfig` (`config/server.py:25`); drop the `dict` base from
  `SpeculativeDecodingConfig` (dict-style access `cfg.spec_decode["key"]`
  is tests-only — update those tests to attribute access). In
  `config/builder.py`: delete `_SPEC_DECODE_FIELDS` (:31-33) — import
  `SPECULATIVE_DECODING_FIELD_NAMES` from `config.spec_decode` (:123)
  instead (identical content: both derive from
  `SpeculativeDecodingConfig` fields; used by `_split_spec_decode_values`
  at :35). `config/enums.py`: delete `GgufParseError` (:90).

- [ ] **Step 6: Dead Config surface + metadata.** `config/defaults.py`:
  delete `PathsConfig.venv_path` (property at :60 — zero callers, verified),
  the `ServerDefaultsConfig.spec_decode` SETTER only (:200-215 — zero
  callers) — the GETTER (:179-198) is LIVE (`config.server_defaults.spec_decode`
  in `tui/components/form_widgets.py:190,252`) and STAYS; delete the
  dead `*_qwen35_both` / `ctx_size_both_*` fields (`n_gpu_layers_qwen35_both`,
  `ubatch_size_qwen35_both`, `threads_qwen35_both`, `cache_type_qwen35_both_k/v`,
  `ctx_size_both_summary` :124, `ctx_size_both_qwen35` :125 — all verified
  zero references outside defaults.py) and `tui_launch_timeout_s` (:292) /
  `probe_latency_threshold_s` (:294). KEEP `model_qwen35_both`
  (deployment :245 — live: `persistence.py:38` + `test_config_modal.py:93`)
  and all port fields (persistence writes them). The Step-3 derived field
  sets handle persistence coverage automatically — no manual list edits
  needed. Update `test_config_modal.py` expected field sets
  (`_TOP_LEVEL_FIELDS` :125-137, `_DEPLOYMENT_FIELDS` :86-98) accordingly.
  `metadata/`: delete `_GGUF_V2/V3/V4_MAGIC` (_types.py:9-11),
  `_GENERAL_NAME_PATTERN` (_types.py:14-16), `tokenizer_type` record field
  (_types.py:39 — zero production reads) + `_detect_tokenizer_type_from_reader`
  (_reader.py:146), and the unused `GGUFMetadataRecord` fields
  `raw_path`, `tokenizer_type`, `attention_head_count`,
  `attention_head_count_kv`, `parse_timestamp`, `parse_timeout_s`,
  `prefix_cap_bytes` (verified: `_append_parsed_entry` at
  `model_index.py:389-412` copies only the other nine). (The audit's
  "`model_name` parameter of `extract_gguf_metadata`" was a misread —
  `extractor.py:50`'s signature is `model_path`, `prefix_cap_bytes`,
  `parse_timeout_s`; there is no parameter to delete there.) Update
  `test_metadata.py` / `test_gguf_reader.py` / `test_model_index.py`
  constructors/assertions that reference the deleted fields.

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
  - `common/security.py`: delete `safe_log` (:160-194 — zero production
    callers; also drop it from the import in
    `src/tests/runtime/test_security_helpers.py:3-8`).
  - `config/launch_runtime.py`: delete `LaunchRuntimeOverrides` +
    `launch_runtime_as_dict` (both verified zero callers outside
    launch_runtime.py; the `split_mode` field in `LaunchRuntime` dataclass
    + TypedDict STAYS — it's a plain string field, there is no `SplitMode`
    enum).
  - `config/errors.py`: delete `ErrorDetail.error_message` (:25-27 — zero
    callers; the `.error_message` hits elsewhere are
    `BuildResult.error_message`, a different class).
  - `common/file_ops.py`: NO MERGE (audit misread) — `atomic_exclusive_create_json`
    (:37, O_CREAT|O_EXCL lock-file create) and `atomic_write_json` (:58,
    tempfile+rename overwrite) are semantically different primitives with 6
    production callers (`build_pipeline/stages/finalize.py:90`,
    `build_pipeline/lock.py:46,60`, `slot_stats.py:348,421`,
    `config/profile_cache.py:374`, `orchestration/lockfile.py:121,237`,
    `orchestration/artifact.py:97`). Keep both; no change here.
    Merging would conflate exclusive-create with overwrite; skip unless
    the intent was factoring the shared JSON serialization.

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
  `textual_app.py`, `types.py`, `constants.py`, `system_status.tcss`,
  `components/build.py`, `components/system_health.py`,
  `components/server_column.py`, `components/server_log.py`,
  `components/__init__.py`, `components/confirm_modal.py`,
  `components/modal.py` (BINDINGS dedup step)
- Delete (WHOLE, spec-approved — see Global Constraints list):
  `src/llama_cli/tui/components/system_status.py` (Step 7),
  `src/llama_cli/tui/components/gpu_telemetry.py` (Step 6)
- Test: `src/tests/tui/test_tui.py` (GPUStatsPanel + SystemHealth widget
  tests live here — there is no `test_gpu_stats.py` / `test_system_health*`),
  `test_controller.py`, `test_viewmodel.py`, `test_build_component.py`,
  `test_textual_app.py`, `test_confirm_modal.py`

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
  + `viewmodel.py` + `types.py` + `textual_app.py`):
  The build-request flag is never set in production — `request_build`
  (controller.py:425-430) has ZERO callers (BuildModalScreen replaced it) —
  so the whole flag chain is dead. DELETE:
  - `request_build` (controller.py:425-430)
  - the `model.build_request` field (model.py:88)
  - the `viewmodel.build_request` property (viewmodel.py:46)
  - the `controller._build_request` property + setter (controller.py:303-307)
  - `viewmodel.can_select_build_target` (viewmodel.py:59-60) and its
    controller pass-through `controller.can_select_build_target`
    (controller.py:372-373; zero production callers)
  - `CommandMenuState.build_request` (types.py:22)
  - the dead `if state.build_request: return action ==
    "cancel_pending_prompt"` branch in `check_action`
    (textual_app.py:144-145) — KEEP the rest of `check_action` (its
    risk_prompt branch is live)
  - `controller.cancel_pending_prompt` (controller.py:432-439) — always
    returns False once the flag is gone
  KEEP (live behavior, verified): the `ctrl+c` / `escape` bindings
  (textual_app.py:96-101) and `action_cancel_pending_prompt` — their real
  job is the `interrupt()` fallthrough: dispatch shutdown when no risk
  prompt is pending, else refresh. Simplify the body now that
  `cancel_pending_prompt` is gone (behavior identical — `cancelled` was
  always False):

```python
    def action_cancel_pending_prompt(self) -> None:
        if self.controller.interrupt():
            self._dispatch_shutdown(exit_app=True)
            return
        self.refresh_dashboard()
```

  Tests: delete tests for the removed flag chain; keep and update the
  `action_cancel_pending_prompt` tests to the simplified body (ctrl+c /
  escape dispatch shutdown when no risk prompt is pending; refresh when a
  risk prompt is pending).

- [ ] **Step 5: View-model build pass-throughs** (`tui/viewmodel.py`):
  delete `build_selected_backends`, `build_in_progress`, `build_result`,
  `build_error`, `build_selected_backends_options`, `build_stage`,
  `build_progress_percent` properties (viewmodel.py:63-88) — build.py
  reads the model/wizard   state directly (verified: zero readers of all
  seven viewmodel properties). After this step, also delete the now
  write-only model fields `build_selected_backends` (model.py:89, write in
  `begin_build` controller.py:1186) and `build_result` (model.py:91, writes
  at controller.py:1215,1241,1247 — the only reader was the deleted
  viewmodel pass-through). KEEP the model
  `build_in_progress` field + controller pass-through (controller.py:149-155,
  read at textual_app.py:919) and the model `build_selected_backends_options`
  / `build_error` / `build_progress` fields (live via build.py:1275,
  controller.py:1219/1265, `_handle_build_progress`).

- [ ] **Step 6: GPU telemetry widget + system health chain.**
  - Delete `GPUTelemetryWidget` + `_flatten_gpu_lines` — i.e. DELETE the
     whole `components/gpu_telemetry.py` file (it contains nothing else;
     file deletion #7 above), plus `viewmodel.gpu_telemetry_lines`
     (viewmodel.py:49), `_format_gpu_stats_text` (viewmodel.py:238), and
     the `.gpu-telemetry*` CSS rules from `system_status.tcss:183-204`
     (NOT textual_app.py — no such rules there); remove the
     `GPUTelemetryWidget` import + export from `components/__init__.py`
     (lines 6, 29). Test cleanup: the 3 test methods at `test_tui.py:292–334`
     (widget compose/visibility) are deleted, and the assertion at
     `test_tui.py:2339` (`assert not list(app.query(GPUTelemetryWidget))`)
     goes with the class.
  - Delete the `SystemHealthRenderer` string-builder chain
    (`render_cpu_usage` :72, `render_memory_swap_usage` :82,
    `_format_core_grid_lines` :165, `_format_memory_row` :186, `_usage_bar`
    :126, `_usage_color` :130, `_format_uptime` :137, `_task_summary` :118,
    `_load_summary` :121, `_content_width` :107, `_memory_bar_width` :112,
    `_build_core_grid_rows` :142, `MIN/MAX_CONTENT_WIDTH` :64-65,
    `CPU_CORE_BAR_WIDTH`/`CPU_CORE_CELL_WIDTH` :66-67) from
    `components/system_health.py` — the widgets compose the snapshots
    directly. (There is no `render_system_info` — system info renders
    inline via the widget's `system_info_snapshot()` :91-102.) Keep the
    surviving `SystemHealthWidget` and `SystemStatusWidget`-fold (Step 7).

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
  - No `.system-status` CSS rule exists anywhere (the class is only set in
    Python) — nothing to keep. The `#alerts` rule in `system_status.tcss:1`
    keeps applying because the id is preserved on `SystemHealthWidget`.

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
  - Three `BINDINGS` lists re-declare the escape/ctrl-c cancel pair →
    import and reuse `form_widgets.MODAL_CANCEL_BINDINGS` (form_widgets.py:16):
    `components/confirm_modal.py:18-20` and `components/modal.py:13-15` +
    `modal.py:79-81` (two classes in modal.py). `config_modal.py:241` and
    `slot_profile_modal.py:161` already use the shared constant — copy the
    pattern. (No `tui/modals/` directory exists; modals live in
    `components/`.)

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

  - `AsyncSlotPlan` (controller.py:69-78) reduced to the fields its
    consumers actually read: `(success, messages, old_alias)` — drop
    `alias`/`profile_id` (never read; consumers use the separate
    `new_cfg`/`profile_id` params). Verified readers: `plan.success`
    (textual_app.py:674), `plan.messages` (:750), `plan.old_alias`
    (:676-685). Update its instantiation at `controller.py:606-611` and
    the test constructor at `test_textual_app.py:625`.
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
  `gpu_telemetry/stats.py`, `gpu_telemetry/vendor.py`,
  `gpu_telemetry/__init__.py`, `gpu_telemetry/level_zero_sysfs.py`,
  `gpu_telemetry/level_zero_fdinfo.py`,
  `src/llama_manager/system_stats.py`, `src/llama_manager/benchmark/`,
  `src/llama_manager/profile_orchestrator.py`,
  `src/llama_cli/commands/profile.py`,
  `src/llama_cli/tui/components/gpu_stats.py`, `components/system_health.py`
  (there is no `src/llama_manager/gpu_stats.py` — the manager-level
  `GPUStats` lives in `gpu_telemetry/stats.py`)
- Test: `src/tests/system/test_gpu_stats.py`, `test_gpu_telemetry_stats.py`,
  `test_benchmark.py`, `test_system_stats.py`, `test_foundation_contracts.py`

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
  (72-line re-export shim — production uses only
   `collect_level_zero_stats`). Point the 2 importers at
   `level_zero_telemetry`: `stats.py:15` and `__init__.py:4`. No test
  imports to update — tests patch `llama_manager.gpu_telemetry.stats.
  collect_level_zero_stats` (test_gpu_telemetry_stats.py:244,294,317,337),
  which keeps working since the name still lands in `stats`' namespace.

- [ ] **Step 3: Dead stats API.** Delete (verified zero production
  callers): `collect_nvtop_stats` — defined in `gpu_telemetry/vendor.py:174`
  (legacy non-selector aggregate — production uses
  `collect_nvtop_stats_for_selector` at vendor.py:140), with its re-exports
  in `gpu_telemetry/__init__.py:15,25`; `make_gpu_collector`
  (`gpu_telemetry/stats.py:192` + `__init__.py:10,31`);
  `GPUStats.gpu_util` / `memory_util` / `format_stats_text`
  (`gpu_telemetry/stats.py:111` / `:118` / `:90`; `test_viewmodel.py:709,723`
  asserts `format_stats_text` is never called — that assertion + its
  callers go with the methods).

- [ ] **Step 4: Parsing dups.** In `gpu_telemetry/level_zero_sysfs.py` +
  `level_zero_fdinfo.py`: dedup `_safe_read_text` (copies at
  `level_zero_sysfs.py:17` and `level_zero_fdinfo.py:17` → one helper),
  `_unique_paths` (sysfs.py:199) ≈ `_unique_existing_dirs` (sysfs.py:77)
  → one, and the two similar `card*` drm prefix-scan blocks
  (`_drm_paths_from_device_roots` sysfs.py:159-171 and
  `_drm_paths_from_class_drm` sysfs.py:174-187) → one loop.
  (`level_zero_telemetry.py` itself has no such dups — it's the ctypes
  collector.)

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

- [ ] **Step 6: Benchmark.** Delete the `SubprocessResult` wrapper
  (`benchmark/runner.py:11-22`) — replace it with the stdlib
  `subprocess.CompletedProcess` (attribute is `returncode`, not
  `exit_code`). Update all sites: `run_benchmark` (runner.py:107-115,
  reads `.exit_code`), the `BenchmarkRunner` alias (runner.py:27),
  re-exports (`benchmark/__init__.py:9,15`), `profile_orchestrator.py:22`
  (+ `_default_subprocess_runner`), and `commands/profile.py:29,184,190,250`
  (+ the constructions at :258 exit_code=130 and in `_handle_timeout`
  exit_code=124). Do NOT use `subprocess.run(..., check=True)` — non-zero
  exit codes (130 cancel, 124 timeout, other failures) are part of the
  runner protocol and `run_benchmark` maps them to `None`; keep the
  injectable `BenchmarkRunner` seam (tests inject fake runners). Delete
  `_split_contiguous_blocks` (benchmark/parser.py — unreachable, input
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
  `test_gpu_telemetry_stats.py` (both under `src/tests/system/`): delete
  `collect_nvtop_stats` (non-selector — 9 call sites in
  test_gpu_stats.py:101-208), `make_gpu_collector`,
  `gpu_util`/`memory_util`/`format_stats_text` tests. `test_benchmark.py`:
  delete `SubprocessResult` + `_split_contiguous_blocks` tests.
  `test_foundation_contracts.py`: delete any GPU stats API tests now
  orphaned.

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
- Test: `src/tests/test_ui_output.py`, `src/tests/cli/test_profile_cli.py`,
  `src/tests/config/test_profile_orchestrator.py`,
  `src/tests/cli/test_cli_parser.py`, `src/tests/tui/test_tui.py` (logo),
  `src/tests/smoke/test_probe_config_models.py` (provenance — no
  `test_provenance*` file exists)

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
    `gpu_stats.py`, `log_buffer.py`, `colors.py` — all but `log_buffer.py`
    are gone; it still exists) with the current package layout. Match the
    file tree to what `rtk ls src/llama_manager/` actually returns
    (packages: benchmark, build_pipeline, common, config, gpu_telemetry,
    metadata, orchestration, probe, reports, toolchain, validation +
    top-level modules).
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
  - `src/tests/cli/test_profile_cli.py`: lines 270/289/294/864/877
    reference `quality` — switch to `balanced`/`fast`, and ADD one test
    that `--flavor quality` exits non-zero (argparse `choices` rejection).
  - `src/tests/config/test_profile_orchestrator.py:233`:
    `resolve_benchmark_config(cfg, ProfileFlavor.QUALITY, config)` →
    delete that test (no QUALITY member).
  - `src/tests/cli/test_cli_parser.py:506-513`
    (`test_handle_profile_quality`, `flavor="quality"`): switch to a valid
    flavor or assert the parser now rejects `quality`.
  - `src/tests/tui/test_tui.py`: logo tests (3 `LLM_RUNNER_LOGO` refs,
    import at :124) — content string changes (no padding). Update the
    literal or assert structural invariants (row count = 7, contains `LLM`
    block).
  - `src/tests/smoke/test_probe_config_models.py`: `_resolve_sha` /
    provenance tests — delete the `.git/HEAD` manual-parse cases; keep the
    `git rev-parse` success + failure cases (mock `subprocess.run`).

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

## Plan-fix verification log

Per task: references checked against the tree (grep/read), references
corrected in this pass, and NEEDS-DECISION items (final state). The
interrupted run had already rewritten Tasks 1–8 from the verified
inventories; this pass re-verified every file path, symbol name, and
line number and fixed the residual drift.

- **Task 1** (Orchestration/slot/probe): 58 refs checked, 0 corrected
  (interrupted run's rewrite verified accurate — manager.py 8 symbols,
  audit.py 7, types.py 12, artifact.py 7, probe/smoke.py, launch.py,
  risk.py, slot_lockfile.py, slot_state.py, launcher.py, test_launcher.py
  D10 all match). NEEDS-DECISION: none.
- **Task 2** (Build pipeline + toolchain): 50 refs checked, 0 corrected
  this pass (interrupted run had already fixed the toolchain package
  path, `setup_venv.py`, and stage/test paths; line numbers re-verified).
  RESOLVED (controller): `tools_present` was an audit misread — no such
  symbol exists; the Interfaces line now cites the real surface
  (`detect_tool`/`detect_toolchain`/`ToolchainStatus`).
- **Task 3** (Validation clusters): 40 refs checked, 0 corrected
  (builder.py 11, validators.py 10, enums.py 3, errors.py 3, doctor.py,
  keep-list callers, 5 test paths + line numbers all match).
  NEEDS-DECISION: none.
- **Task 4** (Reports/logging/orchestrator copy): 45 refs checked,
  0 corrected (failure.py 7, rotation.py 2, reports/__init__.py,
  _context.py:110, redaction.py no-URL, security.py, profile_orchestrator
  13 symbols, logging_setup 10, callers, 4 test paths all match; the
  redaction DEVIATION NOTE is accurate). NEEDS-DECISION: none.
- **Task 5** (Config cluster): 55 refs checked, 0 corrected
  (builder.py, config/__init__.py, persistence.py 8, defaults.py 10,
  enums.py, spec_decode.py, file_ops.py, validators.py, security.py,
  errors.py, launch_runtime.py, metadata 6, profile_io.py, model_index.py,
  8 test paths all match). RESOLVED (controller): 2 — (a) `model_name`
  param of `extract_gguf_metadata` does not exist (audit misread; nothing
  to delete there); (b) `atomic_write`/`atomic_write_json` merge dropped —
  the two live helpers (`atomic_exclusive_create_json`, `atomic_write_json`)
  are semantically different primitives, both kept.
- **Task 6** (TUI dead code): 50 refs checked, 0 corrected
  (server_column.py, gpu_telemetry.py, system_status.py 17 lines,
  system_health.py 18 symbols, viewmodel.py, controller.py, textual_app.py,
  form_widgets.py, 4 modal BINDINGS, constants.py 4, system_status.tcss,
  test_tui.py all match). RESOLVED (controller): 1 — `cancel_pending_prompt`
  has no non-build branch (audit misread); the live Escape/ctrl+c path is
  the `interrupt()` fallthrough in `action_cancel_pending_prompt`. Step 4
  rewritten: delete the dead flag chain, keep + simplify the binding/action.
- **Task 7** (gpu_telemetry + benchmark): 40 refs checked, 1 corrected
  (`stats.py:14` → `stats.py:15` for the `level_zero` import; level_zero.py
  72 lines, vendor.py, __init__.py re-exports, sysfs/fdinfo dups,
  system_stats.py, benchmark runner/parser, gpu_stats.py, 4 test paths
  all match). NEEDS-DECISION: none.
- **Task 8** (Behavior changes + AGENTS.md): 30 refs checked, 4 corrected
  (test list → explicit paths; AGENTS.md layout claim "none exist" →
  "all but `log_buffer.py` are gone"; typo "agaiences" → "against";
  test line numbers 878→877 + explicit test names. ui_output.py 72 lines,
  profile_cache.py:49, orchestrator quality branch, profile.py choices,
  digital_clock.py 8 symbols, provenance.py:38-83, 6 test paths all match).
  NEEDS-DECISION: none.

**Total: 368 references checked, 5 corrected, 4 NEEDS-DECISION items —
all 4 resolved by the controller (3 audit misreads voided, 1 live-behavior
clarification incorporated into Step 4). No open items.**