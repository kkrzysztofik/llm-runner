# Design: Repository Over-Engineering Cleanup (Audit Campaign)

Date: 2026-08-16
Status: design approved in-session (parts 1–3 + br decision)
Branch: `int` (batches committed directly onto it)

## Context

A whole-repository over-engineering audit (ponytail-audit) of the 31,310-line
source tree produced a ranked findings list with a net of ~3,400 removable
source lines and **zero** removable dependencies (loguru, psutil, httpx, gguf,
rich, textual all have live callers). Findings were spot-verified in-session
(repo-wide greps + compile/import checks on Python 3.14).

This design turns that list into an executable, batched cleanup campaign.

## Scope

All audit findings execute in 8 risk-ordered batches: pure dead-code
deletion first, dedup/shrinks in the middle, behavior changes and docs last.

Approved inclusions:
- All deletions, shrinks, yagni/native rewrites from the audit.
- `ProfileFlavor.QUALITY` removal (CLI-visible choice goes away).
- `ui_output.py` rewrite onto `rich.console.Console`.
- `digital_clock.py` logo width-machinery removal.
- AGENTS.md targeted refresh (stale sections only — see batch 8).

## Non-goals

- No rewrites of the `except A, B:` syntax (22 files): legal on the
  project's Python 3.14, CI green. Out of scope for over-engineering.
- No changes to: toolchain detection logic, oneAPI env sourcing,
  cancel/process-group termination, lockfile atomicity, slot-stats
  persistence, smoke/backend behavior (beyond the listed deletions).
- TUI's 3-layer MVC (model → viewmodel → controller) stays; only dead
  pass-throughs are cut.
- `benchmark/` package split stays until a second production consumer
  appears (batch 8 leaves the audit footnote, no code change).
- No new files with new functionality, no new dependencies.
- `docs/ARCHITECTURE.md`, `docs/PRD.md` untouched in this campaign
  (stale-section refresh beyond AGENTS.md is separate work).

## Baseline

- Branch `int` at `f70e54e` (feat: split-mode configurable), tree clean.
- Atomicity rule: a batch is exactly one commit. If the maintainer's WIP
  commits to `int` mid-batch, the batch finishes or aborts first —
  `int` never carries a half-applied deletion.
- Editing discipline (AGENTS.md): no script-based code changes. Mechanical
  per-file deletions may be delegated to parallel subagents that edit by
  hand; the lead reviews every subagent diff before the gate. Subtle
  rewrites (persistence asdict, profile_io merge, ui_output Console,
  logging merge, usage-meter helper) are done directly, not delegated.

## Verification

Per batch, in order:

1. `uv run pre-commit run --all-files` (ruff lint + format, pyright)
2. `uv run pytest`
3. Both green → commit `chore(cleanup): batch-N — <theme>` → **push**
   (per-batch push keeps the GitHub remote tracking `int`; CI/SonarCloud
   see each increment while reviewable).

No br issue tracking in this campaign: no `br create`/`br close`/`br sync`
at any point. If `.beads/` ever appears in `git status`, flag and ask —
do not touch it.

## Test policy

- Every code deletion removes its tests in the same commit. No orphaned
  tests, no test referencing a deleted symbol.
- Whole dead test files — the complete deletion list (explicit permission
  recorded via approval of this spec):
  - `src/tests/config/test_dashboard_controller_save_profile.py` (batch 5)
  - `src/tests/config/test_dashboard_view_model.py` (batch 5)
- Partially-dead test files are pruned, file kept:
  `test_foundation_contracts.py` (validation cluster), `test_reports.py`
  (mutating/rotate/redaction sections; keep `write_failure_report` tests),
  `test_toolchain.py` (version suites), `test_pipeline_orchestration.py`
  + `test_pipeline_clone_sources.py` (run_both_backends), `test_profile_cli.py`
  (pipeline patches), `test_config_builders.py` (factory trio),
  `test_tui.py` / `test_controller.py` / `test_viewmodel.py` /
  `test_build_component.py` / `test_textual_app.py` (TUI batches),
  `test_gpu_stats.py` / `test_gpu_telemetry_stats.py` / `test_benchmark.py` /
  `test_system_stats.py` (batch 7).
- Shrinks keep their existing tests as proof: the persistence round-trip
  tests cover the asdict rewrite (including the `split_mode` field added in
  `f70e54e`, which hand-maintained field lists were at risk of missing);
  `test_ui_output.py` covers the Console rewrite; existing meter/uptime
  tests cover the shared helpers. New tests only where a shrink makes
  something newly observable.
- Coverage must stay above the SonarQube 80% new-code threshold; any gate
  miss is fixed inside its batch before push.
- Expected final test count: ~2,700–2,800 of 3,187 (deletion ratio, not a
  KPI).

## Batches

Ordered; each batch = one gate-green commit + push. Net lines approximate.

### Batch 1 — Orchestration / slot / probe dead code (~430)

`src/llama_manager/orchestration/`, `slot_state.py`, `smoke.py`, `probe/`,
`risk_ack.py`:

- `ServerManager` test-only API: `on_interrupt`, `on_terminate`,
  `_stream_pipe`, `_wait_for_processes`, `_format_output`,
  `run_server_foreground`, `acquire_lock`, `check_lock_stale`; plus
  `slot_lockfile.check_lock_stale`. TUI owns signals; production calls
  `stream_pipe` / `wait_for_processes` directly.
- File-based audit logging in `audit.py` (rotation + append + fchmod +
  `log_path` chain through `manager.py`) — no `ServerManager` construction
  ever passes `audit_log_path`; keep the in-memory `record_event` trail.
- `ConsecutiveFailureCounter` (probe/smoke.py) — no production consumer;
  the backoff/auto-restart behavior it promises does not exist.
- `SlotRuntime`, `ArtifactMetadata`, `ProcessMetadata` dataclasses;
  `LaunchResult.is_success` / `launch_count`; the 5 duplicate permission
  constants in `types.py` (artifact.py owns its own copies).
- `_build_launch_status_messages` (exact dup of the used
  `_build_launch_only_messages`).
- Ack-token ceremony: `issue_ack_token` / `validate_ack_token` and the
  token threading through `evaluate_risks` → `acknowledge_risk` — the token
  is deterministically `f"ack:{attempt_id}"` so validation can never fail;
  keep attempt_id-scoped ack state without the token.
- `SmokeTarget.backend` field — sole consumer never reads it.
- `slot_lockfile` private import of `_lockfile_error` from `.launch` —
  use the identical one in `.lockfile`, break the upward dependency.
- `resolve_slot_runtime_status` dead `getattr`/`pid_exists` fallback branch.
- `_evaluate_and_handle_risks` constant-None `risk_result` param.
- `resolve_smoke_targets` both/slot branches — one comprehension after
  resolving the profile-id list.
- `_SubprocessHandle` (launcher.py) — subclass `subprocess.Popen`
  overriding `wait` (TimeoutExpired → `ProcessTimeoutError`), or catch
  `TimeoutExpired` at the 3 call sites.
- `RISK_ACK_LABEL` / `"warning_bypass"`: `risk_ack.py` stays the single
  home; the dead duplicate in `tui/constants.py` is deleted in batch 6
  with the rest of that file's dead constants.

### Batch 2 — Build pipeline + toolchain (~310)

`src/llama_manager/build_pipeline/`, `setup_venv.py`, `toolchain/`,
`src/llama_cli/commands/build.py`:

- Toolchain version machinery: `parse_version`, `version_at_least`,
  `CMAKE_MINIMUM_VERSION` + re-exports — the detector never compares
  versions.
- `toolchain/__init__.py` re-exports of the 7 `*_HINT` constants +
  `SYCL/CUDA_REQUIRED_TOOLS` (sole consumer imports from `.constants`).
- `_get_detect_tool()` / `_get_oneapi_bin()` sys.modules indirections —
  exist only so tests can patch the name; patch
  `detector.detect_tool` / `detector._ONEAPI_BIN` instead.
- `VenvResult.is_valid` / `get_python_path` / `get_pip_path`;
  `ToolchainHint.format_hint` / `is_url_available` / `required_for` +
  initializer lists; Windows `Scripts` branches in `setup_venv.py`
  (Linux-only project).
- `BuildPipeline.run_both_backends` + `BuildBackend.BOTH` (callers already
  loop `run_build_for_backend`); `get_lock_error_message` + passthrough
  (contention path returns the static message); `BuildArtifact.is_success`
  / `binary_size_mb`; `BuildConfig.CMAKE_C/CXX_COMPILER_SYCL` ClassVars
  (configure hardcodes `icx`/`icpx`); `MSG_SOURCES_NOT_GIT_REPO`.
- `clone.py`: `source_existed_before_clone` always-False param chain;
  `getattr(config, "clone_timeout", 120)` nonexistent attribute → module
  constant; `build_shallow_clone` getattr (also
  `tui/components/build.py:629`); `status.py` unreachable `if not parts`.
- Shrinks: `_merge_config_overrides` → asdict merge skipping 4 derived
  fields (~8 lines); `_send_termination_signal` + `_send_kill_signal` →
  one `_signal_proc(proc, pgid, sig)`; CLI `_format_duration` import from
  `build_pipeline.utils`; `_format_success_json` →
  `BuildArtifact.to_dict()`; `_get_backends` / `_default_build_dir`
  collapse; `_COMMON_MISSING_TOOLS` comprehension.

### Batch 3 — Validation clusters (~480)

`src/llama_manager/validation/`, `config/errors.py`:

- `commands/builder.py` doctor/fingerprint/VRAM cluster:
  `DoctorCheckResult`, `DoctorReport`, `sort_validation_errors` +
  `sort_key`, `compute_machine_fingerprint` (`_get_cpu_model` /
  `_get_os_name` shell out to `cat` for `Path.read_text`-able data),
  `_get_lspci_output`, `_sycl_device_details` /
  `_sycl_dotted_device_details`, `check_hardware_allowlist`,
  `assess_vram_risk` + `VRamRecommendation` — zero production callers;
  CLI doctor has its own equivalents.
- `validators.py` dead chain: `validate_slots`, `_validate_slot`,
  `_validate_duplicate_slots`, `_convert_results_to_errors`,
  `validate_threads`, `validate_backend_eligibility`. Production launch
  path calls only `require_model` / `validate_port` / `validate_ports` /
  `validate_server_config` (which stay).
- `errors.py`: `MultiValidationError.sort_errors` / `error_count`
  (tests-only; duplicates the `sort_validation_errors` cut above).

### Batch 4 — Reports, logging, orchestrator copy (~430)

`src/llama_manager/reports/`, `logging_setup.py`,
`profile_orchestrator.py`, `common/security.py`:

- `reports` mutating-action cluster: `MutatingActionLogEntry`,
  `log_mutating_action`, `_rotate_mutating_log`, `rotate_reports` —
  nothing in TUI/CLI reads these logs. Keep `write_failure_report`
  (live from `build_pipeline/_context.py:120`).
- `redaction.py::redact_sensitive` — second redaction engine;
  `common/security.py`'s docstring mandates importing it; fold its URL
  pattern into `redact_log_line`, delete the module (3 import sites).
- `logging_setup.py`: `_JsonLogEnvelope` / `_format_json` /
  `_json_default` (json_logs already uses loguru `serialize=True`);
  `configure_logging` vs `configure_logging_split` → one function with
  optional `stderr_level` (duplicated filter closures + sink installs).
- `profile_orchestrator.py` pipeline copy: `run_profile`,
  `create_profile_record`, `_default_subprocess_runner`, `detect_backend`,
  `_stream_to_text`, `DriverVersionProvider`, benchmark-timeout constants —
  CLI `cmd_profile` re-implements all of it with cancellation support and
  imports only the live resolvers. KEEP: `resolve_profile_slot`,
  `resolve_benchmark_config`, `resolve_benchmark_binary`,
  `get_driver_version` (all imported by `commands/profile.py`).

### Batch 5 — Config cluster (~990)

`src/llama_manager/config/`, `common/`, `metadata/`, `model_index.py`,
library-root dashboard pair:

- `dashboard_controller.py` + `dashboard_view_model.py` at the library
  root — TUI has the real `DashboardController` / `DashboardViewModel` /
  `SlotProfilePayload`; library copies imported only by their 2 test
  files (which are deleted whole — see test policy).
- Legacy factory trio: `create_summary_balanced_cfg`,
  `create_summary_fast_cfg`, `create_qwen35_cfg` + `config/__init__.py`
  re-exports — production resolves via the profile registry; AGENTS.md
  example fixed in batch 8.
- `persistence.py` hand-rolled serialization → `dataclasses.asdict` +
  one generic writer (~20 lines); the four hand-maintained
  type-coercion frozensets derive from `dataclasses.fields()` types in
  `apply_config_updates` (~85 lines deleted).
- `common/profile_io.py` explicit-field TOML writer → generic
  dict→TOML (or JSON, matching the rest of the repo); eliminates the 3rd
  copy of the profile field list (builder.py, profile_io.py,
  slot_profile_store.py).
- `spec_decode.py`: `SpeculativeDecodingFieldsMixin` (12 flat
  pass-through props — production always reads `cfg.spec_decode.<field>`);
  `SpeculativeDecodingConfig(dict)` dual nature → plain dataclass
  (dict access is tests-only).
- `defaults.py` dead Config surface: `PathsConfig.venv_path`,
  `ServerDefaultsConfig.spec_decode` property+setter,
  `*_qwen35_both` / `ctx_size_both_*` fields, `tui_launch_timeout_s` +
  `probe_latency_threshold_s` (not even in the persisted set — modal
  inputs silently dropped).
- `metadata/` leftovers from the deleted raw-binary parser:
  `_GGUF_V2/V3/V4_MAGIC`, `_GENERAL_NAME_PATTERN`, `tokenizer_type` +
  `_detect_tokenizer_type_from_reader`, `extract_gguf_metadata(model_name=)`
  param, unused record fields.
- `builder.py`: `_profile_to_config_data` 42 explicit assignments →
  asdict + subdict flatten; `_SPEC_DECODE_FIELDS` ==
  `spec_decode.SPECULATIVE_DECODING_FIELD_NAMES` → import, don't re-list.
- `model_index.py`: `model_index_path(config)` — drop the reserved-for-
  future `config` param; double isolation (per-file `multiprocessing.
  Process` around a thread-with-timeout extractor) → keep one layer,
  comment the ceiling.
- `common/` zero-caller helpers: `validators.is_valid_port`,
  `profile_io.profile_dir_path`, `security.safe_log`,
  `launch_runtime.LaunchRuntimeOverrides` / `launch_runtime_as_dict`,
  `errors.ErrorDetail.error_message`.
- `common/file_ops.py`: `atomic_write` vs `atomic_write_json` → one
  helper + serializer callback.
- `config/enums.py`: `GgufParseError`, `DoctorCheckStatus` (zero
  consumers) + re-exports.

### Batch 6 — TUI dead code (~560)

`src/llama_cli/tui/` (+ `components/`):

- Model dead surface: `cpu_percentages`, `memory_usage_rows`,
  `system_info_snapshot`, `collect_memory_usage_rows_now` /
  `collect_system_info_snapshot_now` (dupes of
  `collect_system_health_snapshot`), `set_cached_slot_stats`, `stop`,
  `make_collector` (+ controller passthrough); `build_is_retrying` /
  `build_retries_remaining`; `ServerColumnState.log_lines` (always `()`);
  `types.BuildViewState` (zero references).
- Controller dead surface: legacy slot-ops trio
  (`apply_add_slot_from_form`, `add_slot_from_form`, `remove_live_slot` —
  superseded by the async compute→prepare/stage/complete path),
  `build_llama_cpp` + `_signal_handler_build` + `_original_sigint_handler`
  (app builds via `begin_build`/`run_build_loop`), `handle_slot_transition`,
  `get_stale_warning`, `is_model_index_refreshing`, `model_index_path`,
  `_STATUS_MESSAGE_LIFETIME_S` alias, `_status_messages` / `_status_lock`,
  and the ~20 one-line model pass-through props the app routinely
  bypasses (`controller.model.X`) — keep the externally used few.
- Build-request console-prompt flow: `request_build`, `_build_request`
  prop, build branches in `cancel_pending_prompt` / `check_action` /
  `viewmodel.can_select_build_target`, `model.build_request`,
  `CommandMenuState.build_request` — nothing ever sets the flag
  (BuildModalScreen replaced it). Keep `cancel_pending_prompt`'s
  non-build branch (Escape-bound) and the binding.
- View-model build passthroughs (`build_selected_backends`,
  `build_in_progress`, `build_result`, `build_error`,
  `build_selected_backends_options`, `build_stage`,
  `build_progress_percent`) — build.py reads the model/wizard state
  directly.
- `GPUTelemetryWidget` + `viewmodel.gpu_telemetry_lines` +
  `_format_gpu_stats_text` + `.gpu-telemetry*` CSS — widget never mounted
  in production; drop `__init__.py` export too.
- `SystemHealthRenderer` string-builder chain (~11 functions) — widgets
  compose the snapshots directly.
- `build.py` dead helpers: module-level `_read_build_form_fields`
  (verbatim copy of the production method), `_collect_options`,
  `navigate_wizard_step` (tests-only; `_handle_next` inlines the step
  map — point tests at the live path), `_clear_mounted`,
  `_result_content` / `build_result_content` dead pair (point tests at
  `_build_result_panel`).
- Dead app methods + LAUNCHING deferral chain: `action_interrupt_dashboard`,
  `action_create_profile`, `_refresh_add_slot_startup` →
  `viewmodel.mark_slot_launching` + `_deferred_resolve` + deferred branch
  in `_resolve_slot_status`.
- Dead constants in `tui/constants.py`: `RISK_ACK_LABEL` (duplicated —
  see batch 1), `RISK_CONFIRM_PROMPT`, `STATUS_PREFIX`,
  `STYLE_BOLD_YELLOW`.
- Five modals re-declaring escape/ctrl-c cancel `BINDINGS` → reuse
  `form_widgets.MODAL_CANCEL_BINDINGS` (2 modals already do).
- `SystemStatusWidget` (one-yield wrapper, only job is `id="alerts"`) →
  pass `id` to `SystemHealthWidget` at `textual_app.py:134`.
- `SystemHealthProvider` Protocol + `_EmptySystemHealthProvider` (one
  production implementer; the empty one exists for no-arg test
  construction) → type widgets against the viewmodel
  (TYPE_CHECKING import, existing pattern in `gpu_telemetry.py`).
- `ServerColumnPanel.on_mount` docstring-only no-op; `server_log.py`
  `perf_counter` + dual `logger.debug` instrumentation around a
  once-per-mount compose.
- Shrinks folded in: bounded status buffer → `collections.deque(maxlen=5)`
  (model.py); `AsyncSlotPlan` reduced to `(alias, profile_id, old_alias)`
  (always carried `success=True` / `messages=[]`); core-grid layout math
  duplicated between `viewmodel.cpu_usage_rows` + `_content_width` and
  `SystemHealthRenderer` → one owner; `_format_model_line` → call
  `_find_model_index_entry`; `_optional_numeric_display` import from
  `form_widgets`.

### Batch 7 — gpu_telemetry + benchmark (~120)

- `gpu_telemetry/level_zero.py` re-export shim (25+ re-exports + 2 no-op
  passthroughs) — production uses only `collect_level_zero_stats`;
  point importers (stats.py, `__init__.py`, 2 test imports) at
  `level_zero_telemetry.py`.
- Dead stats API: `collect_nvtop_stats` (legacy aggregate; production uses
  `collect_nvtop_stats_for_selector`), `make_gpu_collector`,
  `GPUStats.gpu_util` / `memory_util` / `format_stats_text`
  (`test_viewmodel.py` asserts it's never called).
- Parsing dups: `_safe_read_text` copy-pasted (sysfs + fdinfo),
  `_unique_paths` ≈ `_unique_existing_dirs`, two identical prefix-scan
  blocks in `level_zero_telemetry.py`.
- `system_stats.py`: 8× `type: ignore` function-attribute TTL cache in
  `_get_task_stats` → 3 lines of module state; `_format_uptime` →
  `str(datetime.timedelta(seconds=int))`.
- `benchmark/`: `SubprocessResult` wrapper → inline
  `CalledProcessResult` at the 2 call sites (stdlib);
  `_split_contiguous_blocks` unreachable (input pre-filtered to `|` lines)
  → call `_parse_table_block` directly.
- Shared usage-meter helper: fill string + 85/60 threshold bands
  duplicated between `GPUStatsPanel._usage_meter` and
  `SystemHealthRenderer._usage_bar` → one shared helper, keep the two
  display mappings. `SystemHealthRenderer` dies in batch 6, so the helper
  lands where batch 6 leaves the meter: `GPUStatsPanel` + `SystemHealthWidget`.

### Batch 8 — Behavior changes + docs (~120 + docs)

1. `ui_output.py` → `rich.console.Console`: six `emit_*` functions on two
   module-level `Console`s (stdout/stderr); signatures unchanged;
   Console auto-disables styling on non-TTY. `test_ui_output.py` stays
   the proof (ANSI assertions may need format tweaks).
2. `digital_clock.py`: `_clean_markup` + `_pad_markup_line` + width math
   deleted; `LLM_RUNNER_LOGO` stays a static string (rainbow wordmark +
   robot per layout preferences; Digits clock, date-on-right, 1s tick all
   untouched).
3. `ProfileFlavor.QUALITY`: enum member, the `resolve_benchmark_config`
   branch returning the identical BALANCED config, and the CLI
   `--flavor quality` choice deleted; `quality` becomes an invalid choice
   (no compat shim).
4. `probe/provenance.py::_resolve_sha`: manual `.git/HEAD` + ref-file
   parsing gone; `git rev-parse HEAD` is the one path.
5. **AGENTS.md refresh** — targeted edits only, no restructuring/merging:
   - Repo-layout block: current package layout (still lists
     `config.py` / `config_builder.py` / `server.py` / `process_manager.py`
     / `gpu_stats.py` / `log_buffer.py` / `colors.py`, none of which
     exist).
   - Python 3.12 → 3.14 (.python-version, requires-python, ruff target,
     pyright).
   - Architecture example: `create_summary_balanced_cfg(...)` →
     profile-registry pattern (factory dead after batch 5).
   - Pitfall references verified against final code state.
   - **Issue-Tracking (br) section removed** (br is no longer part of the
     workflow; `.beads/` data stays on disk, only the mandating docs go).

## Risks

| Risk | Mitigation |
| --- | --- |
| SonarCloud 80% new-code gate | New code is small (asdict writer, Console wrap, merged logging fn, shared meter) and under existing tests; gate miss fixed inside its batch before push. |
| `int` moving under us (maintainer WIP) | Batches are atomic commits; WIP landing mid-batch forces finish-or-abort. |
| Subagent mechanical deletions | Lead reviews every diff before the gate; hand edits only (AGENTS.md no-script rule). |
| `test_dashboard_*` whole-file deletion | Explicitly listed above; approval of this spec is the recorded permission; veto → keep as skipped stubs. |
| Hand-maintained config field lists already drifting (split_mode) | Batch 5 asdict rewrite removes the class of bug; round-trip tests are the proof. |

## Success criteria

- 8 gate-green commits on `int` + pushes; final `uv run pytest` green;
  `uv run pre-commit run --all-files` green.
- ~3,400 source lines net removed; ~2,700–2,800 tests remaining, all
  passing.
- No production behavior change beyond the three approved items
  (quality flavor removal, ui_output styling via rich, no others).
- AGENTS.md matches the final tree.
- `uv run pip-audit` clean at end of campaign.
