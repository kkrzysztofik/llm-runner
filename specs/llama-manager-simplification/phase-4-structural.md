# Phase 4 — Structural Changes

Depends on Tasks 1, 5, 6, 7 from earlier phases.

---

## Task 8: Split `Config` Dataclass by Domain

**Priority**: High | **Effort**: ~6h | **Files**: ~15
**Depends on**: Task 7 (extract spec decode first)

### Current State

`Config` (in `config/defaults.py`, 286 lines) has 62 fields spanning 8 domains:

| Domain | Field Count | Examples |
|---|---|---|
| XDG paths | 3 + 6 properties | `xdg_cache_base`, `xdg_state_base`, `xdg_data_base` |
| Binary paths | 3 | `llama_cpp_root`, `llama_server_bin_intel`, `llama_server_bin_nvidia` |
| Model paths | 5 | `model_summary_balanced`, `model_qwen35`, `models_dir` |
| Network | 4 | `host`, `summary_balanced_port`, `summary_fast_port`, `qwen35_port` |
| Server defaults (per-profile) | 24 | `default_ctx_size_summary`, `default_threads_qwen35`, etc. |
| Template defaults (new profiles) | 14 | `default_profile_port`, `default_batch_size`, etc. |
| Build pipeline | 7 | `build_git_remote`, `build_git_branch`, `build_retry_attempts`, etc. |
| Smoke probe | 8 | `smoke_listen_timeout_s`, `smoke_prompt`, etc. |
| Misc | 5 | `gguf_metadata_prefix_cap_bytes`, `tui_launch_timeout_s`, etc. |

### Steps

1. **Create focused dataclasses** in `config/defaults.py` (or split into submodules):

   ```python
   @dataclass
   class PathsConfig:
       xdg_cache_base: str
       xdg_state_base: str
       xdg_data_base: str
       llama_cpp_root: str
       models_dir: str
       # Properties: venv_path, builds_dir, reports_dir, etc.

   @dataclass
   class BuildPipelineConfig:
       git_remote: str
       git_branch: str
       retry_attempts: int
       retry_delay: int
       max_reports: int
       output_truncate_bytes: int
       args_default: str
       toolchain_timeout_seconds: int

   @dataclass
   class SmokeProbeConfig:
       listen_timeout_s: int
       http_request_timeout_s: int
       inter_slot_delay_s: int
       max_tokens: int
       prompt: str
       skip_models_discovery: bool
       api_key: str
       first_token_timeout_s: int
       total_chat_timeout_s: int

   @dataclass
   class ServerDefaultsConfig:
       # All default_* fields for new profile templates
       port: int
       ctx_size: int
       ubatch_size: int
       threads: int
       # ... etc
   ```

2. **Slim `Config`** to compose these:
   ```python
   @dataclass
   class Config:
       paths: PathsConfig = field(default_factory=PathsConfig)
       build: BuildPipelineConfig = field(default_factory=BuildPipelineConfig)
       smoke: SmokeProbeConfig = field(default_factory=SmokeProbeConfig)
       server_defaults: ServerDefaultsConfig = field(default_factory=ServerDefaultsConfig)
       # Keep model paths and network at top level (small, frequently accessed)
       model_summary_balanced: str = ...
       host: str = "127.0.0.1"
       # ... etc
   ```

3. **Update all consumers** — this is the bulk of the work:
   - `config/builder.py` — factory functions access `config.paths.llama_cpp_root` etc.
   - `build_pipeline/orchestration.py` — `config.build.git_remote`
   - `probe/` — `config.smoke.listen_timeout_s`
   - TUI config modal — field access patterns
   - `model_index.py`, `reports/`, `risk_ack.py`

4. **Provide backward-compatible properties** on `Config` during transition (optional):
   ```python
   @property
   def llama_cpp_root(self) -> str:
       return self.paths.llama_cpp_root
   ```
   Remove after all consumers are updated.

### Pass Criteria

- [ ] `Config` top-level field count ≤ 15
- [ ] 4 focused sub-dataclasses with clear domain boundaries
- [ ] No circular imports introduced
- [ ] All existing tests pass

---

## Task 9: Split `ServerManager` God Class

**Priority**: Critical | **Effort**: ~8h | **Files**: ~8
**Depends on**: Task 1 (redaction consolidation)

### Current State

`ServerManager` (982 lines) handles: process lifecycle, output streaming, ownership
verification, lockfile management, slot launching, audit logging, risk acknowledgement,
shutdown/cleanup, signal handling, and redaction.

Existing helper modules already extracted:
- `launcher.py` (92 lines) — `ProcessLauncher` protocol + `DefaultProcessLauncher`
- `lockfile.py` (343 lines) — standalone lockfile functions
- `artifact.py` (274 lines) — standalone artifact I/O

### Method Inventory by Responsibility

**Process Lifecycle** (9 methods):
`start_server_background`, `run_server_foreground`, `start_servers`,
`_wait_for_processes`, `wait_for_any`, `cleanup_servers`, `on_interrupt`,
`on_terminate`, `shutdown_slot`

**Output Streaming** (2 methods):
`_stream_pipe`, `_format_output`

**Ownership Verification** (3 methods):
`_verify_process_ownership`, `_filter_owned_running_pids`, `_send_signals_to_pids`

**Lockfile Management** (5 methods):
`acquire_lock`, `release_lock`, `check_lock_stale`, `_process_slot`, `launch_all_slots`

**Audit Logging** (3 methods + 2 module-level):
`_record_lifecycle_event`, `_append_audit_log`, `_rotate_audit_log`

**Risk Acknowledgement** (6 methods):
`begin_launch_attempt`, `issue_ack_token`, `validate_ack_token`,
`acknowledge_risk`, `is_risk_acknowledged`, `clear_risk_acknowledgements`

**Standalone helpers** (module-level):
`_verify_shutdown_ownership`, `_redact_sensitive`, `_evaluate_and_handle_risks`,
`_make_validation_error`, `_lockfile_error`, `_artifact_error`

### Steps

1. **Extract `AuditLogger`** into `orchestration/audit.py`:
   ```python
   class AuditLogger:
       def __init__(self, log_path: Path | None = None): ...
       def record_event(self, event: str, pid: int, details: str) -> None: ...
       def append_log(self, entry: dict) -> None: ...  # was _append_audit_log
       def rotate_log(self) -> None: ...  # was _rotate_audit_log
   ```
   - Move `_record_lifecycle_event`, `_append_audit_log`, `_rotate_audit_log`
     (module-level)
   - Move `_lifecycle_audit` list

2. **Extract `RiskAckManager`** into `orchestration/risk.py`:
   ```python
   class RiskAckManager:
       def begin_launch_attempt(self, attempt_id: str | None = None) -> str: ...
       def issue_ack_token(self, attempt_id: str) -> str: ...
       def validate_ack_token(self, token: str, attempt_id: str) -> bool: ...
       def acknowledge_risk(self, slot_id: str, risk_label: str) -> None: ...
       def is_risk_acknowledged(self, slot_id: str, risk_label: str) -> bool: ...
       def clear_all(self) -> None: ...
   ```
   - Move `begin_launch_attempt`, `issue_ack_token`, `validate_ack_token`,
     `acknowledge_risk`, `is_risk_acknowledged`, `clear_risk_acknowledgements`
   - Move `_risky_acknowledged_cache`, `_current_launch_attempt_id`

3. **Slim `ServerManager`** to process lifecycle + slot launching:
   - Keep: `start_server_background`, `run_server_foreground`, `wait_for_any`,
     `start_servers`, `cleanup_servers`, `on_interrupt`, `on_terminate`,
     `shutdown_slot`, `launch_all_slots`, `_process_slot`, `acquire_lock`,
     `release_lock`, `check_lock_stale`
   - Compose `AuditLogger` and `RiskAckManager` as attributes
   - Delegate: `self._audit.record_event(...)`, `self._risk.acknowledge_risk(...)`

4. **Move standalone helpers** from `manager.py`:
   - `_verify_shutdown_ownership` → `lockfile.py` (merge with `_verify_lock_owner`)
   - `_redact_sensitive` → already moved in Task 1
   - `_evaluate_and_handle_risks` → `risk.py` or keep as method on `ServerManager`

5. **Update `orchestration/__init__.py`** — export new classes.

6. **Update callers**:
   - `tui/controller.py` — `self.server_manager` still works, but internal delegation
     changes
   - `tui/model.py` — same
   - `commands/dry_run.py` — same

### Pass Criteria

- [ ] `ServerManager` ≤ 400 lines
- [ ] `AuditLogger` is independently testable
- [ ] `RiskAckManager` is independently testable
- [ ] No behavior change — all existing tests pass
- [ ] `launch_orchestrate` still works

---

## Task 10: Prune Root `__init__.py` Re-exports

**Priority**: High | **Effort**: ~3h | **Files**: 1 + test imports
**Depends on**: Tasks 1, 5, 6 (which change symbol names/locations)

### Current State

Root `__init__.py` re-exports ~216 symbols across 24 source submodules. Audit found
~140 symbols are never imported via the root — consumers use deep imports
(`from llama_manager.config import ...`).

### Symbols Actually Used via Root (by `llama_cli/`)

```
Config, ServerConfig, ModelSlot, ErrorCode, ErrorDetail, SlotState,
GPUStats, GpuTelemetrySelector, LogBuffer, ServerManager, LaunchResult,
RiskAckResult, ProfileFlavor, ProfileRecord, BenchmarkResult,
SubprocessResult, DryRunResult, DryRunSlotPayload, SmokeCompositeReport,
SmokeProbeConfiguration, SmokeTarget,
build_server_cmd, build_benchmark_cmd, run_benchmark,
collect_gpu_stats, collector_for_config, selector_for_config,
gpu_index_for_config, get_gpu_identifier,
compute_slot_transition, resolve_slot_runtime_status,
resolve_risk_action, resolve_smoke_targets, resolve_runtime_dir,
run_dry_run, write_dry_run_artifact,
create_default_profile_registry, load_model_index, model_index_path,
refresh_model_index, load_profile_with_staleness,
require_executable, require_model, validate_port, validate_ports,
run_smoke_probes, write_profile, resolve_backend_from_profile,
launch_orchestrate, RISK_ACK_LABEL, build_config
```

### Steps

1. **Keep only the ~50 symbols** listed above in `__init__.py`. Remove all others.

2. **Add a comment** at the top of `__init__.py`:
   ```python
   # Public API — consumers should prefer deep imports for new code.
   # This surface is maintained for backward compatibility.
   ```

3. **Update any test imports** that relied on removed re-exports — switch to deep
   imports.

### Pass Criteria

- [ ] Root `__init__.py` exports ≤ 50 symbols (was ~216)
- [ ] No consumer import breaks
- [ ] All existing tests pass
