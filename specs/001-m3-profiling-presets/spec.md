# Feature Specification: M3 Profiling + Presets

**Feature Branch**: `m3-profiling-presets`
**Created**: 2026-04-20
**Status**: Draft
**Input**: Implement M3 from PRD — FR-007 (Manual profiling), FR-008 (Profile persistence), FR-009 (Overrides)

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Manual profiling from TUI (Priority: P1)

A user triggers profiling from the TUI for a specific model + GPU slot combination. The tool runs benchmark subprocesses against each preset flavor (balanced, fast, quality), collects performance metrics (tokens/sec, latency, VRAM usage), persists results under a cache directory keyed by GPU identifier + backend + flavor, and surfaces the data in TUI monitoring. Subsequent server launches reference cached profiles to display preset guidance and staleness warnings.

**Why this priority**: Without profiling, the preset system has no empirical basis for guidance. This is the core data-gathering flow that feeds everything else.

**Independent Test**: Can be fully tested by triggering a TUI profile request and verifying that results are persisted in cache with correct metadata keys, even with synthetic benchmark fixtures.

**Acceptance Scenarios**:

1. **Given** a configured GPU slot and model, **When** user triggers profiling from TUI, **Then** benchmark subprocess runs and results are saved under `~/.cache/llm-runner/profiles/` with GPU identifier, backend, and flavor keys.
2. **Given** profiling completes successfully, **When** TUI displays monitoring, **Then** profile guidance is available for balanced, fast, and quality flavors.
3. **Given** profiling fails for one flavor, **When** user checks results, **Then** failed flavor is marked as unavailable while successful flavors remain usable.

---

### User Story 2 — Profile persistence and cache lookup (Priority: P2)

Profiles persist across sessions under `~/.cache/llm-runner/profiles/` keyed by GPU identifiers + backend + flavor, including driver and version metadata. When a user launches a model, the system loads cached profiles to inform preset selection. Profiles carry timestamp and version stamps so the system can detect staleness.

**Why this priority**: Persistence ensures profiles survive restarts and provide ongoing tuning guidance. Without it, profiling must be re-run every session, which defeats its purpose.

**Independent Test**: Can be fully tested by writing a synthetic profile file to the cache directory and verifying that the config builder correctly loads and merges it at launch time.

**Acceptance Scenarios**:

1. **Given** a cached profile from a prior session, **When** user launches a model for that GPU slot, **Then** profile guidance is applied to preset parameters (threads, ctx_size, ubatch_size, cache settings).
2. **Given** multiple flavors (balanced, fast, quality) cached for a slot, **When** user selects a flavor, **Then** only that flavor's parameters are applied.
3. **Given** no cached profile exists for a slot, **When** user launches a model, **Then** default config values are used with no profile guidance.

---

### User Story 3 — Staleness warnings and strict mode (Priority: P3)

When a cached profile's staleness is detected (driver version changed, backend updated, or profile age exceeds threshold), the TUI and TUI-based doctor display a stale profile warning. Strict profiles mode (`--strict-profiles`) is documented but deferred to post-MVP. Staleness is a warning, not a hard block, in MVP.

**Why this priority**: Staleness warnings prevent misleading guidance from outdated profiles but are secondary to the core profiling and persistence flows. Deferred strict mode keeps MVP scope tight.

**Independent Test**: Can be fully tested by creating an intentionally stale profile (e.g., mismatched driver version hash) and verifying that the warning surfaces in TUI and doctor output.

**Acceptance Scenarios**:

1. **Given** a profile cached with a driver version hash, **When** runtime detects a different driver version, **Then** a staleness warning appears in TUI monitoring.
2. **Given** a profile older than staleness threshold, **When** doctor runs, **Then** profile is flagged as stale with guidance to re-profile.
3. **Given** `--strict-profiles` flag (post-MVP deferral noted), **When** stale profile detected, **Then** strict mode would block (but MVP only warns).

---

### User Story 4 — Deterministic override precedence (Priority: P1)

User provides explicit CLI or config overrides (e.g., `--threads`, `--ctx-size`). Override precedence follows: repo defaults < slot configuration < workstation configuration < profile guidance < explicit override. Explicit overrides always win. The merge is deterministic and documented.

**Why this priority**: Override precedence is critical for determinism — users must trust that their explicit values are honored regardless of presets or profile guidance.

**Independent Test**: Can be fully tested by providing various combinations of defaults, profile values, and explicit overrides, then verifying the final merged ServerConfig reflects correct precedence.

**Acceptance Scenarios**:

1. **Given** a profile suggests `threads=8`, **When** user specifies `--threads=12`, **Then** final config uses threads=12.
2. **Given** profile guidance for balanced flavor sets `ctx_size=16384`, **When** no explicit override is provided, **Then** final config uses ctx_size=16384 from the profile.
3. **Given** a profile suggests `ubatch_size=512`, slot config sets `ubatch_size=1024`, and user passes `--ubatch-size=2048`, **Then** final config uses ubatch_size=2048 (explicit override wins).

---

### TUI Profile Interaction

Profiling is triggered and monitored from the TUI with the following interaction patterns:

**Trigger**:
- Keybinding: `P` when a slot is focused in the status panel
- Confirmation prompt: "Profile slot \<slot_id\>? (y/N)"
- Flavor selection sub-menu: "Which flavor? (1) balanced (2) fast (3) quality"
- Profiling runs for all three flavors sequentially by default (user can select one)

**Progress Display**:
- Status panel shows: "Profiling: balanced [running...]"
- Progress bar visible during benchmark subprocess execution
- On completion: "Profiling: balanced ✓ saved to cache" or "Profiling: balanced ✗ failed — see logs"
- User can abort profiling mid-run with `Ctrl+C`; completed flavors remain saved

**Stale Warning Display**:
- Yellow warning badge next to slot status: "⚠ profile stale — \<reason\>"
- Reason text: "driver changed" / "profile 45 days old" / "binary updated"
- Badge appears in both the per-slot health row and the doctor status panel

### Profile Loading Flow

Profile loading integrates into the existing server launch sequence:

1. At server launch time, slot configuration is resolved
2. A profile loading function reads the cache directory for the matching GPU identifier + backend + flavor
3. Returns a structured override dict (or none if no profile exists or file is corrupt/missing)
4. This dict is passed as the profile layer to the existing merge function that already handles precedence: defaults < slot < workstation < profile < explicit override
5. The merge function returns both the merged configuration and a list of warning messages. If the profile is stale, a warning (including the reason: driver changed, age exceeded, or binary updated) is included in the returned warning list
6. The caller (TUI or CLI) receives the merged configuration and the warning list, displaying any warnings in the monitoring interface
7. Final merged configuration is used to construct the server launch command

**Integration point**: The existing merge function already accepts a profile layer dict. M3 extends its return type to include a warning list alongside the merged configuration. The merge precedence logic itself is not changed. Returning warnings as part of the function result (rather than mutating a passed parameter) preserves the existing functional style and avoids breaking caller contracts.

### Error Recovery

| Condition | Behavior |
| --- | --- |
| Corrupt or invalid structured data in profile file | Warning logged; profile treated as non-existent; defaults used |
| Missing required fields in profile file | Warning logged; profile treated as non-existent; defaults used |
| Unrecognized schema version | Warning logged; profile treated as non-existent; defaults used |
| File read error (permissions, I/O) | Warning logged; profile treated as non-existent; defaults used |
| Benchmark subprocess crash during profiling | That flavor marked unavailable; other flavors continue |
| Profiling timeout | Flavor marked unavailable; partial results discarded for that flavor |
| Profiling attempted while server running on slot | Blocked with user-facing message: "Server running on \<slot_id\> — stop before profiling" |
| Concurrent profile writes for same slot | File lock pattern (consistent with existing build lock); second writer blocks until first completes |
| GPU identifier changes (hardware swap) | Existing profiles become stale (driver/version hash mismatch); user prompted to re-profile |

### Cache Eviction Strategy

MVP scope for profile cache management is minimal:

- Profiles that are unparsable or have unrecognized schema versions are left on disk (not auto-deleted) for manual inspection
- `doctor` output lists all cached profiles with their staleness status, giving operators visibility to manually prune if needed
- `doctor --repair` offers to remove profiles that are stale beyond a configurable max-age threshold (default 90 days)
- No automatic background eviction in MVP — profiles are only written and read on demand

Post-MVP: automatic cache eviction based on age + access frequency + disk space constraints.

### Edge Cases

- What happens when user triggers profiling while a server is already running on that slot?
  - **Resolved**: Blocked with message "Server running on \<slot_id\> — stop before profiling"
- How does the system handle profiling timeout or benchmark subprocess crash?
  - **Resolved**: That flavor marked unavailable; other flavors continue
- What if GPU identifier changes between profile save and load (e.g., hardware swap)?
  - **Resolved**: Existing profiles become stale; user prompted to re-profile
- How does the cache handle concurrent profile writes for the same slot?
  - **Resolved**: File lock pattern (consistent with existing build lock); second writer blocks until first completes

## Requirements *(mandatory)*

### Functional Requirements

- **FR-M3-001**: System MUST support manual profiling triggered by the user for a given slot and preset flavor, collecting performance metrics (tokens per second, latency, VRAM usage) via benchmark subprocesses. Benchmark commands MUST be constructed as a list of arguments (never as a single shell string) and passed to the subprocess runner in list form. Any user-controllable benchmark configuration values MUST be validated before use. Benchmark tools are implementation-defined; graceful degradation when unavailable.
- **FR-M3-002**: System MUST persist profiles under the XDG-standard cache profiles directory, keyed by GPU identifier, backend, and flavor.
- **FR-M3-003**: System MUST store driver version metadata and profiling timestamp in each persisted profile to enable staleness detection.
- **FR-M3-004**: System MUST detect stale profiles by comparing stored driver version hash against runtime GPU state and warn in TUI monitoring.
- **FR-M3-005**: System MUST support a deterministic config merge precedence: defaults < slot < workstation < profile guidance < explicit override. The merge precedence chain is already implemented in the existing config builder; M3 adds the profile cache loading function that populates the profile layer dict from disk and integrates it into the server launch flow. Profile guidance is one layer in the precedence chain, not a replacement for defaults.
- **FR-M3-006**: System MUST provide a TUI-facing profile trigger (button/menu action) that initiates profiling for selected slot + flavor and displays progress status.
- **FR-M3-007**: System MUST allow profile results to feed preset configuration so that per-slot preset functions may optionally accept profile-guided parameter overrides.
- **FR-M3-008**: System MUST surface stale profiles in `doctor` output with actionable guidance (e.g., "Re-profile recommended: driver version changed from X to Y").
- **FR-M3-009**: System MUST NOT block model launch due to stale profiles in MVP — staleness is a warning only. `--strict-profiles` post-MVP deferral is documented.
- **FR-M3-010**: System MUST support three flavor profiles per slot: `balanced`, `fast`, `quality`. Flavor selection determines which cached profile layer is applied.
- **FR-M3-011**: System MUST provide a `profile` CLI subcommand for headless profiling parity with TUI invocation.
- **FR-M3-012**: System MUST handle benchmark subprocess failure gracefully — marking specific flavors as unavailable without crashing the rest of the profiling run.

### Constitution Alignment *(mandatory)*

- **CA-001 Code Quality**: The profiling module (`llama_manager/profiler.py`) provides pure library functions that construct benchmark commands, parse benchmark output, and manage profile cache read/write. The actual subprocess execution of benchmark tools happens in `llama_cli/` (the I/O layer), consistent with the `llama_manager` purity constraint — no `argparse`, no terminal rendering, no direct `subprocess` calls at module level in the library. The library exposes an injectable benchmark runner interface: `llama_manager` constructs the command list and accepts a callable from `llama_cli` that executes the command and returns parsed output. This matches the existing injectable collector pattern used by `GPUStats`. Benchmark runner CLI handler added in `llama_cli/cli_parser.py` and `llama_cli/server_runner.py`. All code passes `ruff` + `pyright`.
- **CA-002 Testing**: Unit tests required for: profile cache read/write (mock filesystem), merge_config_overrides with profile layer, staleness detection logic (mock GPU state), CLI subcommand parsing. Benchmark subprocess calls mocked. No GPU hardware required for CI.
- **CA-003 UX Consistency**: Profile warnings in TUI follow existing error/warning message patterns. `doctor` profile status uses the same check list output format. CLI `profile` subcommand uses same argument style as existing `build` and `setup`.
- **CA-004 Safety and Observability**: Profiling subprocesses logged to the existing `reports` module on failure. VRAM usage during profiling is captured and recorded in profile metadata. No silent profile mutation — profile writes only occur on explicit user trigger.

### Profile Cache Schema

#### Directory Structure

Profiles are stored under `~/.cache/llm-runner/profiles/` with one file per slot + backend + flavor:

```
~/.cache/llm-runner/profiles/
  nvidia-rtx3090-0-cuda-balanced.json
  nvidia-rtx3090-0-cuda-fast.json
  intel-arc-b580-0-sycl-balanced.json
  ...
```

#### ProfileRecord Structured Type Definition

The persisted profile is a structured record with the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `schema_version` | integer | Yes | Enables future schema evolution. Unrecognized versions are skipped with a warning. |
| `gpu_identifier` | string | Yes | Deterministic GPU identifier (see GPU Identifier Format below). |
| `backend` | string enum (`cuda`, `sycl`) | Yes | Must match slot backend. |
| `flavor` | string enum (`balanced`, `fast`, `quality`) | Yes | The preset flavor this profile represents. |
| `driver_version` | string | Yes | Human-readable driver version string for display in staleness warnings. |
| `driver_version_hash` | string (16 hex chars) | Yes | Short deterministic hash of `driver_version` used for staleness comparison. Algorithm: SHA-256 of the driver version string, truncated to the first 16 hexadecimal characters. |
| `server_binary_version` | string | Yes | Build identifier of the `llama-server` binary used during profiling (e.g. `b3024`). |
| `profiled_at` | ISO 8601 UTC timestamp | Yes | When profiling completed. Used for age-based staleness detection. |
| `metrics` | structured object | Yes | Performance measurements: tokens per second (float), average latency in milliseconds (float), peak VRAM in megabytes (float). |
| `parameters` | structured object | Yes | The server configuration values that produced these metrics. Limited to the profile-overridable whitelist (see Profile Override Scope). |

**JSON representation (example)**:

```json
{
  "schema_version": 1,
  "gpu_identifier": "nvidia-rtx3090-0",
  "backend": "cuda",
  "flavor": "balanced",
  "driver_version": "535.183.01",
  "driver_version_hash": "a1b2c3d4e5f67890",
  "server_binary_version": "b3024",
  "profiled_at": "2026-04-20T14:30:00Z",
  "metrics": {
    "tokens_per_second": 45.2,
    "avg_latency_ms": 12.5,
    "peak_vram_mb": 14800
  },
  "parameters": {
    "threads": 12,
    "ctx_size": 32768,
    "ubatch_size": 1024,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0"
  }
}
```

**Field semantics**:

| Field | Required | Notes |
| --- | --- | --- |
| `schema_version` | Yes | Integer. Enables future schema evolution. Unrecognized versions → skip with warning. |
| `gpu_identifier` | Yes | Deterministic GPU identifier (see GPU Identifier Format below). |
| `backend` | Yes | `cuda` or `sycl`. Must match slot backend. |
| `flavor` | Yes | `balanced`, `fast`, or `quality`. |
| `driver_version` | Yes | Human-readable driver version string for staleness display. |
| `driver_version_hash` | Yes | Deterministic hash of driver version for staleness detection. |
| `server_binary_version` | Yes | `llama.cpp` build identifier (e.g. `b3024`). |
| `profiled_at` | Yes | ISO 8601 UTC timestamp when profiling completed. |
| `metrics` | Yes | Numeric performance measurements from benchmark. |
| `parameters` | Yes | The config values that produced these metrics. |

**Serialization**: The structured type is serialized to a JSON file for persistence. The implementation provides explicit serialization helpers that convert the structured type to JSON and back, validating required fields on deserialization. Unrecognized schema versions are rejected with a warning. The implementation uses immutable structured types (frozen records) to prevent accidental mutation of cached profile data.

**Atomic writes and permissions**: Profile file writes must use an atomic write pattern (write to a temporary file, sync to disk, then rename into place) to prevent corruption from interrupted writes. Written files must have owner-only permissions (mode `0600`), matching the existing artifact write pattern in the codebase. Post-write permission verification ensures the file was created with the expected access controls.

**Missing or corrupt profile handling**: If any required field is absent or the file cannot be parsed as valid structured data, the profile is treated as non-existent (falls back to defaults). Corrupt profiles are logged and left on disk for manual inspection — not auto-deleted.

#### GPU Identifier Format

GPU identifiers are deterministic strings derived from the same sources as existing GPU statistics:

- **CUDA**: `nvidia-{gpu_name_lowercase}-{device_index}` (e.g., `nvidia-rtx3090-0`)
- **SYCL**: `intel-{gpu_name_lowercase}-{device_index}` (e.g., `intel-arc-b580-0`)

Driver version hash: deterministic short hash (first 16 hex characters) of the driver version string.

#### Config.profiles_dir Property

The configuration defaults class must expose a `profiles_dir` computed property:

```python
@property
def profiles_dir(self) -> Path:
    """Return the profile cache directory."""
    return Path(self.xdg_cache_base) / "llm-runner" / "profiles"
```

This property follows the existing XDG pattern already used by `venv_path`, `builds_dir`, and `reports_dir` in the same class.

Additionally, the configuration class must expose:

- `profile_staleness_days`: integer, default **30**. Number of days after which a profile is considered stale due to age.
- `server_binary_version`: string, default empty. The version identifier of the currently built `llama-server` binary, used for binary-version staleness detection. Populated at build time or derived from the binary path.

**Cache directory creation**: The profile cache directory must be created with owner-only permissions (mode `0700`) to prevent other users on the same system from reading or tampering with profile data. This follows the same permission pattern already used for runtime directories and build artifacts in the existing codebase.

### Staleness Detection Rules

A profile is considered stale if **any** of the following conditions are true:

| Condition | Severity | Detection |
| --- | --- | --- |
| Driver version hash mismatch | STALE | Stored hash differs from runtime driver version hash |
| Profile age exceeds threshold | STALE | `profiled_at` timestamp is older than `profile_staleness_days` default (30 days, configurable) |
| Server binary version changed | STALE | `server_binary_version` in profile differs from current built binary |
| Corrupt / unparsable profile file | UNAVAILABLE | File exists but cannot be parsed or has missing required fields |

Staleness is **warning-only** in MVP. `--strict-profiles` is documented as post-MVP deferral with the intended behavior: stale profiles would be treated as non-existent, requiring explicit user re-profiling before profile guidance is applied.

### ProfileFlavor Type

A `ProfileFlavor` is a classification with exactly three allowed values: `balanced`, `fast`, `quality`. Each flavor maps to a distinct set of optimized parameters. The flavor is selected by the user at profiling time and determines which cached profile layer is loaded at server launch.

### Profile Override Scope

Profiles can only override a **whitelisted** subset of server configuration fields. All other fields always come from defaults, slot, workstation, or explicit override layers:

| Field | Profile-Overridable | Rationale |
| --- | --- | --- |
| `threads` | Yes | Profiled per-flavor; performance-impacting |
| `ctx_size` | Yes | Profiled per-flavor; balances quality vs. VRAM |
| `ubatch_size` | Yes | Profiled per-flavor; pipeline efficiency |
| `cache_type_k` | Yes | KV cache quantization trade-off |
| `cache_type_v` | Yes | KV cache quantization trade-off |
| `n_gpu_layers` | **No** | Hardware-dependent, not profile-tuned |
| `tensor_split` | **No** | GPU topology-specific |
| `model` | **No** | Slot-specific, never overridden by profile |
| `port` | **No** | Slot-owned network binding |
| `server_bin` | **No** | Backend artifact path |
| `backend` | **No** | Backend selection, not a tuning parameter |

This whitelist prevents profiles from accidentally overriding structural config (ports, models, hardware assignments).

### Key Entities

- **ProfileRecord**: A persisted profile conforming to the Profile Cache Schema above. Contains performance metrics (tokens per second, latency, VRAM usage), driver version metadata, GPU identifier, flavor name (`balanced`/`fast`/`quality`), timestamp, and server binary version. Stored as a structured file in the cache profiles directory.
- **ProfileFlavor**: A classification representing one of three profile flavors — `balanced`, `fast`, `quality`. Each flavor maps to a distinct set of optimized parameters (threads, context size, micro-batch size, cache types).
- **ProfileOverrideLayer**: A structured collection representing profile guidance as configuration overrides. Applied as one layer in the precedence chain, limited to the whitelist of profile-overridable fields.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can trigger profiling for a single slot and flavor, and receive cached results within 5 seconds for profile loading during model launch.
- **SC-002**: Profile persistence survives a full session restart — profiles written in one session are readable and applicable in a subsequent session.
- **SC-003**: Config merge with profile guidance produces deterministic output — identical inputs produce identical merged configuration on repeated runs.
- **SC-004**: Stale profiles surface a warning in the monitoring interface within one refresh cycle of cache load and driver state comparison.
- **SC-005**: Override precedence is testable end-to-end — for each combination of defaults, slot config, profile guidance, and explicit overrides, the final value matches the highest-precedence source.
- **SC-006**: Profiling adds zero import-time side effects to the core library — module remains pure with no subprocess starts, no terminal I/O, no argument parsing at import.

## Assumptions

- Benchmark tool availability: `llama-bench` or equivalent is assumed to be co-located with the built `llama-server` binary paths already tracked in `Config`. If not available, profiling gracefully degrades with a user-facing error.
- GPU identification: GPU identifiers are derived from the same sources as existing `gpu_stats.py` (nvidia-smi for CUDA, sycl-ls for SYCL). The fingerprint format is implementation-defined but must be deterministic.
- Driver version: For staleness detection, driver version is obtained from GPU stats at profile save time and compared at profile load time. Version changes invalidate profile freshness.
- Profiles are per-slot and per-backend: a SYCL profile and a CUDA profile for the same flavor are independent.
- MVP scope limits profiling to llama.cpp only; vLLM profiling paths are reserved in the key schema but not implemented.
- Profile cache is not shared across machines; profiles are inherently machine-local and hardware-specific.
