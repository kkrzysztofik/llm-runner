# M3 Profiling + Presets — Implementation Plan

## Executive Summary

This plan implements **FR-007 (Manual profiling)**, **FR-008 (Profile persistence)**, and **FR-009 (Overrides)** from the PRD, as specified in `specs/001-m3-profiling-presets/spec.md`. The implementation adds:

1. **Profile cache layer** — structured JSON persistence under `~/.cache/llm-runner/profiles/` with atomic writes and owner-only permissions
2. **Benchmark command construction** — pure library functions that build benchmark commands (list-of-strings), parsed by an injectable runner callable (mirroring the `GPUStats` collector pattern)
3. **Profile-guided config merging** — `merge_config_overrides` extended with a non-breaking `warnings` side-effect parameter, with staleness detection and a 5-field override whitelist
4. **CLI subcommand** — `llm-runner profile <slot> <flavor>` for headless profiling parity
5. **Doctor integration** — profile staleness checks in `doctor check` output
6. **TUI integration** — profile trigger keybinding (`P`) and stale warning badge

**Constraint enforcement**: `llama_manager/` remains pure library — no subprocess at module level, no Rich, no argparse. Benchmark execution happens in `llama_cli/`.

---

## Module Breakdown

### New Files (3)

| File                               | Purpose                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `src/llama_manager/profile_cache.py` | ProfileRecord dataclass, cache read/write, staleness detection, GPU identifier generation |
| `src/llama_manager/benchmark.py`     | Benchmark command construction, output parsing (pure library)                             |
| `src/llama_cli/profile_cli.py`       | Profile CLI subcommand handler, profile execution orchestration                           |

### Modified Files (7)

| File                                | Changes                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `src/llama_manager/config.py`         | Add `profiles_dir` property, `profile_staleness_days`, `server_binary_version` fields |
| `src/llama_manager/config_builder.py` | Extend `merge_config_overrides` with optional `warnings` side-effect parameter, add whitelist filter |
| `src/llama_manager/gpu_stats.py`      | Add `get_gpu_identifier()` function for deterministic GPU naming                 |
| `src/llama_manager/__init__.py`       | Export new symbols                                                              |
| `src/llama_cli/cli_parser.py`         | Add `profile` subcommand parsing                                                  |
| `src/llama_cli/server_runner.py`      | Wire profile subcommand handler                                                 |
| `src/llama_cli/doctor_cli.py`         | Add profile staleness checks to doctor                                          |
| `src/llama_cli/tui_app.py`            | Add profile trigger and stale warning display                                   |

### Test Files (new tests in existing files, + 2 new test files)

| File                             | Purpose                                                            |
| -------------------------------- | ------------------------------------------------------------------ |
| `src/tests/test_profile_cache.py`  | ProfileRecord serialization, cache read/write, staleness detection |
| `src/tests/test_benchmark.py`      | Benchmark command construction, output parsing                     |
| `src/tests/test_config.py`         | ADD — new Config fields, profiles_dir property                     |
| `src/tests/test_us3_precedence.py` | ADD — profile_config merging with warnings side-effect             |
| `src/tests/test_cli_parser.py`     | ADD — `profile` subcommand parsing                                 |
| `src/tests/test_server_runner.py`  | ADD — profile handler wiring                                       |
| `src/tests/test_doctor_cli.py`     | ADD — profile staleness checks                                     |
| `src/tests/test_tui_app.py`        | ADD — profile trigger data flow (mock Rich)                        |

---

## Data Model Definitions

### 1. ProfileFlavor Enum (StrEnum)

```python
# src/llama_manager/profile_cache.py

from enum import StrEnum

class ProfileFlavor(StrEnum):
    BALANCED = "balanced"
    FAST = "fast"
    QUALITY = "quality"
```

### 2. ProfileMetrics Dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """Performance metrics from a benchmark run."""
    tokens_per_second: float
    avg_latency_ms: float
    peak_vram_mb: float
```

### 3. ProfileRecord Dataclass (frozen)

```python
from dataclasses import dataclass, field
from typing import Any

# The 5 profile-overridable fields
PROFILE_OVERRIDE_FIELDS: frozenset[str] = frozenset([
    "threads",
    "ctx_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
])

CURRENT_SCHEMA_VERSION: int = 1

@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """Persisted profile record (immutable).

    Serialized to JSON in ~/.cache/llm-runner/profiles/.
    Unrecognized schema versions cause the profile to be skipped with a warning.
    Uses Unix timestamp (float) for profiled_at to match existing LockMetadata pattern.
    """
    schema_version: int
    gpu_identifier: str
    backend: str  # "cuda" or "sycl"
    flavor: ProfileFlavor
    driver_version: str
    driver_version_hash: str  # 16 hex chars
    server_binary_version: str
    profiled_at: float  # Unix timestamp, consistent with LockMetadata.started_at
    metrics: ProfileMetrics
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileRecord | None":
        """Deserialize from dict. Returns None if required fields missing or schema unsupported."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        ...
```

### 4. StalenessResult

```python
from enum import StrEnum

class StalenessReason(StrEnum):
    DRIVER_CHANGED = "driver_changed"
    BINARY_CHANGED = "binary_changed"
    AGE_EXCEEDED = "age_exceeded"
```

### 5. Config Extensions

```python
# In src/llama_manager/config.py — additions to existing Config class

@property
def profiles_dir(self) -> Path:
    """Return the profile cache directory."""
    return Path(self.xdg_cache_base) / "llm-runner" / "profiles"

# New fields (with defaults)
profile_staleness_days: int = 30
server_binary_version: str = ""  # Populated from SERVER_BINARY_VERSION env var, default ""
```

### 6. merge_config_overrides Extension (Non-Breaking)

```python
# Current signature preserved — no breaking change:
def merge_config_overrides(
    defaults: Config,
    slot_config: dict | None = None,
    workstation_config: dict | None = None,
    profile_config: dict | None = None,
    override_config: dict | None = None,
    warnings: list[str] | None = None,
) -> ServerConfig:
    """Returns merged ServerConfig.

    If ``warnings`` is provided (a mutable list), staleness and override
    warnings are appended to it. This preserves backward compatibility
    while enabling callers that need warning accumulation.
    """
```

### 7. SubprocessResult Dataclass

```python
@dataclass(frozen=True)
class SubprocessResult:
    """Result of a benchmark subprocess execution."""
    exit_code: int
    stdout: str
    stderr: str
```

---

## API Signatures for New Modules

### `src/llama_manager/profile_cache.py`

```python
"""Profile cache: read, write, and staleness detection for profiling results."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# --- Types ---

class ProfileFlavor(StrEnum):
    BALANCED = "balanced"
    FAST = "fast"
    QUALITY = "quality"

PROFILE_OVERRIDE_FIELDS: frozenset[str] = frozenset([
    "threads", "ctx_size", "ubatch_size", "cache_type_k", "cache_type_v",
])
CURRENT_SCHEMA_VERSION: int = 1

class StalenessReason(StrEnum):
    DRIVER_CHANGED = "driver_changed"
    BINARY_CHANGED = "binary_changed"
    AGE_EXCEEDED = "age_exceeded"

@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    tokens_per_second: float
    avg_latency_ms: float
    peak_vram_mb: float

@dataclass(frozen=True, slots=True)
class ProfileRecord:
    schema_version: int
    gpu_identifier: str
    backend: str
    flavor: ProfileFlavor
    driver_version: str
    driver_version_hash: str
    server_binary_version: str
    profiled_at: float  # Unix timestamp
    metrics: ProfileMetrics
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileRecord | None"
    def to_dict(self) -> dict[str, Any]

@dataclass(frozen=True)
class StalenessResult:
    is_stale: bool
    reasons: list[StalenessReason]
    driver_version_display: str
    age_days: float

    @property
    def warning_message(self) -> str

# --- Filename Sanitization ---

_PROFILE_NAME_PATTERN: re.Pattern[str] = re.compile(r"[^a-z0-9_-]")

def _sanitize_filename_component(name: str) -> str:
    """Sanitize a string for use in a filename component.
    
    Removes all characters except lowercase letters, digits, underscores, and hyphens.
    Raises ValueError if the result is empty.
    """
    normalized = _PROFILE_NAME_PATTERN.sub("", name.strip().lower())
    if not normalized:
        raise ValueError("filename component must contain at least one valid character")
    return normalized

# --- GPU Identifier ---

def compute_gpu_identifier(backend: str, gpu_name: str, device_index: int) -> str
    """Compute deterministic GPU identifier.
    
    CUDA: nvidia-{gpu_name_lowercase}-{device_index}
    SYCL: intel-{gpu_name_lowercase}-{device_index}
    """

def compute_driver_version_hash(driver_version: str) -> str
    """SHA-256 of driver_version string, truncated to first 16 hex chars."""

# --- Cache I/O ---

def ensure_profiles_dir(profiles_dir: Path) -> None
    """Create profiles directory with owner-only permissions (0700)."""

def get_profile_path(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
) -> Path
    """Get the file path for a profile.
    
    All components are sanitized before filename construction to prevent path traversal.
    Format: {sanitized_gpu_id}-{sanitized_backend}-{sanitized_flavor}.json
    """

def read_profile(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
) -> ProfileRecord | None
    """Read and deserialize a profile. Returns None if not found, corrupt, or unsupported schema."""

def write_profile(
    profiles_dir: Path,
    record: ProfileRecord,
) -> Path
    """Atomic write profile to disk using _atomic_write_json pattern.
    
    File permissions: FILE_MODE_OWNER_ONLY (0600).
    Directory created if missing with DIR_MODE_OWNER_ONLY (0700).
    Returns the written path.
    """

# --- Staleness ---

def check_staleness(
    record: ProfileRecord,
    current_driver_version: str,
    current_binary_version: str,
    staleness_days: int,
) -> StalenessResult
    """Check if a profile record is stale.
    
    Staleness conditions:
    - driver_version_hash mismatch
    - server_binary_version mismatch
    - age > staleness_days
    """

def load_profile_with_staleness(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
    current_driver_version: str,
    current_binary_version: str,
    staleness_days: int,
) -> tuple[ProfileRecord | None, StalenessResult | None]
    """Load a profile and check staleness. Returns (record, staleness_result)."""

# --- Profile Config Extraction ---

def profile_to_override_dict(record: ProfileRecord) -> dict[str, Any]
    """Extract profile parameters as a config override dict (filtered to PROFILE_OVERRIDE_FIELDS)."""
```

### `src/llama_manager/benchmark.py`

```python
"""Benchmark command construction and output parsing.

Pure library module — no subprocess calls at module level.
Benchmark execution is handled by an injectable runner callable.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SubprocessResult:
    """Result of a benchmark subprocess execution."""
    exit_code: int
    stdout: str
    stderr: str

@dataclass(frozen=True)
class BenchmarkResult:
    """Parsed benchmark result."""
    tokens_per_second: float
    avg_latency_ms: float
    peak_vram_mb: float | None

# --- Command Construction ---

def build_benchmark_cmd(
    server_bin: str,
    model: str,
    port: int,
    threads: int,
    ctx_size: int,
    ubatch_size: int,
    cache_type_k: str,
    cache_type_v: str,
    n_gpu_layers: int | str = "all",
) -> list[str]:
    """Build llama-bench command as list of strings.
    
    The benchmark tool is assumed to be co-located with llama-server
    (same directory, different name) or available on PATH.
    
    Returns a subprocess-safe list of arguments.
    """

# --- Output Parsing ---

def parse_benchmark_output(output: str) -> BenchmarkResult | None
    """Parse benchmark tool stdout/stderr for metrics.
    
    Expected output format (implementation-defined, llama.cpp bench tool):
    - tokens/s line
    - latency line
    - VRAM line (if available)
    
    Returns None if output cannot be parsed.
    """

# --- Injectable Runner (mirrors GPUStats collector pattern) ---

BenchmarkRunner = Callable[[list[str]], SubprocessResult]
"""Type alias for benchmark runner callable.
    
Args:
    cmd: List of command arguments (subprocess-safe list form).
    
Returns:
    SubprocessResult with exit_code, stdout, stderr.
    The caller (llama_cli) provides the actual subprocess implementation.
    The library constructs the command and accepts the result.
"""

def run_benchmark(
    cmd: list[str],
    runner: BenchmarkRunner,
) -> BenchmarkResult | None:
    """Run a benchmark and parse results.
    
    Args:
        cmd: Benchmark command as list of strings.
        runner: Injectable callable that executes the command.
    
    Returns:
        Parsed BenchmarkResult or None if benchmark failed/could not parse.
    """
```

### `src/llama_cli/profile_cli.py`

```python
"""Profile CLI subcommand for headless profiling.

Usage: llm-runner profile <slot_id> <flavor> [options]

Flavors: balanced, fast, quality
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from llama_manager import (
    Config,
    ProfileFlavor,
    ProfileRecord,
    ProfileMetrics,
    compute_gpu_identifier,
    compute_driver_version_hash,
    write_profile,
    ensure_profiles_dir,
    get_profile_path,
    build_benchmark_cmd,
    run_benchmark,
    SubprocessResult,
)

def _default_subprocess_runner(cmd: list[str]) -> SubprocessResult:
    """Default benchmark runner using subprocess.run with shell=False.
    
    SECURITY: Always passes cmd as list[str] with shell=False (the default).
    Benchmark binary path is validated with require_executable() before execution.
    """
    ...

def create_profile_parser() -> argparse.ArgumentParser
    """Create argument parser for profile subcommand."""

def cmd_profile(parsed: argparse.Namespace) -> int
    """Execute profile subcommand.
    
    Steps:
    1. Validate slot is not running (check lockfile)
    2. Resolve benchmark binary path from Config
    3. Validate benchmark binary with require_executable()
    4. Construct benchmark command via build_benchmark_cmd
    5. Run benchmark via _default_subprocess_runner
    6. Parse results
    7. Write profile to cache
    """
```

---

## Integration Sequence

### Phase 1: Data Model & Cache Layer (Foundation)

**Files**: `profile_cache.py`, `config.py` (extensions), `gpu_stats.py` (extension)

1. **Add `ProfileFlavor`, `ProfileMetrics`, `ProfileRecord`** to `profile_cache.py`
2. **Add `compute_gpu_identifier`** and **`compute_driver_version_hash`** to `profile_cache.py`
3. **Add `get_gpu_identifier()`** to `gpu_stats.py` — parses `nvidia-smi`/`sycl-ls` for GPU name and index
4. **Add `profiles_dir` property, `profile_staleness_days`, `server_binary_version`** to `Config` class. `server_binary_version` reads from `SERVER_BINARY_VERSION` env var, defaulting to `""`
5. **Implement `read_profile`, `write_profile`, `ensure_profiles_dir`** using existing `_atomic_write_json` pattern and `FILE_MODE_OWNER_ONLY`/`DIR_MODE_OWNER_ONLY` constants from `process_manager.py`
6. **Implement `check_staleness`** with three conditions: driver hash mismatch, binary version mismatch, age > threshold
7. **Export new symbols** from `__init__.py`

**Tests**: `test_profile_cache.py`
- `test_profile_record_serialization_roundtrip`
- `test_profile_record_missing_fields_returns_none`
- `test_profile_record_unsupported_schema_returns_none`
- `test_compute_gpu_identifier_cuda`
- `test_compute_gpu_identifier_sycl`
- `test_compute_driver_version_hash`
- `test_staleness_driver_mismatch`
- `test_staleness_binary_mismatch`
- `test_staleness_age_exceeded`
- `test_staleness_fresh_profile`
- `test_read_profile_not_found_returns_none`
- `test_read_profile_corrupt_returns_none`
- `test_write_read_profile_roundtrip` (tmp_path)
- `test_ensure_profiles_dir_creates_with_0700` (tmp_path)
- `test_sanitize_filename_component`
- `test_sanitize_filename_component_rejects_empty`

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

### Phase 2: Benchmark Module (Pure Library)

**Files**: `benchmark.py`

1. **Implement `build_benchmark_cmd`** — constructs `llama-bench` command as `list[str]`
2. **Implement `parse_benchmark_output`** — parses benchmark stdout for metrics
3. **Define `SubprocessResult`** dataclass and `BenchmarkRunner` callable type
4. **Implement `run_benchmark`** — accepts injectable runner, returns parsed result

**Key design decisions**:
- Benchmark command uses `llama-bench` (co-located with llama-server) with `--model`, `--port`, `--threads`, `--ctx-size`, `--ubatch-size`, `--cache-type-k`, `--cache-type-v` flags
- Output parsing is best-effort; returns `None` on parse failure (graceful degradation per FR-M3-001)
- The runner callable is injected from `llama_cli` — matches GPUStats pattern

**Tests**: `test_benchmark.py`
- `test_build_benchmark_cmd_contains_required_flags`
- `test_build_benchmark_cmd_is_list_of_strings`
- `test_build_benchmark_cmd_n_gpu_layers_all`
- `test_parse_benchmark_output_success`
- `test_parse_benchmark_output_empty_returns_none`
- `test_parse_benchmark_output_partial_parses_what_it_can`
- `test_run_benchmark_calls_runner`
- `test_run_benchmark_returns_none_on_nonzero_exit`

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

### Phase 3: Config Builder Extension (Non-Breaking)

**Files**: `config_builder.py`

1. **Add optional `warnings: list[str] | None = None` parameter** to `merge_config_overrides` — preserves backward compatibility
2. **Filter `profile_config` through `PROFILE_OVERRIDE_FIELDS`** before merging — only whitelisted fields are applied
3. **Append staleness warnings** to the provided warnings list when profile data is stale
4. **No changes to existing callers** needed — they pass `warnings=None` (the default)

**Key design decisions**:
- The whitelist filter ensures profiles cannot override structural config (ports, models, hardware assignments)
- The warnings parameter is a mutable side-effect — consistent with Python patterns for accumulating errors
- Existing callers are unaffected; new callers can pass a list to receive warnings

**Tests**: `test_us3_precedence.py` (additions)
- `test_merge_with_profile_config_applies_overrides`
- `test_merge_profile_config_ignores_non_whitelisted_fields`
- `test_merge_with_warnings_list_populated`
- `test_merge_stale_profile_includes_warning`
- `test_merge_all_precedence_levels`

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

### Phase 4: CLI Subcommand (Headless Profiling)

**Files**: `cli_parser.py`, `profile_cli.py`, `server_runner.py`

1. **Add `profile` to `VALID_MODES`** in `cli_parser.py`
2. **Add `_handle_profile_case`** function in `cli_parser.py` — parses `profile <slot_id> <flavor>` with `--json` flag
3. **Create `profile_cli.py`** with `cmd_profile` handler:
   - Validates slot is not running (check lockfile using existing `read_lock` or `ServerManager` state)
   - Resolves benchmark binary path from Config — validates with `require_executable()`
   - Resolves GPU identifier via `get_gpu_identifier()` from `gpu_stats.py`
   - Constructs benchmark command via `build_benchmark_cmd`
   - Runs benchmark via `subprocess.run(cmd_list, shell=False)` (in `llama_cli` layer)
   - Parses results, writes profile via `write_profile`
4. **Wire `profile` subcommand** in `server_runner.py` — add handler in `main()` function

**Key design decisions**:
- `profile` subcommand follows the same pattern as `doctor` and `setup` — parsed in `cli_parser.py`, handled in a dedicated module
- Benchmark subprocess execution happens in `profile_cli.py` (I/O layer), not in `benchmark.py` (pure library)
- Slot-running check uses existing lockfile mechanism from `process_manager.py`
- Benchmark binary path validated with `require_executable()` before execution

**Tests**: `test_server_runner.py` (additions)
- `test_cmd_profile_validates_slot_not_running`
- `test_cmd_profile_validates_benchmark_binary`
- `test_cmd_profile_writes_profile_on_success`
- `test_cmd_profile_handles_benchmark_failure_gracefully`

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

### Phase 5: Doctor Integration (Staleness Checks)

**Files**: `doctor_cli.py`

1. **Add `_check_profiles`** function:
   - Lists profile files in `profiles_dir`
   - Checks staleness for each
   - Adds warnings for stale profiles
2. **Add `--repair` action** for stale profiles: offer to remove profiles older than configurable max-age (default 90 days)

**Key design decisions**:
- Doctor profile check uses the same staleness detection logic as `check_staleness` in `profile_cache.py`
- Repair action lists stale profiles but does NOT auto-delete (MVP scope)
- Follows existing doctor check/repair pattern with `RepairAction` dataclass

**Tests**: `test_doctor_cli.py` (additions)
- `test_doctor_check_profiles_stale`
- `test_doctor_check_profiles_fresh`
- `test_doctor_check_profiles_no_profiles_dir`

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

### Phase 6: TUI Integration (Profile Trigger + Stale Warnings)

**Files**: `tui_app.py`

1. **Add input polling infrastructure**:
   - Add a daemon thread that polls `sys.stdin` for keypresses using non-blocking input
   - Store keypresses in a thread-safe queue (`queue.Queue`)
   - Main TUI loop checks queue and dispatches actions
   - This is consistent with the existing daemon thread pattern used in `ServerManager._stream_pipe()`

2. **Add `P` keybinding** in TUI event loop:
   - When a slot is focused, `P` triggers profiling
   - Confirmation prompt: "Profile slot <slot_id>? (y/N)"
   - Flavor selection: "(1) balanced (2) fast (3) quality"
   - Runs profiling for selected flavor in a background thread

3. **Add stale warning badge** in status panel:
   - Yellow warning: "⚠ profile stale — <reason>"
   - Reason: "driver changed" / "profile 45 days old" / "binary updated"

4. **Load profiles at startup** and check staleness:
   - At TUI init, load profiles for each configured slot
   - Check staleness, populate warning state

**Key design decisions**:
- Profile execution runs in a background thread to avoid blocking the TUI
- Profile progress displayed in the status panel (using existing `status_panel` mechanism)
- Stale warnings appear in the alerts panel (following existing risk/status panel pattern)
- **Never call `console.print()` while `Live` is active** — use layout updates

**Tests**: `test_tui_app.py` (additions)
- `test_profile_trigger_stale_cache_shows_warning` (mock cache, mock Live)
- `test_profile_trigger_fresh_cache_no_warning` (mock cache, mock Live)
- `test_profile_trigger_no_cache_no_warning` (mock cache, mock Live)
- `test_keypress_queue_dispatches_p_key` (mock stdin, mock queue)

**CI gates**: `ruff check`, `ruff format --check`, `pyright`, `pytest`

---

## File Structure Summary

```
src/
├── llama_manager/
│   ├── __init__.py              # [MODIFIED] Export new profile/benchmark symbols
│   ├── config.py                # [MODIFIED] Add profiles_dir, profile_staleness_days, server_binary_version
│   ├── config_builder.py        # [MODIFIED] Add warnings side-effect parameter, whitelist filter
│   ├── profile_cache.py         # [NEW] ProfileRecord, cache I/O, staleness detection
│   ├── benchmark.py             # [NEW] Benchmark command construction, output parsing
│   ├── gpu_stats.py             # [MODIFIED] Add get_gpu_identifier()
│   └── process_manager.py       # [UNCHANGED] Reuses _atomic_write_json, FILE_MODE_OWNER_ONLY
│
├── llama_cli/
│   ├── cli_parser.py            # [MODIFIED] Add profile subcommand parsing
│   ├── server_runner.py         # [MODIFIED] Wire profile subcommand handler
│   ├── profile_cli.py           # [NEW] Profile CLI subcommand handler
│   ├── doctor_cli.py            # [MODIFIED] Add profile staleness checks
│   └── tui_app.py               # [MODIFIED] Add profile trigger, stale warning display, input polling
│
└── tests/
    ├── test_config.py           # [ADD] Config field tests
    ├── test_us3_precedence.py   # [ADD] Profile merge precedence tests
    ├── test_cli_parser.py       # [ADD] profile subcommand tests
    ├── test_server_runner.py    # [ADD] profile handler tests
    ├── test_doctor_cli.py       # [ADD] profile staleness tests
    ├── test_tui_app.py          # [ADD] profile trigger data flow tests
    ├── test_profile_cache.py    # [NEW] ProfileRecord, cache I/O, staleness
    └── test_benchmark.py        # [NEW] Benchmark command construction, parsing
```

---

## Test Strategy

### Unit Tests (No GPU, No Subprocess)

| Test File              | What's Tested                                              | How It's Tested                                         |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `test_profile_cache.py`  | ProfileRecord serialization, cache read/write, staleness   | `tmp_path` fixture, synthetic JSON, mocked driver version |
| `test_benchmark.py`      | Command construction, output parsing, runner injection     | Direct function calls, mock runner callable             |
| `test_config.py`         | New Config fields, profiles_dir property                   | Direct instantiation, assert properties                 |
| `test_us3_precedence.py` | Profile config filtering, warning accumulation, precedence | Dict inputs, assert return values and warnings list     |
| `test_cli_parser.py`     | CLI parsing for profile subcommand                         | argparse call, assert namespace                         |
| `test_server_runner.py`  | Profile handler wiring, slot validation                    | Mock subprocess, mock lockfile                          |
| `test_doctor_cli.py`     | Profile staleness checks in doctor output                  | Mock cache files, assert warning output                 |
| `test_tui_app.py`        | Profile trigger data flow (not rendering)                  | Mock cache, mock Live, assert data passed               |

### Key Mocking Strategies

1. **Benchmark subprocess**: Mock `subprocess.run` in `profile_cli.py` tests to return synthetic stdout/stderr
2. **GPU stats**: Mock `get_gpu_identifier` and `compute_driver_version_hash` to return deterministic values
3. **Lockfile checks**: Mock `read_lock` to simulate "slot running" vs "slot free"
4. **File I/O**: Use `tmp_path` fixture for all profile cache operations
5. **Time-based staleness**: Use fixed Unix timestamps for `profiled_at`

---

## Risk/Mitigation Table

| Risk                                | Severity                | Mitigation                                                                                              |
| ----------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Return type change breaks callers**   | High                    | **AVOIDED** — use non-breaking `warnings` side-effect parameter instead of changing return type         |
| **Benchmark output format drift**       | Medium                  | Make `parse_benchmark_output` defensive — return `None` on parse failure, logged as warning                 |
| **Subprocess in llama_manager**         | Critical (architecture) | Keep `build_benchmark_cmd` and `run_benchmark` in pure library; subprocess execution in `profile_cli.py` only |
| **Race condition on profile writes**    | Medium                  | Use existing `_atomic_write_json` pattern with temp file + rename                                         |
| **Staleness detection false positives** | Low                     | Use SHA-256 hash (not raw string comparison) for driver version; document hash truncation               |
| **TUI blocking on profile execution**   | Medium                  | Run profiling in background thread; TUI polls for progress                                              |
| **Path traversal in filename**          | Critical (security)     | Sanitize all filename components with `_sanitize_filename_component` before path construction             |
| **Unvalidated benchmark binary path**   | Critical (security)     | Validate with `require_executable()` before subprocess execution                                        |
| **CI pyright errors on new types**      | Medium                  | Annotate all new functions; run `pyright` before committing                                             |
| **ruff lint errors on new code**        | Low                     | Run `ruff check --fix` before committing; follow existing style                                         |

---

## Estimated Effort

| Phase                 | Files                          | Estimated Effort | Complexity                  |
| --------------------- | ------------------------------ | ---------------- | --------------------------- |
| 1. Data Model & Cache | 2 new, 1 modified              | 2-3 hours        | Low                         |
| 2. Benchmark Module   | 1 new                          | 1.5-2 hours      | Low                         |
| 3. Config Builder     | 1 modified                     | 1-1.5 hours      | Medium (warnings parameter) |
| 4. CLI Subcommand     | 2 new, 2 modified              | 2-3 hours        | Medium                      |
| 5. Doctor Integration | 1 modified                     | 1-1.5 hours      | Low                         |
| 6. TUI Integration    | 1 modified                     | 3-4 hours        | High                        |
| **Tests**                 | 2 new test files, additions to 6 | 3-4 hours        | Medium                      |
| **TOTAL**                 | 6 new, 7 modified, 2 new tests   | **13-18 hours**      | **Medium**                      |

---

## CI Gate Checklist

Before any phase is considered complete, verify:

```bash
uv run ruff check .                    # Lint — zero errors
uv run ruff format --check .           # Format — all files formatted
uv run pyright                         # Type check — zero errors
uv run pytest                          # Tests — all pass
uv run pytest --cov --cov-report=term-missing  # Coverage — no unexpected gaps
```

**Pre-commit hooks** (if installed):
```bash
uv run pre-commit run --all-files
```

---

## Implementation Order (Dependency Graph)

```
Phase 1 (Data Model)
  ├── Phase 2 (Benchmark Module)
  │     └── Phase 4 (CLI Subcommand)
  │           └── Phase 5 (Doctor Integration)
  │                 └── Phase 6 (TUI Integration)
  └── Phase 3 (Config Builder)
        └── Phase 4 (CLI Subcommand)
              └── Phase 5 (Doctor Integration)
                    └── Phase 6 (TUI Integration)
```

**Recommended commit order**:
1. Phase 1: Data model + cache layer (foundation, no breaking changes)
2. Phase 2: Benchmark module (pure library, no dependencies)
3. Phase 3: Config builder extension (non-breaking, coordinate with Phase 4)
4. Phase 4: CLI subcommand (depends on 1, 2, 3)
5. Phase 5: Doctor integration (depends on 1, 3)
6. Phase 6: TUI integration (depends on 1-5)

---

## Post-MVP Notes (Documented, Not Implemented)

1. **`--strict-profiles` flag**: When stale profiles are detected, block launch. MVP only warns.
2. **Automatic cache eviction**: Based on age + access frequency + disk space. MVP: manual via doctor.
3. **vLLM profiling paths**: Reserved in schema but not implemented.
4. **Multi-machine profile sharing**: Profiles are machine-local in MVP.
5. **Background auto-profiling**: MVP requires explicit user trigger.
6. **TUI keybinding system**: MVP adds minimal input polling thread; future could use Rich's keybinding framework.
