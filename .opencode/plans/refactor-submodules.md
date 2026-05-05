# Refactor: Flat Modules → Subpackages in `llama_manager/`

> **Plan ID**: `refactor-submodules`
> **Date**: 2026-05-04
> **Status**: PROPOSED
> **Effort**: ~8–12 hours
> **Risk**: LOW (pure restructure, no behavior change)

---

## 1. Overview

Refactor six flat Python modules in `src/llama_manager/` into proper subpackages with focused submodules. The refactoring preserves **100% of the public API** — every symbol currently importable from `llama_manager` must remain importable after the refactor.

### What Changes

| Flat file (~lines) | New subpackage structure |
|---|---|
| `process_manager.py` (1738) | `orchestration/__init__.py` + `manager.py` + `lockfile.py` + `artifact.py` |
| `server.py` (914) | `validation/__init__.py` + `validators.py` + `commands/__init__.py` + `builder.py` |
| `smoke.py` (769) | `probe/__init__.py` + `smoke.py` + `provenance.py` |
| `benchmark.py` (463) | `benchmark/__init__.py` + `runner.py` + `parser.py` |
| `toolchain.py` (443) | `toolchain/__init__.py` + `detector.py` + `constants.py` |
| `reports.py` (403) | `reports/__init__.py` + `failure.py` + `rotation.py` + `redaction.py` |

### What Does NOT Change

- `slot_state.py`, `log_buffer.py`, `setup_venv.py`, `gpu_stats.py`, `risk_ack.py`, `slot_manager.py` — stay as single modules
- All existing `llama_manager` imports in `llama_cli/`, `tests/`, and `__init__.py`
- All public API surface — `from llama_manager import X` continues to work
- `llama_manager` remains a pure library (no argparse, no Rich, no subprocess at module level)

---

## 2. Current Import Graph

### 2.1. What imports FROM each flat file

#### `process_manager.py`
- **Internal** (llama_manager): `slot_manager.py`, `risk_ack.py`, `__init__.py`
- **Internal**: `server.py` (imports `build_server_cmd` from `process_manager.py`'s `start_servers` method)
- **Tests** (22+ files): `test_launch_flow.py`, `test_lock_integrity.py`, `test_foundation_contracts.py`, `test_dry_run_artifacts.py`, `test_dry_run_validation_contracts.py`, `test_launch_degraded_vs_blocked.py`, `runtime/process_manager_cases.py`, `tui/test_tui.py`, `support/runtime.py`, `smoke/test_smoke_lifecycle.py`, `config/config_cases.py`
- **Private symbols also tested**: `_verify_shutdown_ownership`, `_rotate_audit_log`, `_redact_sensitive`

#### `server.py`
- **Internal**: `process_manager.py` (imports `build_server_cmd`), `risk_ack.py` (imports `detect_risky_operations`), `config/builder.py` (imports `ServerConfig` — but this is `config/server.py`, not `server.py`)
- **CLI**: `llama_cli/commands/dry_run.py` (imports `detect_risky_operations`)
- **Tests** (8+ files): `test_server.py`, `test_dry_run_artifacts.py`, `test_dry_run_validation_contracts.py`, `test_dry_run_schema.py`, `test_dry_run_flag_bundles.py`, `test_validation_regression_contracts.py`, `system/test_validation_regression_contracts.py`, `system/test_sc006_performance.py`, `config/test_determinism.py`

#### `smoke.py`
- **Internal**: `__init__.py`
- **CLI**: `llama_cli/commands/smoke.py` (imports `probe_slot`, `SmokeProbeResult`, `SmokeCompositeReport`, `compute_overall_exit_code`, `resolve_provenance`, `ConsecutiveFailureCounter`)
- **Tests** (5+ files): `smoke/smoke_cases.py`, `test_smoke_lifecycle.py`, `test_smoke_json_schema.py`, `test_smoke_tui_cli_parity.py`, `support/factories.py`, `cli/smoke_cli_cases.py`
- **Private symbols also tested**: `_EXIT_CODE_MAP`, `_resolve_sha`, `_probe_models`, `_probe_chat`, `_tcp_connect`

#### `benchmark.py`
- **Internal**: `__init__.py`
- **Tests** (1 file): `system/test_benchmark.py`

#### `toolchain.py`
- **Internal**: `__init__.py`
- **CLI**: `llama_cli/commands/doctor.py`, `llama_cli/commands/setup.py`
- **Tests** (6+ files): `system/test_toolchain.py`, `system/test_setup_venv.py`, `system/test_profile_foundation.py`, `system/test_toolchain_diagnostics.py`, `cli/doctor_cli_cases.py`, `cli/test_setup_cli.py`

#### `reports.py`
- **Internal**: `__init__.py`
- **Tests** (2 files): `system/test_reports.py`, `support/factories.py`, `system/test_profile_foundation.py`

### 2.2. Internal cross-dependencies between files being refactored

```
process_manager.py
  → imports from: .config, .common.*, .gpu_stats, .log_buffer, .risk_ack (TYPE_CHECKING)
  → imports from: .server (build_server_cmd, lazy import in start_servers)
  → imports from: .slot_state (compute_slot_transition, lazy import in launch_orchestrate)

server.py
  → imports from: .common.security, .config
  → imports from: .config (VRamRecommendation, lazy import in assess_vram_risk)

smoke.py
  → imports from: .config, .metadata

benchmark.py
  → imports from: (stdlib only — self-contained)

toolchain.py
  → imports from: .build_pipeline (BuildBackend), .config (ErrorCode)

reports.py
  → imports from: .common.security, .config (Config, lazy import)
```

### 2.3. Dependency graph for new subpackages

```
orchestration/
  ├── __init__.py    → re-exports from manager, lockfile, artifact
  ├── manager.py     → ServerManager, launch_orchestrate, SlotRuntime, LaunchResult, LaunchOrchestrationResult
  ├── lockfile.py    → create_lock, read_lock, update_lock, release_lock, check_lockfile_integrity, LockMetadata
  └── artifact.py    → write_artifact, resolve_runtime_dir, ArtifactMetadata, DryRunArtifactPayload

validation/
  ├── __init__.py    → re-exports from validators, commands
  ├── validators.py  → validate_port, validate_threads, validate_ports, validate_slots,
  │                    validate_backend_eligibility, validate_server_config, detect_risky_operations,
  │                    require_model, require_executable
  └── commands/
      ├── __init__.py → re-exports from builder
      └── builder.py → build_server_cmd, build_dry_run_slot_payload, DryRunSlotPayload,
                       VllmEligibility, ValidationResults, sort_validation_errors,
                       DoctorCheckResult, DoctorReport, assess_vram_risk,
                       compute_machine_fingerprint, check_hardware_allowlist

probe/
  ├── __init__.py    → re-exports from smoke, provenance
  ├── smoke.py       → probe_slot, SmokeProbeResult, SmokeCompositeReport,
  │                    ConsecutiveFailureCounter, compute_overall_exit_code,
  │                    resolve_api_key, resolve_model_id_from_gguf, _EXIT_CODE_MAP,
  │                    _tcp_connect, _probe_models, _probe_chat, _handle_models_status,
  │                    _models_failure_result
  └── provenance.py  → resolve_provenance, ProvenanceRecord, _resolve_sha, _resolve_version

benchmark/
  ├── __init__.py    → re-exports from runner, parser
  ├── runner.py      → BenchmarkRunner (type alias), run_benchmark, build_benchmark_cmd, SubprocessResult
  └── parser.py      → parse_benchmark_output, BenchmarkResult, _extract_first_float,
                       _extract_number_from_patterns, _TOKENS_PATTERNS, _LATENCY_PATTERNS,
                       _VRAM_PATTERNS, _parse_markdown_table_metrics, _extract_tokens_per_second,
                       _extract_latency, _extract_vram, _find_column_indices,
                       _parse_data_row, _split_contiguous_blocks, _parse_table_block

toolchain/
  ├── __init__.py    → re-exports from detector, constants
  ├── detector.py    → detect_tool, get_toolchain_hints, parse_version, version_at_least,
  │                    detect_toolchain, ToolchainStatus, ToolchainErrorDetail,
  │                    _extract_version, _try_tool, _INTEL_ONEAPI_TOOLS, _INTEL_ONEAPI_BIN
  └── constants.py   → SYCL_REQUIRED_TOOLS, CUDA_REQUIRED_TOOLS, CMAKE_MINIMUM_VERSION,
                       GCC_HINT, MAKE_HINT, GIT_HINT, CMAKE_HINT, SYCL_HINT, CUDA_HINT, NVTOP_HINT,
                       ToolchainHint

reports/
  ├── __init__.py    → re-exports from failure, rotation, redaction
  ├── failure.py     → FailureReport, MutatingActionLogEntry, write_failure_report, log_mutating_action
  ├── rotation.py    → rotate_reports, _rotate_mutating_log
  └── redaction.py   → redact_sensitive
```

---

## 3. Detailed File-by-File Split

### 3.1. `process_manager.py` → `orchestration/`

#### 3.1.1. `orchestration/__init__.py`

```python
"""orchestration package — process management, lockfiles, and artifacts."""

from .artifact import (
    DryRunArtifactPayload,
    write_artifact,
)
from .lockfile import (
    LockMetadata,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
)
from .manager import (
    ArtifactMetadata,
    DryRunArtifactPayload,  # re-export for backward compat
    LaunchOrchestrationResult,
    LaunchResult,
    ProcessMetadata,
    ServerManager,
    SlotRuntime,
    ValidationException,
    launch_orchestrate,
)

__all__ = [
    # Lockfile operations
    "LockMetadata",
    "check_lockfile_integrity",
    "create_lock",
    "read_lock",
    "release_lock",
    "resolve_runtime_dir",
    "update_lock",
    # Artifact operations
    "write_artifact",
    "DryRunArtifactPayload",
    "ArtifactMetadata",
    # Server lifecycle
    "ServerManager",
    "SlotRuntime",
    "ProcessMetadata",
    "LaunchResult",
    "LaunchOrchestrationResult",
    "launch_orchestrate",
    # Exceptions
    "ValidationException",
]
```

**Imports**: `.config`, `.common.constants`, `.common.file_ops`, `.common.security`, `.gpu_stats`, `.log_buffer`, `.risk_ack` (TYPE_CHECKING), `.server` (lazy in `start_servers`)

#### 3.1.2. `orchestration/lockfile.py`

**Contents**:
- `LockMetadata` dataclass
- `resolve_runtime_dir()`
- `_get_lock_path()`
- `create_lock()`
- `read_lock()`
- `update_lock()`
- `release_lock()`
- `check_lockfile_integrity()`
- `_clear_lockfile()`
- `_build_indeterminate_owner_error()`
- `_verify_lock_owner()`
- Module constants: `LOCKFILE_CHECK_NAME`, `MAX_COLLISION_RETRIES`

**Imports**:
```python
import contextlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Final

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json, atomic_write_json
from ..config import Config, ErrorCode, ErrorDetail
import psutil
```

**Do NOT import**: `.manager` (circular — lockfile is used by manager)

#### 3.1.3. `orchestration/artifact.py`

**Contents**:
- `ArtifactMetadata` dataclass
- `DryRunArtifactPayload` TypedDict
- `write_artifact()`
- `_validate_artifact_fields()`
- `_redact_sensitive_in_dict()`
- Module constants: `ARTIFACT_CHECK_NAME`, `OWNER_ONLY_PERMISSIONS_FAILURE`, `PERMISSION_SUPPORT_HINT`, `PERMISSION_WRITABILITY_HINT`, `MAX_COLLISION_RETRIES`

**Imports**:
```python
import contextlib
import os
import stat
import time
from pathlib import Path
from typing import Any, Final, TypedDict

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json, atomic_write_json
from ..common.security import REDACTED_VALUE, SENSITIVE_KEY_NAME_PATTERN, SENSITIVE_WORD_PATTERN, is_sensitive_key
from ..config import ErrorCode
```

**Do NOT import**: `.lockfile` (artifact.py is independent)

#### 3.1.4. `orchestration/manager.py`

**Contents**:
- `ProcessMetadata` dataclass
- `ArtifactMetadata` dataclass
- `ValidationException` class
- `_make_validation_error()`
- `_lockfile_error()`
- `_artifact_error()`
- `LaunchResult` dataclass
- `LaunchOrchestrationResult` dataclass
- `SlotRuntime` dataclass
- `ServerManager` class
- `_rotate_audit_log()`
- `_AUDIT_LOG_MAX_BYTES`, `_AUDIT_LOG_MAX_FILES`
- `_redact_sensitive()`
- `_verify_shutdown_ownership()`
- `_append_audit_log()`
- `launch_orchestrate()`

**Imports**:
```python
import subprocess
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Final, TextIO, cast

import psutil

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json, atomic_write_json
from ..common.security import REDACTED_VALUE, SENSITIVE_KEY_NAME_PATTERN, SENSITIVE_WORD_PATTERN, is_sensitive_key
from ..config import Config, ErrorCode, ErrorDetail, ModelSlot, MultiValidationError, ServerConfig, SlotState, apply_profile_overrides
from ..gpu_stats import GPUStats
from ..log_buffer import LogBuffer

if TYPE_CHECKING:
    from ..risk_ack import RiskAckResult
```

**Internal imports**:
```python
from .lockfile import (
    LockMetadata,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
)
from .artifact import write_artifact, ArtifactMetadata, DryRunArtifactPayload
```

**Lazy imports** (inside functions):
- `from .server import build_server_cmd` — inside `ServerManager.start_servers()`
- `from ..risk_ack import RiskAckResult, evaluate_risks` — inside `launch_orchestrate()`
- `from ..slot_state import compute_slot_transition` — inside `launch_orchestrate()`

---

### 3.2. `server.py` → `validation/` + `commands/`

#### 3.2.1. `validation/__init__.py`

```python
"""validation package — input validation and server command building."""

from .builder import (
    DoctorCheckResult,
    DoctorReport,
    DryRunSlotPayload,
    ValidationResults,
    VllmEligibility,
    assess_vram_risk,
    build_dry_run_slot_payload,
    build_server_cmd,
    check_hardware_allowlist,
    compute_machine_fingerprint,
    sort_validation_errors,
)
from .validators import (
    detect_risky_operations,
    require_executable,
    require_model,
    validate_backend_eligibility,
    validate_ports,
    validate_server_config,
    validate_slots,
    validate_port,
    validate_threads,
)

__all__ = [
    # Validators
    "validate_port",
    "validate_threads",
    "validate_ports",
    "validate_slots",
    "validate_backend_eligibility",
    "validate_server_config",
    "require_model",
    "require_executable",
    "detect_risky_operations",
    # Command builder
    "build_server_cmd",
    "build_dry_run_slot_payload",
    # Payload types
    "DryRunSlotPayload",
    "VllmEligibility",
    "ValidationResults",
    # Doctor diagnostics
    "DoctorCheckResult",
    "DoctorReport",
    # VRAM assessment
    "assess_vram_risk",
    # Hardware fingerprinting
    "compute_machine_fingerprint",
    "check_hardware_allowlist",
    # Sorting
    "sort_validation_errors",
]
```

#### 3.2.2. `validation/validators.py`

**Contents**:
- `validate_port()`
- `validate_threads()`
- `validate_ports()`
- `require_model()`
- `require_executable()`
- `validate_backend_eligibility()`
- `validate_server_config()`
- `_validate_duplicate_slots()`
- `_validate_slot()`
- `_convert_results_to_errors()`
- `validate_slots()`
- `detect_risky_operations()`
- `sort_validation_errors()`

**Imports**:
```python
import os
from typing import Any

from ..config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
)
```

**Lazy imports** (inside functions):
- `from ..config import detect_duplicate_slots` — inside `_validate_duplicate_slots()`
- `from ..config import normalize_slot_id` — inside `_validate_slot()`

#### 3.2.3. `commands/__init__.py`

```python
"""commands subpackage — server command building and dry-run payloads."""

from .builder import (
    DoctorCheckResult,
    DoctorReport,
    DryRunSlotPayload,
    ValidationResults,
    VllmEligibility,
    assess_vram_risk,
    build_dry_run_slot_payload,
    build_server_cmd,
    check_hardware_allowlist,
    compute_machine_fingerprint,
    sort_validation_errors,
)

__all__ = [
    "build_server_cmd",
    "build_dry_run_slot_payload",
    "DryRunSlotPayload",
    "VllmEligibility",
    "ValidationResults",
    "DoctorCheckResult",
    "DoctorReport",
    "assess_vram_risk",
    "compute_machine_fingerprint",
    "check_hardware_allowlist",
    "sort_validation_errors",
]
```

#### 3.2.4. `commands/builder.py`

**Contents**:
- `DoctorCheckResult` dataclass
- `DoctorReport` dataclass
- `VllmEligibility` dataclass
- `ValidationResults` dataclass
- `DryRunSlotPayload` dataclass
- `build_server_cmd()`
- `sort_validation_errors()`
- `build_dry_run_slot_payload()`
- `_build_environment_redacted()`
- `_build_openai_flag_bundle()`
- `_build_hardware_notes()`
- `_parse_device_details()`
- `_get_lspci_output()`
- `_get_cpu_model()`
- `_get_os_name()`
- `compute_machine_fingerprint()`
- `check_hardware_allowlist()`
- `assess_vram_risk()`

**Imports**:
```python
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Final

from ..common.security import redact_env_value
from ..config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
    VRamRecommendation,
)
```

**Lazy imports** (inside functions):
- `from ..config import VRamRecommendation` — inside `assess_vram_risk()`

---

### 3.3. `smoke.py` → `probe/`

#### 3.3.1. `probe/__init__.py`

```python
"""probe package — smoke testing for llama.cpp inference servers."""

from .provenance import ProvenanceRecord, resolve_provenance
from .smoke import (
    ConsecutiveFailureCounter,
    SmokeCompositeReport,
    SmokeProbeResult,
    _EXIT_CODE_MAP,
    _models_failure_result,
    _probe_chat,
    _probe_models,
    _tcp_connect,
    _handle_models_status,
    compute_overall_exit_code,
    probe_slot,
    resolve_api_key,
    resolve_model_id_from_gguf,
)

__all__ = [
    # Public probe API
    "probe_slot",
    "resolve_api_key",
    "resolve_model_id_from_gguf",
    # Result types
    "SmokeProbeResult",
    "SmokeCompositeReport",
    "ConsecutiveFailureCounter",
    "ProvenanceRecord",
    # Computation
    "compute_overall_exit_code",
    # Provenance
    "resolve_provenance",
    # Internal (exported for tests)
    "_EXIT_CODE_MAP",
    "_tcp_connect",
    "_probe_models",
    "_probe_chat",
    "_handle_models_status",
    "_models_failure_result",
]
```

#### 3.3.2. `probe/smoke.py`

**Contents**:
- `_API_KEY_ENV_VAR` constant
- `resolve_api_key()`
- `resolve_model_id_from_gguf()`
- `_EXIT_CODE_MAP`
- `ProvenanceRecord` dataclass
- `SmokeProbeResult` dataclass
- `SmokeCompositeReport` dataclass
- `ConsecutiveFailureCounter` dataclass
- `_ensure_report_dir()`
- `probe_slot()`
- `_tcp_connect()`
- `_models_failure_result()`
- `_probe_models()`
- `_handle_models_status()`
- `_probe_chat()`

**Imports**:
```python
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import SmokeFailurePhase, SmokePhase, SmokeProbeConfiguration, SmokeProbeStatus
from ..metadata import extract_gguf_metadata, normalize_filename
```

#### 3.3.3. `probe/provenance.py`

**Contents**:
- `resolve_provenance()`
- `_resolve_sha()`
- `_resolve_version()`

**Imports**:
```python
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, run

from ..config import Config
from importlib.metadata import version as _importlib_version
```

---

### 3.4. `benchmark.py` → `benchmark/`

#### 3.4.1. `benchmark/__init__.py`

```python
"""benchmark package — benchmark result types and runner protocol."""

from .parser import (
    BenchmarkResult,
    _extract_first_float,
    _extract_latency,
    _extract_number_from_patterns,
    _extract_tokens_per_second,
    _extract_vram,
    _find_column_indices,
    _parse_data_row,
    _parse_markdown_table_metrics,
    _parse_table_block,
    _split_contiguous_blocks,
    parse_benchmark_output,
    _LATENCY_PATTERNS,
    _TOKENS_PATTERNS,
    _VRAM_PATTERNS,
)
from .runner import (
    SubprocessResult,
    build_benchmark_cmd,
    run_benchmark,
)

# Type alias re-export
BenchmarkRunner = run_benchmark.__annotations__.get("runner", "Callable[[list[str]], SubprocessResult]")
# Actually, BenchmarkRunner is defined as a type alias in runner.py
from .runner import BenchmarkRunner

__all__ = [
    "SubprocessResult",
    "BenchmarkResult",
    "BenchmarkRunner",
    "build_benchmark_cmd",
    "parse_benchmark_output",
    "run_benchmark",
    # Internal (exported for tests)
    "_extract_first_float",
    "_extract_number_from_patterns",
    "_TOKENS_PATTERNS",
    "_LATENCY_PATTERNS",
    "_VRAM_PATTERNS",
    "_parse_markdown_table_metrics",
    "_extract_tokens_per_second",
    "_extract_latency",
    "_extract_vram",
    "_find_column_indices",
    "_parse_data_row",
    "_split_contiguous_blocks",
    "_parse_table_block",
]
```

#### 3.4.2. `benchmark/runner.py`

**Contents**:
- `SubprocessResult` dataclass
- `BenchmarkResult` dataclass
- `BenchmarkRunner` type alias
- `build_benchmark_cmd()`
- `run_benchmark()`

**Imports**:
```python
import os
from dataclasses import dataclass
from typing import Callable
```

#### 3.4.3. `benchmark/parser.py`

**Contents**:
- `_extract_first_float()`
- `_extract_number_from_patterns()`
- `_find_column_indices()`
- `_parse_data_row()`
- `_split_contiguous_blocks()`
- `_parse_table_block()`
- `_parse_markdown_table_metrics()`
- `_TOKENS_PATTERNS`
- `_LATENCY_PATTERNS`
- `_VRAM_PATTERNS`
- `_extract_tokens_per_second()`
- `_extract_latency()`
- `_extract_vram()`
- `parse_benchmark_output()`

**Imports**:
```python
import math
import re
from typing import cast
```

---

### 3.5. `toolchain.py` → `toolchain/`

#### 3.5.1. `toolchain/__init__.py`

```python
"""toolchain package — build toolchain detection and status checking."""

from .constants import (
    CMAKE_HINT,
    CMAKE_MINIMUM_VERSION,
    CUDA_HINT,
    CUDA_REQUIRED_TOOLS,
    GCC_HINT,
    GIT_HINT,
    MAKE_HINT,
    NVTOP_HINT,
    SYCL_HINT,
    SYCL_REQUIRED_TOOLS,
    ToolchainHint,
)
from .detector import (
    ToolchainErrorDetail,
    ToolchainStatus,
    detect_tool,
    detect_toolchain,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)

__all__ = [
    # Status
    "ToolchainStatus",
    "ToolchainErrorDetail",
    # Hints
    "ToolchainHint",
    "GCC_HINT",
    "MAKE_HINT",
    "GIT_HINT",
    "CMAKE_HINT",
    "SYCL_HINT",
    "CUDA_HINT",
    "NVTOP_HINT",
    # Required tools
    "SYCL_REQUIRED_TOOLS",
    "CUDA_REQUIRED_TOOLS",
    "CMAKE_MINIMUM_VERSION",
    # Detection
    "detect_tool",
    "detect_toolchain",
    "get_toolchain_hints",
    # Version parsing
    "parse_version",
    "version_at_least",
]
```

#### 3.5.2. `toolchain/constants.py`

**Contents**:
- `SYCL_REQUIRED_TOOLS`
- `CUDA_REQUIRED_TOOLS`
- `CMAKE_MINIMUM_VERSION`
- `GCC_HINT`
- `MAKE_HINT`
- `GIT_HINT`
- `CMAKE_HINT`
- `SYCL_HINT`
- `CUDA_HINT`
- `NVTOP_HINT`
- `ToolchainHint` dataclass

**Imports**:
```python
from dataclasses import dataclass, field
```

#### 3.5.3. `toolchain/detector.py`

**Contents**:
- `_INTEL_ONEAPI_TOOLS`
- `_INTEL_ONEAPI_BIN`
- `_extract_version()`
- `_try_tool()`
- `detect_tool()`
- `get_toolchain_hints()`
- `parse_version()`
- `version_at_least()`
- `detect_toolchain()`
- `ToolchainStatus` dataclass
- `ToolchainErrorDetail` dataclass

**Imports**:
```python
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import (
    CUDA_HINT,
    CUDA_REQUIRED_TOOLS,
    CMAKE_HINT,
    GCC_HINT,
    GIT_HINT,
    MAKE_HINT,
    NVTOP_HINT,
    SYCL_HINT,
    SYCL_REQUIRED_TOOLS,
    ToolchainHint,
)

if TYPE_CHECKING:
    from ..config import ErrorCode
```

**Lazy imports** (inside functions):
- `from ..config import ErrorCode` — inside `get_toolchain_hints()`

---

### 3.6. `reports.py` → `reports/`

#### 3.6.1. `reports/__init__.py`

```python
"""reports package — build failure reporting and log rotation."""

from .failure import (
    FailureReport,
    MutatingActionLogEntry,
    log_mutating_action,
    write_failure_report,
)
from .redaction import redact_sensitive
from .rotation import rotate_reports

__all__ = [
    "redact_sensitive",
    "FailureReport",
    "MutatingActionLogEntry",
    "write_failure_report",
    "log_mutating_action",
    "rotate_reports",
]
```

#### 3.6.2. `reports/redaction.py`

**Contents**:
- `redact_sensitive()`

**Imports**:
```python
import re

from ..common.security import REDACTED_VALUE
```

#### 3.6.3. `reports/failure.py`

**Contents**:
- `FailureReport` dataclass
- `MutatingActionLogEntry` dataclass
- `write_failure_report()`
- `log_mutating_action()`

**Imports**:
```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redaction import redact_sensitive
from .rotation import _rotate_mutating_log
```

**Lazy imports** (inside functions):
- `from ..config import Config` — inside `write_failure_report()` and `log_mutating_action()`

#### 3.6.4. `reports/rotation.py`

**Contents**:
- `rotate_reports()`
- `_rotate_mutating_log()`

**Imports**:
```python
import contextlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
```

**Lazy imports** (inside function):
- `from ..config import Config` — inside `rotate_reports()`

---

## 4. Migration Strategy

### Phase 1: Create new subpackages, move code

**Step 1.1**: Create directory structure

```
src/llama_manager/
├── orchestration/
│   ├── __init__.py
│   ├── manager.py
│   ├── lockfile.py
│   └── artifact.py
├── validation/
│   ├── __init__.py
│   ├── validators.py
│   └── commands/
│       ├── __init__.py
│       └── builder.py
├── probe/
│   ├── __init__.py
│   ├── smoke.py
│   └── provenance.py
├── benchmark/
│   ├── __init__.py
│   ├── runner.py
│   └── parser.py
├── toolchain/
│   ├── __init__.py
│   ├── detector.py
│   └── constants.py
└── reports/
    ├── __init__.py
    ├── failure.py
    ├── rotation.py
    └── redaction.py
```

**Step 1.2**: Move code into new files

For each old file:
1. Create the new subpackage directory with `__init__.py`
2. Split the old file's content into the new subfiles
3. Update all relative imports within the new files (change `from .X import Y` to `from ..X import Y` where X is now in a sibling package)

**Step 1.3**: Update internal cross-references

The following internal imports need updating:

| From | Old import | New import |
|---|---|---|
| `orchestration/manager.py` | `from .server import build_server_cmd` | `from ..validation.commands import build_server_cmd` |
| `orchestration/manager.py` | `from .risk_ack import ...` | `from ..risk_ack import ...` |
| `orchestration/manager.py` | `from .slot_state import ...` | `from ..slot_state import ...` |
| `orchestration/manager.py` | `from .common.* import ...` | `from ..common.* import ...` |
| `orchestration/manager.py` | `from .config import ...` | `from ..config import ...` |
| `orchestration/manager.py` | `from .gpu_stats import ...` | `from ..gpu_stats import ...` |
| `orchestration/manager.py` | `from .log_buffer import ...` | `from ..log_buffer import ...` |
| `validation/validators.py` | `from .config import ...` | `from ..config import ...` |
| `validation/commands/builder.py` | `from .config import ...` | `from ..config import ...` |
| `validation/commands/builder.py` | `from .common.security import ...` | `from ..common.security import ...` |
| `probe/smoke.py` | `from .config import ...` | `from ..config import ...` |
| `probe/smoke.py` | `from .metadata import ...` | `from ..metadata import ...` |
| `probe/provenance.py` | `from .config import ...` | `from ..config import ...` |
| `toolchain/detector.py` | `from .build_pipeline import ...` | `from ..build_pipeline import ...` |
| `toolchain/detector.py` | `from .config import ...` | `from ..config import ...` |
| `reports/failure.py` | `from .common.security import ...` | `from ..common.security import ...` |
| `reports/failure.py` | `from .config import ...` | `from ..config import ...` |
| `reports/rotation.py` | `from .config import ...` | `from ..config import ...` |

### Phase 2: Update `__init__.py` to import from new locations

Update `src/llama_manager/__init__.py` to import from the new package locations. The re-export interface must remain **identical** — same symbols, same names.

Key changes in `__init__.py`:

```python
# OLD:
from .process_manager import (
    ArtifactMetadata, DryRunArtifactPayload, LaunchOrchestrationResult,
    LaunchResult, LockMetadata, ServerManager, ValidationException,
    create_lock, launch_orchestrate, read_lock, release_lock,
    resolve_runtime_dir, update_lock, write_artifact,
)

# NEW:
from .orchestration import (
    ArtifactMetadata, DryRunArtifactPayload, LaunchOrchestrationResult,
    LaunchResult, LockMetadata, ServerManager, ValidationException,
    create_lock, launch_orchestrate, read_lock, release_lock,
    resolve_runtime_dir, update_lock, write_artifact,
)
```

Similar updates for all six packages.

### Phase 3: Update external consumers

#### 3.3.1. `llama_cli/` imports

| File | Old import | New import |
|---|---|---|
| `llama_cli/commands/dry_run.py` | `from llama_manager.server import detect_risky_operations` | `from llama_manager.server import detect_risky_operations` (no change — re-exported from `llama_manager`) |
| `llama_cli/commands/smoke.py` | `from llama_manager.smoke import ...` | `from llama_manager.smoke import ...` (no change — re-exported) |
| `llama_cli/commands/doctor.py` | `from llama_manager.toolchain import ...` | `from llama_manager.toolchain import ...` (no change — re-exported) |
| `llama_cli/commands/setup.py` | `from llama_manager.toolchain import ...` | `from llama_manager.toolchain import ...` (no change — re-exported) |

**No changes needed in `llama_cli/`** — all imports go through `llama_manager` top-level, which re-exports everything.

#### 3.3.2. Test files

All test files import via `from llama_manager.X import Y` or `from llama_manager import Y`. Since `llama_manager.__init__.py` re-exports everything, **no test file changes are needed**.

However, tests that import private symbols directly (e.g., `from llama_manager.smoke import _tcp_connect`) will need updating:

| Old import path | New import path |
|---|---|
| `from llama_manager.smoke import _tcp_connect` | `from llama_manager.probe.smoke import _tcp_connect` |
| `from llama_manager.smoke import _probe_models` | `from llama_manager.probe.smoke import _probe_models` |
| `from llama_manager.smoke import _probe_chat` | `from llama_manager.probe.smoke import _probe_chat` |
| `from llama_manager.smoke import _resolve_sha` | `from llama_manager.probe.provenance import _resolve_sha` |
| `from llama_manager.smoke import _EXIT_CODE_MAP` | `from llama_manager.probe.smoke import _EXIT_CODE_MAP` |
| `from llama_manager.process_manager import _rotate_audit_log` | `from llama_manager.orchestration.manager import _rotate_audit_log` |
| `from llama_manager.process_manager import _redact_sensitive` | `from llama_manager.orchestration.manager import _redact_sensitive` |
| `from llama_manager.process_manager import _verify_shutdown_ownership` | `from llama_manager.orchestration.manager import _verify_shutdown_ownership` |
| `from llama_manager.process_manager import SlotRuntime` | `from llama_manager.orchestration.manager import SlotRuntime` |

---

## 5. `__init__.py` Update

The `src/llama_manager/__init__.py` file needs its import statements updated. The `__all__` list remains **unchanged** — every symbol currently exported must still be exported.

### Import block changes

```python
# REPLACED:
from .benchmark import (
    BenchmarkResult, BenchmarkRunner, SubprocessResult,
    build_benchmark_cmd, parse_benchmark_output, run_benchmark,
)

# WITH:
from .benchmark import (
    BenchmarkResult, BenchmarkRunner, SubprocessResult,
    build_benchmark_cmd, parse_benchmark_output, run_benchmark,
)
# (no change — benchmark is now a package, but __init__.py re-exports the same symbols)

# REPLACED:
from .process_manager import (
    ArtifactMetadata, DryRunArtifactPayload, LaunchOrchestrationResult,
    LaunchResult, LockMetadata, ServerManager, ValidationException,
    create_lock, launch_orchestrate, read_lock, release_lock,
    resolve_runtime_dir, update_lock, write_artifact,
)

# WITH:
from .orchestration import (
    ArtifactMetadata, DryRunArtifactPayload, LaunchOrchestrationResult,
    LaunchResult, LockMetadata, ServerManager, ValidationException,
    create_lock, launch_orchestrate, read_lock, release_lock,
    resolve_runtime_dir, update_lock, write_artifact,
)

# REPLACED:
from .reports import (
    FailureReport, MutatingActionLogEntry, redact_sensitive,
    rotate_reports, write_failure_report,
)

# WITH:
from .reports import (
    FailureReport, MutatingActionLogEntry, redact_sensitive,
    rotate_reports, write_failure_report,
)
# (no change — reports is now a package, but __init__.py re-exports the same symbols)

# REPLACED:
from .server import (
    DryRunSlotPayload, ValidationResults, VllmEligibility,
    build_dry_run_slot_payload, build_server_cmd,
    require_executable, require_model, validate_backend_eligibility,
    validate_port, validate_ports, validate_server_config,
    validate_slots, validate_threads,
)

# WITH:
from .validation import (
    DryRunSlotPayload, ValidationResults, VllmEligibility,
    build_dry_run_slot_payload, build_server_cmd,
    require_executable, require_model, validate_backend_eligibility,
    validate_port, validate_ports, validate_server_config,
    validate_slots, validate_threads,
)

# REPLACED:
from .smoke import (
    ConsecutiveFailureCounter, ProvenanceRecord, SmokeCompositeReport,
    SmokeProbeResult, compute_overall_exit_code, probe_slot, resolve_provenance,
)

# WITH:
from .probe import (
    ConsecutiveFailureCounter, ProvenanceRecord, SmokeCompositeReport,
    SmokeProbeResult, compute_overall_exit_code, probe_slot, resolve_provenance,
)

# REPLACED:
from .toolchain import (
    CMAKE_HINT, CMAKE_MINIMUM_VERSION, CUDA_HINT, CUDA_REQUIRED_TOOLS,
    GCC_HINT, GIT_HINT, MAKE_HINT, NVTOP_HINT, SYCL_HINT, SYCL_REQUIRED_TOOLS,
    ToolchainErrorDetail, ToolchainHint, ToolchainStatus,
    detect_tool, get_toolchain_hints, parse_version, version_at_least,
)

# WITH:
from .toolchain import (
    CMAKE_HINT, CMAKE_MINIMUM_VERSION, CUDA_HINT, CUDA_REQUIRED_TOOLS,
    GCC_HINT, GIT_HINT, MAKE_HINT, NVTOP_HINT, SYCL_HINT, SYCL_REQUIRED_TOOLS,
    ToolchainErrorDetail, ToolchainHint, ToolchainStatus,
    detect_tool, get_toolchain_hints, parse_version, version_at_least,
)
# (no change — toolchain is now a package, but __init__.py re-exports the same symbols)
```

---

## 6. Files to Delete (After Verification)

**Only after all CI checks pass and the refactor is verified:**

| Old file | Reason |
|---|---|
| `src/llama_manager/process_manager.py` | Replaced by `orchestration/` |
| `src/llama_manager/server.py` | Replaced by `validation/` + `commands/` |
| `src/llama_manager/smoke.py` | Replaced by `probe/` |
| `src/llama_manager/benchmark.py` | Replaced by `benchmark/` |
| `src/llama_manager/toolchain.py` | Replaced by `toolchain/` |
| `src/llama_manager/reports.py` | Replaced by `reports/` |

---

## 7. Validation Criteria

### 7.1. Pre-commit gates (MUST pass)

```bash
uv run pre-commit run --all-files
```

This runs:
1. `ruff check .` — lint
2. `ruff format --check .` — formatting
3. `pyright` — type checking

### 7.2. Test gates (MUST pass)

```bash
uv run pytest
```

### 7.3. API compatibility verification

```python
# These MUST all continue to work after the refactor:
from llama_manager import (
    ServerManager, build_server_cmd, probe_slot,
    validate_port, validate_slots, BenchmarkResult,
    ToolchainStatus, FailureReport, redact_sensitive,
    LockMetadata, create_lock, read_lock, release_lock,
    DryRunSlotPayload, SmokeProbeResult, ProvenanceRecord,
    # ... all symbols in __all__
)

# These MUST also continue to work (direct module imports):
from llama_manager.process_manager import ServerManager  # DEPRECATED but still works
from llama_manager.server import build_server_cmd         # DEPRECATED but still works
from llama_manager.smoke import probe_slot                 # DEPRECATED but still works
from llama_manager.benchmark import BenchmarkResult        # DEPRECATED but still works
from llama_manager.toolchain import ToolchainStatus        # DEPRECATED but still works
from llama_manager.reports import FailureReport            # DEPRECATED but still works
```

### 7.4. Private symbol accessibility (for tests)

```python
# These MUST still be importable from their new locations:
from llama_manager.probe.smoke import _tcp_connect
from llama_manager.probe.smoke import _probe_models
from llama_manager.probe.smoke import _probe_chat
from llama_manager.probe.provenance import _resolve_sha
from llama_manager.probe.smoke import _EXIT_CODE_MAP
from llama_manager.orchestration.manager import _rotate_audit_log
from llama_manager.orchestration.manager import _redact_sensitive
from llama_manager.orchestration.manager import _verify_shutdown_ownership
from llama_manager.orchestration.manager import SlotRuntime
```

---

## 8. Implementation Order

Execute in this order to minimize merge conflicts and allow incremental verification:

### Step 1: `benchmark/` (easiest — self-contained, fewest imports)
- No internal cross-dependencies
- Only one consumer in tests
- Can verify independently

### Step 2: `toolchain/` (simple — two subfiles, one cross-dependency)
- `detector.py` imports from `constants.py` and `build_pipeline`
- `constants.py` is self-contained

### Step 3: `reports/` (simple — three subfiles, minimal cross-deps)
- `failure.py` imports from `redaction.py` and `rotation.py`
- `rotation.py` is self-contained
- `redaction.py` imports from `common.security`

### Step 4: `probe/` (moderate — smoke testing module)
- `smoke.py` imports from `config` and `metadata`
- `provenance.py` imports from `config`
- Many test consumers

### Step 5: `validation/` + `commands/` (complex — many consumers)
- `validators.py` imports from `config`
- `commands/builder.py` imports from `config` and `common.security`
- Many test consumers and one CLI consumer

### Step 6: `orchestration/` (most complex — many internal cross-deps)
- `manager.py` imports from `lockfile.py`, `artifact.py`, `server.py`, `risk_ack.py`, `slot_state.py`
- `lockfile.py` imports from `common.*` and `config`
- `artifact.py` imports from `common.*` and `config`
- Many test consumers

### Step 7: Update `llama_manager/__init__.py`
- After all subpackages are created and verified

### Step 8: Run full CI gate
- `uv run pre-commit run --all-files`
- `uv run pytest`

### Step 9: Delete old flat files
- Only after CI passes

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Circular import between new subpackages | ImportError at runtime | Keep `orchestration/` imports from `validation/` lazy (inside functions); `validation/` has no imports from `orchestration/` |
| Missing re-export in `__init__.py` | `from llama_manager import X` fails | Compare `__all__` before and after; use `pytest` to import every symbol |
| Test imports of private symbols break | Test failures | Update test imports to new paths; keep private symbols importable from their new locations |
| `pyright` type errors from changed import paths | Type check failure | Run `pyright` after each subpackage migration |
| `ruff` import sorting errors | Lint failure | Run `ruff check --fix` after each change |
| `subprocess` at module level in new files | Violates pure library rule | Verify no module-level `import subprocess` in any new file; subprocess must be inside functions |
| `DoctorReport` and `DoctorCheckResult` in `commands/builder.py` | Tests expect them in `server` module | Re-export from `validation.__init__.py` and `llama_manager.__init__.py` |

---

## 10. New Files Summary

| New file | Lines (est.) | Purpose |
|---|---|---|
| `orchestration/__init__.py` | ~40 | Re-exports from all submodules |
| `orchestration/manager.py` | ~500 | ServerManager, launch_orchestrate, SlotRuntime, LaunchResult, LaunchOrchestrationResult, ProcessMetadata, ValidationException |
| `orchestration/lockfile.py` | ~300 | Lockfile operations (create, read, update, release, integrity check) |
| `orchestration/artifact.py` | ~150 | Artifact I/O (write_artifact, resolve_runtime_dir, ArtifactMetadata, DryRunArtifactPayload) |
| `validation/__init__.py` | ~40 | Re-exports from all submodules |
| `validation/validators.py` | ~200 | All validation functions |
| `validation/commands/__init__.py` | ~20 | Re-exports from builder |
| `validation/commands/builder.py` | ~400 | build_server_cmd, build_dry_run_slot_payload, DoctorReport, VRAM assessment, hardware fingerprinting |
| `probe/__init__.py` | ~40 | Re-exports from all submodules |
| `probe/smoke.py` | ~450 | probe_slot, SmokeProbeResult, SmokeCompositeReport, ConsecutiveFailureCounter, probe internals |
| `probe/provenance.py` | ~80 | resolve_provenance, ProvenanceRecord, _resolve_sha, _resolve_version |
| `benchmark/__init__.py` | ~40 | Re-exports from all submodules |
| `benchmark/runner.py` | ~100 | SubprocessResult, BenchmarkRunner type alias, build_benchmark_cmd, run_benchmark |
| `benchmark/parser.py` | ~250 | parse_benchmark_output, BenchmarkResult, all parsing helpers |
| `toolchain/__init__.py` | ~40 | Re-exports from all submodules |
| `toolchain/constants.py` | ~100 | ToolchainHint, all HINT constants, required tool tuples, CMAKE_MINIMUM_VERSION |
| `toolchain/detector.py` | ~200 | detect_tool, detect_toolchain, get_toolchain_hints, parse_version, version_at_least, ToolchainStatus, ToolchainErrorDetail |
| `reports/__init__.py` | ~20 | Re-exports from all submodules |
| `reports/failure.py` | ~180 | FailureReport, MutatingActionLogEntry, write_failure_report, log_mutating_action |
| `reports/rotation.py` | ~50 | rotate_reports, _rotate_mutating_log |
| `reports/redaction.py` | ~40 | redact_sensitive |

**Total new files**: 20
**Total old files to delete**: 6
**Net change**: +14 files (but much better organized)

---

## 11. Backward Compatibility Notes

The refactor **must not break** any of these import patterns:

1. **Top-level imports** (primary usage):
   ```python
   from llama_manager import ServerManager, build_server_cmd, probe_slot
   ```

2. **Direct module imports** (secondary usage, deprecated but must work):
   ```python
   from llama_manager.process_manager import ServerManager
   from llama_manager.server import build_server_cmd
   from llama_manager.smoke import probe_slot
   from llama_manager.benchmark import BenchmarkResult
   from llama_manager.toolchain import ToolchainStatus
   from llama_manager.reports import FailureReport
   ```

3. **Private symbol imports in tests**:
   ```python
   from llama_manager.probe.smoke import _tcp_connect
   from llama_manager.orchestration.manager import _redact_sensitive
   ```

If direct module imports (pattern 2) break, the `__init__.py` won't help — the tests and CLI that use `from llama_manager.process_manager import ...` will fail. **Solution**: Add compatibility re-exports at the end of each new `__init__.py` by creating thin shim modules:

```python
# src/llama_manager/process_manager.py (SHIM — delete after verification)
"""Backward compatibility shim. Import from llama_manager.orchestration instead."""
from .orchestration import *  # noqa: F401, F403
from .orchestration.manager import (  # noqa: F401
    ServerManager, SlotRuntime, ProcessMetadata,
    LaunchResult, LaunchOrchestrationResult, launch_orchestrate,
    ValidationException, ArtifactMetadata, DryRunArtifactPayload,
)
from .orchestration.lockfile import (  # noqa: F401
    LockMetadata, create_lock, read_lock, update_lock, release_lock,
    check_lockfile_integrity, resolve_runtime_dir,
)
from .orchestration.artifact import write_artifact  # noqa: F401
```

Similarly for `server.py`, `smoke.py`, `benchmark.py`, `toolchain.py`, `reports.py`.

These shim files are thin (~10 lines each) and exist solely to maintain backward compatibility for direct module imports. They can be deleted after verification.

---

## 12. Final Directory Structure

After the refactor:

```
src/llama_manager/
├── __init__.py                    # Updated imports, same __all__
├── log_buffer.py                  # Unchanged
├── gpu_stats.py                   # Unchanged
├── slot_state.py                  # Unchanged
├── slot_manager.py                # Unchanged
├── risk_ack.py                    # Unchanged
├── setup_venv.py                  # Unchanged
│
├── config/                        # Existing — unchanged
│   ├── __init__.py
│   ├── builder.py
│   ├── defaults.py
│   ├── enums.py
│   ├── errors.py
│   ├── profiles.py
│   ├── profile_cache.py
│   └── server.py
│
├── common/                        # Existing — unchanged
│   ├── __init__.py
│   ├── constants.py
│   ├── file_ops.py
│   └── security.py
│
├── metadata/                      # Existing — unchanged
│   ├── __init__.py
│   ├── _binary.py
│   ├── _reader.py
│   └── _types.py
│
├── build_pipeline/                # Existing — unchanged
│   ├── __init__.py
│   ├── lock.py
│   ├── models.py
│   ├── orchestration.py
│   ├── pipeline.py
│   └── utils.py
│
├── orchestration/                 # NEW (from process_manager.py)
│   ├── __init__.py
│   ├── manager.py
│   ├── lockfile.py
│   └── artifact.py
│
├── validation/                    # NEW (from server.py)
│   ├── __init__.py
│   ├── validators.py
│   └── commands/
│       ├── __init__.py
│       └── builder.py
│
├── probe/                         # NEW (from smoke.py)
│   ├── __init__.py
│   ├── smoke.py
│   └── provenance.py
│
├── benchmark/                     # NEW (from benchmark.py)
│   ├── __init__.py
│   ├── runner.py
│   └── parser.py
│
├── toolchain/                     # NEW (from toolchain.py)
│   ├── __init__.py
│   ├── detector.py
│   └── constants.py
│
├── reports/                       # NEW (from reports.py)
│   ├── __init__.py
│   ├── failure.py
│   ├── rotation.py
│   └── redaction.py
│
├── process_manager.py             # SHIM — delete after verification
├── server.py                      # SHIM — delete after verification
├── smoke.py                       # SHIM — delete after verification
├── benchmark.py                   # SHIM — delete after verification
├── toolchain.py                   # SHIM — delete after verification
└── reports.py                     # SHIM — delete after verification
```
