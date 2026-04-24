# Data Model — M4: Operational Hardening and Smoke Verification

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)  
**Feature Branch**: `003-m4-op-hardening`

---

## 1. Entity Definitions

### 1.1 Slot

A GPU slot owning bind host, port, backend flavor, environment overlays, and at most one active model process. Identified by a slot ID (e.g. `arc_b580`, `rtx3090`).

| Field | Type | Validation |
| --- | --- | --- |
| `slot_id` | `str` | Normalized via `normalize_slot_id()` (lowercase, `a-z0-9_-` only) |
| `bind_host` | `str` | Default `"127.0.0.1"` |
| `port` | `int` | `1–65535` |
| `backend_flavor` | `BackendFlavor` | `cuda` or `sycl` |
| `env_overlays` | `dict[str, str]` | Free-form environment overrides |
| `model_path` | `str` | Absolute or relative path to GGUF file |
| `lock_path` | `Path` | Computed at runtime from resolved runtime dir + slot ID; not persisted on the Slot entity |
| `state` | `SlotState` | One of the six states defined in §1.1.1 |
| `pid` | `int \| None` | Set on launch, cleared on shutdown |
| `server_config` | `ServerConfig` | Per-instance launch parameters (existing dataclass) |

#### 1.1.1 Operational States

```python
class SlotState(StrEnum):
    """Six operational states for a GPU slot."""

    IDLE = "idle"
    LAUNCHING = "launching"
    RUNNING = "running"
    DEGRADED = "degraded"
    CRASHED = "crashed"
    OFFLINE = "offline"
```

**State transition rules:**

| From | To | Trigger | Guard |
| --- | --- | --- | --- |
| `idle` | `launching` | Launch initiated | Lock not held |
| `launching` | `running` | Process accepts TCP connection | Within `tui_launch_timeout_s` |
| `launching` | `crashed` | Process exits before accepting | N/A |
| `running` | `degraded` | Probe latency > threshold OR GPU errors OR repeated log errors | `probe_latency_threshold_s` exceeded |
| `running` | `crashed` | Process exits unexpectedly | Non-zero exit code or signal |
| `running` | `offline` | Health probe (TCP listen check) fails repeatedly | N/A |
| `running` | `idle` | Shutdown completes successfully | SIGTERM + wait |
| `degraded` | `running` | Health recovers (probe latency < threshold) | N/A |
| `degraded` | `crashed` | Process exits | N/A |
| `degraded` | `offline` | Health probe fails while degraded | N/A |
| `degraded` | `idle` | User-initiated reset | N/A |
| `crashed` | `offline` | Post-exit probe confirms unresponsiveness | N/A |
| `crashed` | `idle` | Crash cleanup completes | N/A |
| `offline` | `idle` | User-initiated reset or cleanup | N/A |
| `offline` | `launching` | User initiates relaunch | N/A |

**No-op transitions** (state remains unchanged): any transition not listed above.

---

### 1.2 Smoke Probe Result

Per-slot outcome from a smoke probe session.

| Field | Type | Validation |
| --- | --- | --- |
| `slot_id` | `str` | Non-empty, normalized |
| `status` | `SmokeProbeStatus` | One of the values in §1.2.1 |
| `phase_reached` | `SmokePhase` | The last phase that was attempted (§1.2.2) |
| `failure_phase` | `SmokeFailurePhase \| None` | The phase where failure occurred; `null` if `status == "pass"` |
| `model_id` | `str \| None` | Resolved from GGUF `general.name`, `/v1/models`, or fallback |
| `latency_ms` | `int \| None` | Single observed value for the probe session; `null` if not measured |
| `provenance` | `ProvenanceRecord` | Binary SHA + tool version (§1.6) |

#### 1.2.1 Smoke Probe Status

```python
class SmokeProbeStatus(StrEnum):
    """Outcome of a single slot smoke probe."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    MODEL_NOT_FOUND = "model_not_found"
    AUTH_FAILURE = "auth_failure"
```

#### 1.2.2 Smoke Probe Phase

```python
class SmokePhase(StrEnum):
    """Ordered phases of a smoke probe session."""

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"
    COMPLETE = "complete"


class SmokeFailurePhase(StrEnum):
    """Phases where a failure can occur (excludes 'complete').

    A failure can only occur during listen, models, or chat phases.
    'complete' signifies all phases passed — it is never a failure phase.
    """

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"
```

Phase progression: `listen` → `models` (optional) → `chat` → `complete`. Each phase attempted exactly once; failure is immediate and non-recoverable for that probe session.

---

### 1.3 GGUF Metadata Record

Extracted header fields from a GGUF file without loading weights.

| Field | Type | Validation |
| --- | --- | --- |
| `raw_path` | `str` | Absolute path to the GGUF file |
| `normalized_stem` | `str` | NFKC-normalized, whitespace-replaced filename stem |
| `general_name` | `str \| None` | `general.name` from GGUF header; `None` if absent |
| `architecture` | `str \| None` | `general.architecture` |
| `tokenizer_type` | `str \| None` | `tokenizer.type` |
| `embedding_length` | `int \| None` | `llama.embedding_length` |
| `block_count` | `int \| None` | `llama.block_count` |
| `context_length` | `int \| None` | `llama.context_length` |
| `attention_head_count` | `int \| None` | `llama.attention.head_count` |
| `attention_head_count_kv` | `int \| None` | `llama.attention.head_count_kv` |
| `parse_timestamp` | `float` | `time.time()` at extraction start |
| `prefix_cap_bytes` | `int` | Max bytes read from file start (default 32 MiB) |
| `parse_timeout_s` | `float` | Wall-clock timeout (default 5 s) |
| `format_version` | `int \| None` | GGUF version from header; `None` if unreadable |

**Error variants** (returned when extraction fails):

| Error | Meaning |
| --- | --- |
| `CORRUPT_FILE` | Bad magic bytes or invalid header structure |
| `PARSE_TIMEOUT` | Wall-clock timeout exceeded |
| `UNSUPPORTED_VERSION` | GGUF v4+ detected (MVP only supports v3) |
| `READ_ERROR` | OS-level read failure |

---

### 1.4 Hardware Allowlist Entry

Machine fingerprint + PCI device IDs, persisted to `~/.config/llm-runner/hardware-allowlist.json`.

| Field | Type | Validation |
| --- | --- | --- |
| `fingerprint` | `str` | SHA256 of normalized `lspci` + `sycl-ls` output |
| `pci_device_ids` | `list[str]` | PCI addresses (e.g. `"0000:01:00.0"`) |
| `created_at` | `float` | `time.time()` at creation |
| `invalidated` | `bool` | `True` when fingerprint or PCI set changes |

**Invalidation rule**: When the machine fingerprint or PCI device set changes, the entry is marked `invalidated=True` and a new warning is triggered.

---

### 1.5 VRAM Risk Assessment

Best-effort free memory analysis with heuristic model footprint.

| Field | Type | Validation |
| --- | --- | --- |
| `free_vram_bytes` | `int \| None` | From GPU telemetry (`nvidia-smi` or Intel equivalent); `None` if unavailable |
| `model_file_size_bytes` | `int` | On-disk file size of the GGUF model |
| `estimated_footprint_bytes` | `int` | Heuristic: `model_file_size_bytes × 1.2` |
| `safety_margin_factor` | `float` | Default `0.85` |
| `recommendation` | `VRamRecommendation` | `proceed`, `warn`, or `confirm-required` (§1.5.1) |
| `warning_message` | `str \| None` | Human-readable explanation |

#### 1.5.1 VRAM Recommendation

```python
class VRamRecommendation(StrEnum):
    """Heuristic VRAM assessment outcome."""

    PROCEED = "proceed"
    WARN = "warn"
    CONFIRM_REQUIRED = "confirm-required"
```

**Decision formula** (FR-013):
- If `free_vram_bytes` is `None`: skip check, return `recommendation = "proceed"` with logged warning.
- If `free_vram_bytes × 0.85 >= model_file_size_bytes × 1.2`: `recommendation = "proceed"`.
- If `free_vram_bytes × 0.85 < model_file_size_bytes × 1.2`: `recommendation = "confirm-required"`.

---

### 1.6 Provenance Record

Build provenance for smoke probes.

| Field | Type | Validation |
| --- | --- | --- |
| `sha` | `str` | Binary tip SHA (40-char hex); `"unknown"` if git unavailable |
| `version` | `str` | Package version from `importlib.metadata.version('llm_runner')`; `"dev"` if unavailable |

---

### 1.7 Lock Metadata

Persisted in `slot-{slot_id}.lock` files.

| Field | Type | Validation |
| --- | --- | --- |
| `pid` | `int` | Process ID of the server |
| `port` | `int` | Port the server is bound to |
| `started_at` | `float` | `time.time()` at lock acquisition |
| `version` | `str` | Lockfile format version (e.g. `"1.0"`) |

**Staleness rule**: A lock is stale if `time.time() - started_at > lock_stale_threshold_s` (default 300 s) or if the owning PID no longer exists (`psutil.pid_exists(pid) == False`).

---

### 1.8 Smoke Composite Report

Top-level output from `smoke both`.

| Field | Type | Validation |
| --- | --- | --- |
| `slots` | `list[SmokeSlotResult]` | Per-slot results in declaration order |
| `overall_exit_code` | `int` | Highest-severity (lowest-numbered) failure code; 0 if all pass |

Each entry in `slots` maps to a `SmokeProbeResult` plus the per-slot exit code.

---

### 1.9 Doctor Check Result

Per-check outcome from `doctor --json`.

| Field | Type | Validation |
| --- | --- | --- |
| `name` | `str` | Check identifier (e.g. `"config"`, `"hardware"`, `"lockfile"`) |
| `status` | `DoctorCheckStatus` | `pass`, `warn`, or `fail` |
| `message` | `str \| None` | Human-readable detail |

```python
class DoctorCheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
```

---

### 1.10 Audit Log Entry

Rotating log entry for mutating actions.

| Field | Type | Validation |
| --- | --- | --- |
| `command` | `str` | Action name (e.g. `"launch"`, `"shutdown"`, `"smoke"`) |
| `timestamp` | `float` | `time.time()` at action start |
| `exit_code` | `int` | Result exit code |
| `truncated_output` | `str` | Output truncated to configured max length |
| `redacted` | `bool` | Whether secrets were redacted |

---

### 1.11 Smoke Probe Configuration

Runtime settings for smoke probes (subset of `Config`).

> **Naming note**: The fields below are dataclass attribute names on `SmokeProbeConfiguration`.
> In config files, CLI help text, and the PRD/spec, these are expressed as nested config keys
> under the `smoke` namespace using dot notation (e.g., `smoke.inter_slot_delay_s`). The mapping
> is one-to-one: the dataclass field `inter_slot_delay_s` maps to the config key `smoke.inter_slot_delay_s`,
> `listen_timeout_s` → `smoke.listen_timeout_s`, etc. This is distinct from the `ServerConfig` fields
> `smoke_api_key` (per-slot override) — the `SmokeProbeConfiguration.api_key` is the global default
> used when no per-slot override is provided.

| Field (dataclass) | Config key | Type | Default | Range |
| --- | --- | --- | --- | --- |
| `inter_slot_delay_s` | `smoke.inter_slot_delay_s` | `int` | 2 | ≥ 0 |
| `listen_timeout_s` | `smoke.listen_timeout_s` | `int` | 120 | ≥ 1 |
| `http_request_timeout_s` | `smoke.http_request_timeout_s` | `int` | 10 | ≥ 1 |
| `max_tokens` | `smoke.max_tokens` | `int` | 16 | 8–32 |
| `prompt` | `smoke.prompt` | `str` | `"Respond with exactly one word."` | Non-empty |
| `skip_models_discovery` | `smoke.skip_models_discovery` | `bool` | `False` | — |
| `api_key` | `smoke.api_key` | `str` | `""` (from env/config) | — |
| `model_id_override` | `smoke.model_id_override` | `str \| None` | `None` | — |
| `first_token_timeout_s` | `smoke.first_token_timeout_s` | `int` | 1200 | ≥ 1 (Phase 3 chat probe only) |
| `total_chat_timeout_s` | `smoke.total_chat_timeout_s` | `int` | 1500 | ≥ 1 (Phase 3 chat probe only) |

---

### 1.12 Consecutive Failure Counter

In-memory tracking of consecutive model-not-found failures (FR-006).

| Field | Type | Scope |
| --- | --- | --- |
| `slot_id` | `str` | Per slot |
| `count` | `int` | Per slot |
| `model_id_override` | `str \| None` | Last override attempt |

**Rules**:
- Counter resets on process restart (in-memory only).
- Counter increments only on exit code 13 (model not found).
- Counter does NOT reset if override fails again.
- After 2 consecutive failures with wrong ID, require explicit `smoke.model_id` override.

---

## 2. State Transition Diagram

```
                    ┌──────────┐
                    │  idle    │
                    └────┬─────┘
                         │ launch initiated (lock free)
                         ▼
                  ┌─────────────┐
                  │  launching  │◄──────────────────────────────┐
                  └──────┬──────┘                                 │
                         │ process accepts TCP                   │ relaunch
                         ▼                                       │
                  ┌─────────────┐         ┌─────────────┐        │
                  │  running    │────────▶│  degraded   │        │
                  └──────┬──────┘ health  └──────┬──────┘        │
                         │                       │ probe fails    │
                         │ process exits         │ probe fails    │
                         ▼                       ▼               │
                  ┌─────────────┐         ┌─────────────┐        │
                  │  crashed    │────────▶│  offline    │────────┘
                  └──────┬──────┘ cleanup └──────┬──────┘
                         │                       │ reset / relaunch
                         │ reset                 │
                         ▼                       │
                    ┌──────────┐◄────────────────┘
                    │  idle    │
                    └──────────┘

Transitions from any active state:
  running/crashed/degraded → offline  (health probe fails, unresponsive)
  any active → idle              (user-initiated shutdown)
```

**Key invariants:**
- Only one process per slot at any time.
- `launching` requires an uncontested lock.
- `offline` slots may transition directly to `launching` (relaunch) or must reset to `idle` first.
- `crashed` slots must complete cleanup before returning to `idle`.

---

## 3. JSON Schemas

### 3.1 Smoke Probe Result (per slot)

```json
{
  "slot_id": "string",
  "status": "pass" | "fail" | "timeout" | "crashed" | "model_not_found" | "auth_failure",
  "phase_reached": "listen" | "models" | "chat" | "complete",
  "failure_phase": "listen" | "models" | "chat" | null,
  "model_id": "string" | null,
  "latency_ms": "integer" | null,
  "provenance": {
    "sha": "string",
    "version": "string"
  }
}
```

### 3.2 Smoke Composite Report

```json
{
  "slots": [
    {
      "slot_id": "string",
      "status": "pass" | "fail" | "timeout" | "crashed" | "model_not_found" | "auth_failure",
      "phase_reached": "listen" | "models" | "chat" | "complete",
      "failure_phase": "listen" | "models" | "chat" | null,
      "model_id": "string" | null,
      "latency_ms": "integer" | null,
      "provenance": {
        "sha": "string",
        "version": "string"
      }
    }
  ],
  "overall_exit_code": "integer"
}
```

### 3.3 Doctor JSON Output

```json
{
  "checks": [
    {
      "name": "string",
      "status": "pass" | "warn" | "fail",
      "message": "string" | null
    }
  ],
  "config": {
    "effective_values": {
      "key": "value"
    }
  },
  "hardware": {
    "fingerprint": "string",
    "gpus": [
      {
        "pci_address": "string",
        "vendor": "string",
        "name": "string"
      }
    ]
  }
}
```

### 3.4 Lockfile Entry

```json
{
  "version": "1.0",
  "pid": "integer",
  "port": "integer",
  "started_at": "number"
}
```

### 3.5 Hardware Allowlist Entry

```json
{
  "fingerprint": "string",
  "pci_device_ids": ["string"],
  "created_at": "number",
  "invalidated": "boolean"
}
```

### 3.6 GGUF Metadata Record

```json
{
  "raw_path": "string",
  "normalized_stem": "string",
  "general_name": "string" | null,
  "architecture": "string" | null,
  "tokenizer_type": "string" | null,
  "embedding_length": "integer" | null,
  "block_count": "integer" | null,
  "context_length": "integer" | null,
  "attention_head_count": "integer" | null,
  "attention_head_count_kv": "integer" | null,
  "parse_timestamp": "number",
  "prefix_cap_bytes": "integer",
  "parse_timeout_s": "number",
  "format_version": "integer" | null
}
```

### 3.7 VRAM Risk Assessment

```json
{
  "free_vram_bytes": "integer" | null,
  "model_file_size_bytes": "integer",
  "estimated_footprint_bytes": "integer",
  "safety_margin_factor": "number",
  "recommendation": "proceed" | "warn" | "confirm-required",
  "warning_message": "string" | null
}
```

### 3.8 Provenance Record

```json
{
  "sha": "string",
  "version": "string"
}
```

---

## 4. Relationships Between Entities

```
Config ──────────────────────────────────────────────────────────┐
  │                                                              │
  │ contains                                                     │
  ▼                                                              │
  ├──> ServerConfig × N  ────┐                                   │
  │   │                       │ defines                         │
  │   │ port, bind_address    │ launch parameters               │
  │   │ model_path ───────────┤                                 │
  │   │                       │                                 │
  │   │                         ▼                               │
  │   │              SmokeProbeConfig (inline in Config)        │
  │   │                         │                               │
  │   │                         │ used by                       │
  │   ▼                         ▼                               │
  │   SmokeProbeResult × N      ──►  SmokeCompositeReport       │
  │                                                              │
  ModelSlot × N ─────────────────────────────────────────────────┤
    │ slot_id ───────────────────────────────────────────────────┤
    │ model_path ──► GGUFMetadataRecord (extracted)              │
    │ port ──► LockMetadata (persisted)                          │
    │                                                              │
    │                                                              │
    │                                                              │
    │                                                              │
    │                                                              │
    └──► SlotRuntime (in-memory state)                            │
          │ slot_id (foreign key to ModelSlot)                    │
          │ state (SlotState enum, §1.1.1)                        │
          │ pid (optional, set on launch)                         │
          │ server_config (ServerConfig)                          │
          │                                                        │
          │ ┌──────────────────────────────────────────────────┐  │
          │ │ VRAMRiskAssessment (per-slot, computed)          │  │
          │ │   free_vram_bytes ◄── GPUStats                    │  │
          │ │   model_file_size_bytes ◄── GGUFMetadataRecord   │  │
          │ └──────────────────────────────────────────────────┘  │
          │                                                        │
          │ ┌──────────────────────────────────────────────────┐  │
          │ │ ConsecutiveFailureCounter (per-slot, in-memory)  │  │
          │ │   slot_id, count, model_id_override              │  │
          │ └──────────────────────────────────────────────────┘  │
          │                                                        │
          │ ┌──────────────────────────────────────────────────┐  │
          │ │ ProvenanceRecord (shared across slots)           │  │
          │ │   sha (git HEAD), version (package)              │  │
          │ └──────────────────────────────────────────────────┘  │
          │                                                        │
          └────────────────────────────────────────────────────────┘

HardwareAllowlistEntry (persisted, one per machine)
  │
  │ invalidated when
  ▼
  MachineFingerprint (SHA256 of lspci + sycl-ls)
```

**Dependency graph:**

```
Config ──► ServerConfig
Config ──► SmokeProbeConfig
Config ──► GGUFMetadataRecord (via model_path)
Config ──► VRAMRiskAssessment (via model_path + GPUStats)

ModelSlot ──► ServerConfig (factory function)
ModelSlot ──► SlotRuntime (per-slot state)
ModelSlot ──► LockMetadata (via slot_id)
ModelSlot ──► GGUFMetadataRecord (via model_path)

SlotState (enum) ◄── SlotRuntime.state
SmokePhase (enum)     ◄── SmokeProbeResult.phase_reached
SmokeFailurePhase (enum) ◄── SmokeProbeResult.failure_phase
SmokeProbeStatus (enum) ◄── SmokeProbeResult.status
VRamRecommendation (enum) ◄── VRAMRiskAssessment.recommendation
```

**One-way dependencies (respecting architecture boundaries):**

```
llama_cli/ ──► llama_manager/   (I/O layer depends on pure library)
tests/     ──► llama_manager/   (unit tests depend on pure library)
llama_cli/ ──► llama_cli/       (internal imports only)
```

---

## 5. Python Dataclass Signatures

### 5.1 Enums

```python
from enum import StrEnum


class SlotState(StrEnum):
    """Six operational states for a GPU slot."""

    IDLE = "idle"
    LAUNCHING = "launching"
    RUNNING = "running"
    DEGRADED = "degraded"
    CRASHED = "crashed"
    OFFLINE = "offline"


class SmokePhase(StrEnum):
    """Ordered phases of a smoke probe session."""

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"
    COMPLETE = "complete"


class SmokeFailurePhase(StrEnum):
    """Phases where a failure can occur (excludes 'complete')."""

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"


class SmokeProbeStatus(StrEnum):
    """Outcome of a single slot smoke probe."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    MODEL_NOT_FOUND = "model_not_found"
    AUTH_FAILURE = "auth_failure"


class VRamRecommendation(StrEnum):
    """Heuristic VRAM assessment outcome."""

    PROCEED = "proceed"
    WARN = "warn"
    CONFIRM_REQUIRED = "confirm-required"


class DoctorCheckStatus(StrEnum):
    """Status of a doctor check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GgufParseError(StrEnum):
    """Error types for GGUF metadata extraction."""

    CORRUPT_FILE = "corrupt_file"
    PARSE_TIMEOUT = "parse_timeout"
    UNSUPPORTED_VERSION = "unsupported_version"
    READ_ERROR = "read_error"
```

### 5.2 Core Dataclasses

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class ProvenanceRecord:
    """Build provenance for smoke probes.

    Attributes:
        sha: Binary tip SHA (40-char hex) or 'unknown'.
        version: Package version or 'dev'.
    """

    sha: str
    version: str


@dataclass
class SmokeProbeResult:
    """Per-slot outcome from a smoke probe session.

    Attributes:
        slot_id: Normalized slot identifier.
        status: Probe outcome.
        phase_reached: Last phase attempted.
        failure_phase: Phase where failure occurred (null if pass).
        model_id: Resolved model ID or null.
        latency_ms: Single observed latency in milliseconds or null.
        provenance: Build provenance.
    """

    slot_id: str
    status: SmokeProbeStatus
    phase_reached: SmokePhase
    failure_phase: SmokeFailurePhase | None
    model_id: str | None
    latency_ms: int | None
    provenance: ProvenanceRecord


@dataclass
class SmokeCompositeReport:
    """Top-level output from 'smoke both'.

    Attributes:
        slots: Per-slot results in declaration order.
        overall_exit_code: Highest-severity (lowest-numbered) failure code.
    """

    slots: list[SmokeProbeResult]
    overall_exit_code: int


@dataclass
class GGUFMetadataRecord:
    """Extracted header fields from a GGUF file.

    Attributes:
        raw_path: Absolute path to the GGUF file.
        normalized_stem: NFKC-normalized filename stem.
        general_name: general.name from header, or None.
        architecture: general.architecture, or None.
        tokenizer_type: tokenizer.type, or None.
        embedding_length: llama.embedding_length, or None.
        block_count: llama.block_count, or None.
        context_length: llama.context_length, or None.
        attention_head_count: llama.attention.head_count, or None.
        attention_head_count_kv: llama.attention.head_count_kv, or None.
        parse_timestamp: time.time() at extraction start.
        prefix_cap_bytes: Max bytes read from file start.
        parse_timeout_s: Wall-clock timeout in seconds.
        format_version: GGUF version, or None if unreadable.
    """

    raw_path: str
    normalized_stem: str
    general_name: str | None
    architecture: str | None
    tokenizer_type: str | None
    embedding_length: int | None
    block_count: int | None
    context_length: int | None
    attention_head_count: int | None
    attention_head_count_kv: int | None
    parse_timestamp: float
    prefix_cap_bytes: int
    parse_timeout_s: float
    format_version: int | None


@dataclass
class VRamRiskAssessment:
    """Best-effort free memory analysis.

    Attributes:
        free_vram_bytes: From GPU telemetry, or None if unavailable.
        model_file_size_bytes: On-disk file size of the GGUF model.
        estimated_footprint_bytes: Heuristic model footprint.
        safety_margin_factor: Default 0.85.
        recommendation: Heuristic outcome.
        warning_message: Human-readable explanation, or None.
    """

    free_vram_bytes: int | None
    model_file_size_bytes: int
    estimated_footprint_bytes: int
    safety_margin_factor: float = 0.85
    recommendation: VRamRecommendation = VRamRecommendation.PROCEED
    warning_message: str | None = None


@dataclass
class LockMetadata:
    """Lockfile metadata persisted in slot-{slot_id}.lock files.

    Attributes:
        pid: Process ID of the owner.
        port: Port the server is bound to.
        started_at: time.time() at lock acquisition.
        version: Lockfile format version string.
    """

    pid: int
    port: int
    started_at: float
    version: str = "1.0"


@dataclass
class HardwareAllowlistEntry:
    """Machine fingerprint + PCI device IDs.

    Attributes:
        fingerprint: SHA256 of normalized lspci + sycl-ls output.
        pci_device_ids: PCI addresses (e.g. "0000:01:00.0").
        created_at: time.time() at creation.
        invalidated: True when fingerprint or PCI set changes.
    """

    fingerprint: str
    pci_device_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    invalidated: bool = False


@dataclass
class SmokeProbeConfiguration:
    """Runtime settings for smoke probes.

    Attributes:
        inter_slot_delay_s: Pause between slot probes.
        listen_timeout_s: TCP ready-check timeout per slot.
        http_request_timeout_s: HTTP request timeout.
        max_tokens: Token limit for chat probe (8-32).
        prompt: Single-turn user message text.
        skip_models_discovery: Skip /v1/models phase.
        api_key: API key from CLI flag, config, or env.
        model_id_override: User-provided model ID override.
        first_token_timeout_s: Wall-clock timeout to first token (Phase 3 only).
        total_chat_timeout_s: Hard cap from chat request to final response (Phase 3 only).
    """

    inter_slot_delay_s: int = 2
    listen_timeout_s: int = 120
    http_request_timeout_s: int = 10
    max_tokens: int = 16
    prompt: str = "Respond with exactly one word."
    skip_models_discovery: bool = False
    api_key: str = ""
    model_id_override: str | None = None
    first_token_timeout_s: int = 1200
    total_chat_timeout_s: int = 1500

    def __post_init__(self) -> None:
        if not (8 <= self.max_tokens <= 32):
            raise ValueError("max_tokens must be between 8 and 32")


@dataclass
class ConsecutiveFailureCounter:
    """In-memory tracking of consecutive model-not-found failures.

    Attributes:
        slot_id: Slot identifier.
        count: Number of consecutive failures.
        model_id_override: Last override attempt, or None.
    """

    slot_id: str
    count: int = 0
    model_id_override: str | None = None

    def reset(self) -> None:
        """Reset the counter to zero."""
        self.count = 0
        self.model_id_override = None


@dataclass
class AuditLogEntry:
    """Rotating log entry for mutating actions.

    Attributes:
        command: Action name.
        timestamp: time.time() at action start.
        exit_code: Result exit code.
        truncated_output: Output truncated to max length.
        redacted: Whether secrets were redacted.
    """

    command: str
    timestamp: float
    exit_code: int
    truncated_output: str = ""
    redacted: bool = False


@dataclass
class DoctorCheckResult:
    """Per-check outcome from doctor --json.

    Attributes:
        name: Check identifier.
        status: pass, warn, or fail.
        message: Human-readable detail, or None.
    """

    name: str
    status: DoctorCheckStatus
    message: str | None = None


@dataclass
class DoctorReport:
    """Top-level output from doctor --json.

    Attributes:
        checks: Per-check results.
        config: Effective configuration values.
        hardware: Hardware fingerprint and device list.
    """

    checks: list[DoctorCheckResult]
    config: dict[str, object]
    hardware: dict[str, object]


@dataclass
class SlotRuntime:
    """In-memory per-slot runtime state (not persisted).

    Attributes:
        slot_id: Normalized slot identifier.
        state: Current operational state.
        pid: Process ID, or None.
        server_config: Per-instance launch parameters.
        metadata: Extracted GGUF metadata, or None.
        vram_assessment: VRAM risk assessment, or None.
        failure_counter: Consecutive failure tracking.
    """

    slot_id: str
    state: SlotState = SlotState.IDLE
    pid: int | None = None
    server_config: ServerConfig | None = None
    metadata: GGUFMetadataRecord | None = None
    vram_assessment: VRamRiskAssessment | None = None
    failure_counter: ConsecutiveFailureCounter | None = None
```

---

## 6. Exit Code Mapping

Smoke exit codes (10–19) map to probe outcomes:

| Exit Code | SmokeProbeStatus | SmokePhase |
| --- | --- | --- |
| 10 | `TIMEOUT` | `LISTEN` |
| 11 | `FAIL` | `MODELS` or `CHAT` (network/API) |
| 12 | `FAIL` | N/A (config validation) |
| 13 | `MODEL_NOT_FOUND` | `MODELS` or `CHAT` |
| 14 | `TIMEOUT` | `CHAT` |
| 15 | `AUTH_FAILURE` | `MODELS` or `CHAT` |
| 19 | `CRASHED` | Any (process exited) |

Composite report `overall_exit_code` is the minimum (highest-severity) among all slot results.

---

## 7. Implementation Notes

### 7.1 Module Placement

| Dataclass | Module | Rationale |
| --- | --- | --- |
| `SlotState`, `SmokePhase`, `SmokeProbeStatus`, `VRamRecommendation`, `DoctorCheckStatus`, `GgufParseError` | `llama_manager/config.py` | Enums co-located with existing enums (`ErrorCode`, `ModelSlot`) |
| `ProvenanceRecord` | `llama_manager/smoke.py` | Smoke-specific, pure data |
| `SmokeProbeResult`, `SmokeCompositeReport` | `llama_manager/smoke.py` | Smoke probe output types |
| `GGUFMetadataRecord` | `llama_manager/metadata.py` | GGUF-specific |
| `VRamRiskAssessment` | `llama_manager/server.py` | Tied to VRAM heuristic logic |
| `LockMetadata` | `llama_manager/process_manager.py` | Already exists; add `version` field |
| `HardwareAllowlistEntry` | `llama_manager/server.py` | Tied to hardware detection |
| `SmokeProbeConfiguration` | `llama_manager/config.py` | Extends `Config` defaults |
| `ConsecutiveFailureCounter` | `llama_manager/smoke.py` | Smoke-specific state tracker |
| `AuditLogEntry` | `llama_manager/process_manager.py` | Tied to lifecycle audit |
| `DoctorCheckResult`, `DoctorReport` | `llama_manager/server.py` | Doctor-specific output types |
| `SlotRuntime` | `llama_manager/process_manager.py` | Tied to `ServerManager` lifecycle |

### 7.2 Serialization

All dataclasses that appear in JSON output (`SmokeCompositeReport`, `DoctorReport`, `GGUFMetadataRecord`, `VRamRiskAssessment`, `LockMetadata`, `HardwareAllowlistEntry`) must be JSON-serializable via `dataclasses.asdict()`. Use `field(default_factory=...)` for mutable defaults.

### 7.3 Frozen vs Mutable

- `ProvenanceRecord` — frozen (computed once, never changes)
- `SmokeProbeResult` — mutable (latency may be updated)
- `GGUFMetadataRecord` — frozen (extracted once)
- `LockMetadata` — mutable (updated on restart)
- `SmokeProbeConfiguration` — mutable (CLI flags override defaults)
- `SlotRuntime` — mutable (state transitions)

### 7.4 CI Quality Gates

This data model must pass:
- `uv run ruff check .` — linting
- `uv run ruff format --check .` — formatting
- `uv run pyright` — type checking
- `uv run pytest` — unit tests for all dataclass serialization and validation
