# Phase 5 — Cleanup

No dependencies on earlier phases. Can run in parallel.

---

## Task 11: Split `level_zero.py` into Focused Modules

**Priority**: Medium | **Effort**: ~4h | **Files**: 1 → 4

### Current State

`gpu_telemetry/level_zero.py` (769 lines) contains:
- 12 ctypes struct definitions (lines 44–154)
- Device discovery: `_load_level_zero`, `_enum_handles`, `_discover_level_zero_devices`,
  `_select_level_zero_device` (lines 178–335)
- Memory collection: `_collect_level_zero_memory` (lines 337–356)
- Engine utilization: 4 strategies — native L0, sysfs, fdinfo, DRM (lines 358–730)
- Power + temperature: `_collect_level_zero_power`, `_collect_level_zero_temp`
  (lines 677–769)
- Public entry point: `collect_level_zero_stats` (line 731)

### Function Inventory

| Line | Function | Category |
|------|----------|----------|
| 164 | `_decode_c_string` | Types/util |
| 169 | `_uuid_to_string` | Types/util |
| 174 | `_pci_bdf` | Types/util |
| 178 | `_load_level_zero` | Device |
| 239 | `_enum_handles` | Device |
| 252 | `_discover_level_zero_devices` | Device |
| 310 | `_select_level_zero_device` | Device |
| 337 | `_collect_level_zero_memory` | Telemetry |
| 358 | `_read_engine_stats` | Telemetry |
| 367 | `_collect_level_zero_engine_util` | Telemetry |
| 388 | `_safe_read_text` | Telemetry/sysfs |
| 396 | `_safe_read_int` | Telemetry/sysfs |
| 406 | `_normalize_temp` | Telemetry |
| 413 | `_device_root_matches` | Telemetry/drm |
| 429 | `_iter_device_roots` | Telemetry/drm |
| 458 | `_iter_hwmon_paths` | Telemetry/hwmon |
| 473 | `_read_hwmon_temperature` | Telemetry/hwmon |
| 504 | `_iter_drm_paths` | Telemetry/drm |
| 541 | `_read_drm_engine_busy` | Telemetry/drm |
| 556 | `_collect_sysfs_engine_util` | Telemetry/sysfs |
| 578 | `_fdinfo_matches_device` | Telemetry/fdinfo |
| 593 | `_read_fdinfo_counters` | Telemetry/fdinfo |
| 635 | `_collect_fdinfo_engine_util` | Telemetry/fdinfo |
| 667 | `_collect_native_engine_util` | Telemetry |
| 677 | `_read_power_counters` | Telemetry/power |
| 689 | `_collect_level_zero_power` | Telemetry/power |
| 710 | `_collect_level_zero_temp` | Telemetry/temp |
| 722 | `_collect_sysfs_temp` | Telemetry/temp |
| 731 | `collect_level_zero_stats` | **Public entry point** |

### Struct Inventory

| Line | Class | Description |
|------|-------|-------------|
| 44 | `_ZeDeviceUuid` | ctypes.Structure for device UUID |
| 48 | `_ZeDeviceProperties` | ctypes.Structure for device props |
| 74 | `_ZesDeviceProperties` | ctypes.Structure for ZES device props |
| 89 | `_ZesDeviceExtProperties` | ctypes.Structure for extended props |
| 99 | `_ZesPciAddress` | ctypes.Structure for PCI address |
| 108 | `_ZesPciSpeed` | ctypes.Structure for PCI speed |
| 116 | `_ZesPciProperties` | ctypes.Structure for PCI props |
| 128 | `_ZesMemState` | ctypes.Structure for memory state |
| 138 | `_ZesEngineStats` | ctypes.Structure for engine stats |
| 147 | `_FdinfoCounters` | Dataclass for fdinfo counters |
| 154 | `_LevelZeroDevice` | Dataclass wrapping discovered device |

### Steps

1. **Create `gpu_telemetry/level_zero_types.py`**:
   - Move all 12 ctypes struct definitions + `_LevelZeroDevice`, `_FdinfoCounters`
   - Move `_decode_c_string`, `_uuid_to_string`, `_pci_bdf`

2. **Create `gpu_telemetry/level_zero_device.py`**:
   - Move `_load_level_zero`, `_enum_handles`, `_discover_level_zero_devices`,
     `_select_level_zero_device`
   - Import types from `level_zero_types`

3. **Create `gpu_telemetry/level_zero_telemetry.py`**:
   - Move all engine utilization functions (4 strategies)
   - Move memory, power, temperature collection
   - Move sysfs/fdinfo/hwmon path helpers
   - Move `collect_level_zero_stats` (public entry point)

4. **Slim `level_zero.py`** to a thin facade:
   ```python
   from .level_zero_device import _discover_level_zero_devices, _select_level_zero_device
   from .level_zero_telemetry import collect_level_zero_stats
   from .level_zero_types import ...
   ```

### Pass Criteria

- [ ] No single file > 300 lines
- [ ] `level_zero.py` is a thin re-export facade
- [ ] All existing tests pass

---

## Task 13: Merge `run_profile_store.py` and `slot_profile_store.py`

**Priority**: Low | **Effort**: ~2h | **Files**: 3

### Current State

| Module | Lines | Purpose |
|---|---|---|
| `run_profile_store.py` | 389 | Persist/load run profiles (TOML) — per-slot runtime configs |
| `slot_profile_store.py` | 291 | Persist/load slot profiles (TOML) — per-slot model+hardware configs |

Both deal with TOML-based profile persistence, file I/O, and sanitization. They likely
share patterns for read/write/validate.

### Steps

1. **Audit both modules** for shared patterns:
   - File path resolution
   - TOML read/write
   - Profile validation
   - Sanitization

2. **Extract shared persistence helpers** to `common/profile_io.py`:
   - `read_profile_toml(path) -> dict`
   - `write_profile_toml(path, data: dict) -> None`
   - `profile_dir_path(base_dir, profile_id) -> Path`

3. **Keep both modules** but have them share `profile_io` helpers.
   - OR merge into a single `profile_store.py` with clear sections for run vs slot
     profiles, if the APIs are similar enough.

4. **Update all imports** in `llama_cli/` and tests.

### Pass Criteria

- [ ] No duplicated TOML I/O code
- [ ] Shared helpers in `common/profile_io.py`
- [ ] All existing tests pass
