# NVIDIA GPU default power limit

Date: 2026-08-18
Status: approved

## Goal

Apply a default power cap to every NVIDIA GPU used by a launched profile, set once per launch, before the llama-server process starts. Default on at 290 W. Reduces heat/power/noise during sustained inference on the dual RTX 3090 (stock 350 W) rig.

nvidia-smi power limits reset on reboot/driver reload, so "default" means re-applied on every launch.

## Decisions

- **Apply at launch, not display-only.** The feature writes the cap; the TUI telemetry already shows `power.draw` and is untouched.
- **Global default only.** One value in Config (`server_defaults`), applied to every CUDA device of every NVIDIA profile. No per-profile or per-device override.
- **Fixed watts, not percentage of max.** Stored and applied verbatim. `0` disables.
- **Default 290 W, on by default.** Config ships `290`.
- **Best-effort, warn and continue.** If the cap cannot be applied (permission, missing driver, timeout), log a warning and launch anyway. The cap must never block a server start.
- **`sudo -n` (non-interactive).** Fails fast instead of hanging on a password prompt. Requires passwordless sudo for the cap to actually apply; otherwise the warning path fires and the server still starts.
- **All CUDA devices in the profile**, parsed from the device string (`CUDA0,CUDA1` → both GPUs).
- Side effect lives in the pure library orchestration layer, so the CLI launch, TUI launch, and `llm-runner <mode>` all get it in one place. Not in the shell wrapper (legacy surface).

## Data model

Add to `ServerDefaultsConfig` in `src/llama_manager/config/defaults.py`:

| Field | Default | Notes |
|-------|---------|-------|
| `nvidia_power_limit_watts` | `290` | `0` = disabled. Plain field, NOT on `LaunchRuntimeFields` — no `--` flag ever reaches llama-server; the cap is an nvidia-smi side effect, not a server arg. |

## Orchestration

New module `src/llama_manager/orchestration/power_limit.py`:

- `cuda_ordinals(device: str) -> list[int]`
  - `"CUDA0,CUDA1"` → `[0, 1]`
  - `"cuda:0"` → `[0]`
  - `""` / `"auto"` → `[0]` (NVIDIA auto-detect defaults to device 0)
  - `"SYCL0"`, anything non-CUDA → `[]`
  - Garbage → `[]` (no crash, no cap)
- `apply_nvidia_power_limit(device: str, watts: int, warn: Callable[[str], None]) -> None`
  - If `watts <= 0`: no-op.
  - For each ordinal: `sudo -n nvidia-smi -i <idx> -pl <watts>` via `subprocess.run` with a short timeout (2 s).
  - Non-zero exit, `OSError`, or timeout → call `warn(...)` and continue to the next ordinal.

Hook in `ServerManager.start_servers` (`src/llama_manager/orchestration/manager.py`), immediately before `build_server_cmd(cfg)` for each config:

- If the config's device is NVIDIA and the watts value > 0, call `apply_nvidia_power_limit(cfg.device, watts, warn)`.

The watts value is read from the resolved `Config` (`base_config.server_defaults.nvidia_power_limit_watts`). The `ServerManager` already receives configs; the base `Config` is available at the call site in `launch_orchestrate` / the CLI — the hook is passed the watts value as a parameter so `ServerManager` stays decoupled from `Config`.

## Dry-run

`_build_hardware_notes` in `src/llama_manager/validation/commands/builder.py` gains `"power_limit_watts": <int | None>` (value, or `None` when 0). Dry-run output shows the planned cap alongside the existing hardware notes.

## UI

Config modal (`src/llama_cli/tui/components/config_modal.py`):

- `default_nvidia_power_limit_watts: str = "290"` payload field, mapped to `server_defaults.nvidia_power_limit_watts`.
- Numeric Input in the global-defaults section with a "0 = disabled" hint.
- Validation: non-negative integer; reject negatives and non-numeric input.

## Testing

Unit tests only. No GPU, no real subprocess llama-server, no real nvidia-smi.

- `cuda_ordinals`: table-driven cases for `CUDA0,CUDA1`, `cuda:0`, `""`, `auto`, `SYCL0`, `cuda:1,2`, garbage.
- `apply_nvidia_power_limit`: mocked `subprocess.run` — success (no warn), non-zero exit (warns, no raise), `OSError` (warns, no raise), timeout (warns, no raise), watts=0 (no subprocess call).
- `ServerManager.start_servers`: hook runs for an NVIDIA config, skipped for SYCL, skipped for watts=0.
- Config modal: default `290` loads; `0` and a custom value round-trip through save/load.

## Out of scope

- Displaying `power.limit` in the TUI telemetry (feature is apply-at-launch only).
- Per-profile or per-device cap values.
- Percentage-of-max semantics.
- Password-prompting sudo; interactive elevation.
- Persistent power-limit daemon/service (caps reset on reboot by design; re-applied at each launch).
- Setting the cap in `run_opencode_models.sh`.