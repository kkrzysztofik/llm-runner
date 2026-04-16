# In-progress fixes for 4 critical issues

## Issue 1: FR-004.2a - CMake flag constants

- Need to add class constants to BuildConfig
- Add BOTH to BuildBackend enum

## Issue 2: FR-004.4 - Build lock path wrong

- Change `src/llama_manager/config.py` line 155 to use `xdg_cache_base` instead of `xdg_state_base`

## Issue 3: FR-006.1 - Provenance incomplete

- Get real git commit SHA
- Find binary path
- Create log path
- Keep created_at as datetime (not float)

## Issue 4: SC-003 - Serialized builds for 'both' backend

- Add BOTH to BuildBackend enum
- Add run_both_backends() method
- Update run() to handle BOTH
