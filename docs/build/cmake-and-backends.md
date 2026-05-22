# CMake configuration and backends

Configure logic lives in `llama_manager/build_pipeline/stages/configure.py` (`get_cmake_flags`, `run_configure`). Build compile uses the same build directory via `stages/build.py`.

## Build directories

| Backend | Default `build_dir` (under source root) | Binary output |
|---------|----------------------------------------|---------------|
| SYCL | `build/` | `build/bin/llama-server` |
| CUDA | `build_cuda/` | `build_cuda/bin/llama-server` |

Keeping separate trees allows SYCL and CUDA artifacts to coexist without re-configuring the same directory.

## CMake flags (llm-runner defaults)

`get_cmake_flags(backend)` appends:

### All backends

```text
-DBUILD_SERVER=ON
-DGGML_NATIVE=OFF
```

- **BUILD_SERVER=ON** — build the HTTP server binary, not just CLI tools.
- **GGML_NATIVE=OFF** — disable native CPU tuning flags for portable builds across machines.

### SYCL (Intel Arc)

```text
-DGGML_SYCL=ON
-DCMAKE_C_COMPILER=icx
-DCMAKE_CXX_COMPILER=icpx
```

Intel oneAPI compilers must be available (see [workstation-setup.md](workstation-setup.md)). Detection probes `icpx`, then `icx`, then `dpcpp`, including fallback path `/opt/intel/oneapi/compiler/latest/bin/`.

### CUDA (NVIDIA)

```text
-DGGML_CUDA=ON
```

Uses system `nvcc` from the CUDA toolkit on `PATH`. No extra compiler variables are set by llm-runner.

## Full configure invocation

Conceptually:

```bash
cmake -S "$SOURCE" -B "$BUILD_DIR" \
  -DBUILD_SERVER=ON \
  -DGGML_NATIVE=OFF \
  ... backend-specific flags ...
```

Then:

```bash
cmake --build "$BUILD_DIR" -j "$(nproc)"
```

Exact argv lists appear in `build --dry-run` output and in `build-artifact.json` → `build_command`.

## Intel oneAPI environment wrapper

For `BuildBackend.SYCL` only, `get_build_env_cmd()` may rewrite:

```python
["cmake", "-S", ..., "-B", ..., ...]
```

into:

```python
["bash", "-c", 'source "/opt/intel/oneapi/setvars.sh" && cmake -S ... -B ... ...']
```

| Condition | Behavior |
|-----------|----------|
| Backend is CUDA | Command unchanged |
| `setvars.sh` missing | Command unchanged (you must export oneAPI paths manually) |
| `setvars.sh` present | Wrap so cmake and compile see `icx`/`icpx` and SYCL libraries |

Path constant: `_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")`.

The **build** stage applies the same wrapper to `cmake --build`.

## CMake minimum version

`toolchain.constants.CMAKE_MINIMUM_VERSION = (3, 24, 0)`. Install hints reference distro packages; verify with `cmake --version` before building.

## Reconfigure vs incremental build

| Situation | What happens |
|-----------|----------------|
| First build | configure creates `CMakeCache.txt`, then build compiles |
| Second build, default `update_sources=True` | clone may fetch; configure runs again |
| Second build, `--no-update-sources` and existing cache | configure **skipped**; `cmake --build` only |
| Switch backend | Use separate `build` vs `build_cuda` dirs — no cross-contamination |

To force a clean reconfigure, remove the backend's build directory or delete `CMakeCache.txt` inside it.

## What llm-runner does not set

Examples of upstream options **not** passed by this pipeline (set manually if needed):

- `CMAKE_BUILD_TYPE` (Release/Debug)
- `GGML_CUDA_F16`, tensor split, or specific GPU arch flags
- Custom `CMAKE_INSTALL_PREFIX` (no `cmake --install` step)

For advanced tuning, run cmake manually in the same `build_dir` or extend `get_cmake_flags()` in code. Upstream reference: [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

## Hardware mapping at runtime

Build artifacts are backend-specific; **slot assignment** (which GPU port/model uses which binary) is configured separately via `ServerConfig` and profile registry — not in the build pipeline. Typical mapping on this workstation:

| Binary | Slot / role |
|--------|-------------|
| `build/bin/llama-server` | Intel SYCL summary models (ports 8080, 8082) |
| `build_cuda/bin/llama-server` | NVIDIA CUDA code/reasoning (port 8081) |

See [../ARCHITECTURE.md](../ARCHITECTURE.md) §1 hardware table.
