# Workstation setup (Intel Arc B580 + NVIDIA RTX 3090)

This guide prepares a **Linux** host for llm-runner's llama.cpp builds: SYCL for Intel Arc and CUDA for NVIDIA. It aligns with what `toolchain.detect_toolchain()` and `build` preflight actually check.

Target hardware (from project defaults):

| GPU | Role | Runtime device string |
|-----|------|------------------------|
| Intel Arc B580 | Summary models (Qwen 2.5-class) | `SYCL0` |
| NVIDIA RTX 3090 | Code / reasoning (Qwen 3.5-class) | `GPU 0` |

---

## 1. Common build dependencies

Install on Debian/Ubuntu-style systems:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  gcc g++ make \
  git \
  cmake
```

**CMake:** llm-runner expects **CMake ≥ 3.24** (`CMAKE_MINIMUM_VERSION` in `toolchain/constants.py`). Verify:

```bash
cmake --version
```

If the distro package is too old, install a newer CMake from [cmake.org](https://cmake.org/download/) or Kitware's apt repository.

---

## 2. Intel oneAPI (SYCL backend)

### Install

Official route (inspect scripts before running with sudo):

```bash
curl -sSf https://apt.repos.intel.com/install.sh -o /tmp/intel-install.sh
less /tmp/intel-install.sh   # review
sudo sh /tmp/intel-install.sh
sudo apt-get install -y oneapi-compiler-dpcpp-cpp oneapi-compiler-dpcpp-rt
```

Documentation: [Intel oneAPI DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html).

### Environment

Default environment script:

```bash
source /opt/intel/oneapi/setvars.sh
```

llm-runner **automatically** sources this file when building SYCL if it exists (`build_pipeline/utils.py`). Interactive shells should still add `setvars` to your profile if you run cmake manually.

### Verify compilers

Preflight looks for the first available among `icpx`, `icx`, `dpcpp` (PATH or `/opt/intel/oneapi/compiler/latest/bin/`):

```bash
source /opt/intel/oneapi/setvars.sh
icpx --version
icx --version
```

### Intel GPU / SYCL runtime

For Arc inference (outside the build doc scope but required at runtime):

- Install Intel GPU drivers and Level Zero/SYCL runtime packages for your distro.
- Use `sycl-ls` to confirm the Arc device is visible.
- OpenCode/llm-runner shell notes: `ONEAPI_DEVICE_SELECTOR` may be set at slot level — see legacy [`run_opencode_models.sh`](../../run_opencode_models.sh) comments and [../ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 3. NVIDIA CUDA (CUDA backend)

### Driver

Install a driver version compatible with your CUDA toolkit. Verify:

```bash
nvidia-smi
```

You should see the RTX 3090 and a driver/CUDA version line.

### CUDA toolkit

Example (adjust version to match your driver):

```bash
sudo apt-get install -y cuda-toolkit-12-2
```

Or follow [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) install docs.

Ensure `nvcc` is on `PATH`:

```bash
nvcc --version
```

### nvtop (optional, recommended)

CUDA preflight's `missing_tools(cuda)` lists **nvtop** when absent. It is not required to compile, but the project uses GPU monitoring in the TUI:

```bash
sudo apt-get install -y nvtop
```

---

## 4. llm-runner Python environment

From the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync --extra dev
```

Create the optional XDG venv used by setup:

```bash
uv run llm-runner setup check
uv run llm-runner setup venv --yes
```

---

## 5. Verification checklist

Run in order:

```bash
# 1) Toolchain table
uv run llm-runner setup check

# 2) Dry-run full pipeline (no compile)
uv run llm-runner build sycl --dry-run
uv run llm-runner build cuda --dry-run

# 3) Single-backend production build
uv run llm-runner build sycl
# expect: ~/.cache/llm-runner/llama.cpp/build/bin/llama-server

uv run llm-runner build cuda
# expect: ~/.cache/llm-runner/llama.cpp/build_cuda/bin/llama-server

# 4) Confirm provenance
cat ~/.local/state/llm-runner/builds/sycl/build-artifact.json
cat ~/.local/state/llm-runner/builds/cuda/build-artifact.json

# 5) Binary smoke (optional)
~/.cache/llm-runner/llama.cpp/build/bin/llama-server --version
~/.cache/llm-runner/llama.cpp/build_cuda/bin/llama-server --version
```

### Using a custom llama.cpp checkout

If sources already live outside XDG cache (e.g. legacy `~/src/llama.cpp`):

```bash
export LLAMA_CPP_ROOT=/home/kmk/src/llama.cpp
uv run llm-runner build sycl
uv run llm-runner build cuda
```

Binaries must end up at `$LLAMA_CPP_ROOT/build/bin/llama-server` and `$LLAMA_CPP_ROOT/build_cuda/bin/llama-server` for defaults to match.

---

## 6. Serialized dual build

When both backends are ready:

```bash
uv run llm-runner build both
```

Expect SYCL to finish (or fail) before CUDA starts. See [overview.md](overview.md).

---

## 7. Mapping to FR-005 hints

When `setup check` or preflight fails, llm-runner emits structured hints from `toolchain/constants.py` (`GCC_HINT`, `CMAKE_HINT`, `SYCL_HINT`, `CUDA_HINT`, `NVTOP_HINT`). Use the `how_to_fix` text in CLI output; this page is the expanded operator version of the same requirements.

---

## 8. What this guide does not cover

- Downloading GGUF model weights (`MODEL_*` paths in config)
- vLLM or non-llama.cpp backends (out of MVP scope)
- CI runners without GPUs (build tests mock filesystems; use `pytest`, not real compiles)

For path reference after a successful build, see [paths-and-artifacts.md](paths-and-artifacts.md).
