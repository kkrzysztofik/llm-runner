# llama.cpp build documentation

> **Audience:** Operators and developers building `llama-server` binaries for this project's mixed-GPU workstation (Intel Arc SYCL + NVIDIA CUDA).
> **Canonical implementation:** `src/llama_manager/build_pipeline/`
> **Last updated:** 2026-05-19

llm-runner orchestrates cloning, configuring, and compiling [llama.cpp](https://github.com/ggml-org/llama.cpp) for two backends. It records build provenance under XDG state paths and leaves compiled binaries in the source tree. Launch and dry-run commands resolve those binaries via `Config.llama_server_bin_intel` and `Config.llama_server_bin_nvidia`.

---

## Start here

| If you want to… | Read |
|-----------------|------|
| Understand why there are two backends and how `both` works | [overview.md](overview.md) |
| Install Intel oneAPI, CUDA, and common build tools on this machine | [workstation-setup.md](workstation-setup.md) |
| See where sources, binaries, provenance, and logs live | [paths-and-artifacts.md](paths-and-artifacts.md) |
| Learn what each pipeline stage does | [pipeline-stages.md](pipeline-stages.md) |
| See exact CMake flags and SYCL environment wrapping | [cmake-and-backends.md](cmake-and-backends.md) |
| Run builds from CLI or TUI | [cli-and-tui.md](cli-and-tui.md) |
| Fix locks, clone failures, or stale configure caches | [troubleshooting.md](troubleshooting.md) |

---

## Quick commands

```bash
# Toolchain check (no build)
uv run llm-runner setup check

# Preview all stages without executing (still runs preflight logic in dry-run mode per stage)
uv run llm-runner build sycl --dry-run

# Full SYCL build
uv run llm-runner build sycl

# SYCL then CUDA (serialized)
uv run llm-runner build both
```

---

## Related documentation

| Document | Scope |
|----------|--------|
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Python `src/` layout; §15 summarizes the build system and links here |
| [../PRD.md](../PRD.md) | Product requirements (FR-004 build pipeline, FR-006 provenance) |
| [../../README.md](../../README.md) | User-facing command cheat sheet |
| [../../specs/002-build-setup/quickstart.md](../../specs/002-build-setup/quickstart.md) | M2 milestone validation scenarios (verify paths against this doc set) |

Upstream CMake options not set by llm-runner: [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
