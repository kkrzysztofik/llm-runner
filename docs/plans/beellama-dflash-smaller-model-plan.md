# BeeLlama DFlash Smaller-Model Plan

Status: Retrospective

BeeLlama llama.cpp source flavor and DFlash smaller-VRAM profile support are implemented; this document records the completed work and acceptance shape.

## Summary

Added BeeLlama as a selectable llama.cpp source flavor and DFlash launch support for
smaller, lower-VRAM profiles. The implementation supports a user-supplied target GGUF
and either a local DFlash draft GGUF or BeeLlama's Hugging Face draft resolver.

Default smaller-model guidance:

- Target: Q4_K_M, Q4_K_S, or IQ4_XS Qwen3.6-27B GGUF
- Draft: IQ4_XS or Q4_K_M DFlash GGUF
- KV cache: `q4_0` / `q4_0`
- Context: start at `65536`; allow user to raise/lower
- Ubatch: `512`
- GPU layers: `all` for target and draft when VRAM allows
- DFlash cross context: `512` first, `1024` optional if VRAM allows
- Reasoning: `on` with `{"preserve_thinking":true}` when the workload benefits from thinking tokens

No built-in Qwen3.6 profile should be created. Add the capability and make custom profiles
able to express the smaller-model command shape.

## Implementation Changes

### BeeLlama Build Flavor

- Add a source flavor setting with values `upstream` and `beellama`.
- Keep `upstream` as the default:
  - remote: `https://github.com/ggerganov/llama.cpp.git`
  - branch: `master`
- Add `beellama`:
  - remote: `https://github.com/Anbeeld/beellama.cpp.git`
  - branch: `main`
- Explicit remote/branch config or CLI args override the flavor defaults.
- For BeeLlama CUDA builds, include:
  - `-DGGML_CUDA=ON`
  - `-DGGML_NATIVE=ON`
  - `-DGGML_CUDA_FA=ON`
  - `-DGGML_CUDA_FA_ALL_QUANTS=ON`
- If `llama_cpp_root` already contains a git checkout with a different remote than the
  selected flavor, fail with a clear error instead of building the wrong source tree.

### DFlash Launch Schema

Extend `SpeculativeDecodingConfig`, `ServerConfig`, `SlotProfileSpec`, and profile persistence
with these fields:

- `spec_type`: allow `dflash`
- `spec_draft_model: str = ""`
- `spec_draft_hf: str = ""`
- `spec_draft_ngl: int | str = ""`
- `spec_dflash_cross_ctx: int = 0`

Validation:

- For `spec_type == "dflash"`, require exactly one of `spec_draft_model` or `spec_draft_hf`.
- Reject negative `spec_dflash_cross_ctx`.
- Preserve existing `ngram-mod` and `draft-mtp` behavior.

### Server Command Flags

When `spec_type == "dflash"`, `build_server_cmd()` must emit:

- `--spec-type dflash`
- `--spec-draft-model <path>` when `spec_draft_model` is set
- `--spec-draft-hf <repo:quant>` when `spec_draft_hf` is set
- `--spec-draft-ngl <value>` when `spec_draft_ngl` is set
- `--spec-dflash-cross-ctx <tokens>` when `spec_dflash_cross_ctx > 0`

Add typed server options needed by BeeLlama smaller-model launches:

- `kv_unified: bool = False` -> `--kv-unified`
- `mmproj_offload: bool = True` -> `--no-mmproj-offload` when false
- `mmap: bool = True` -> `--mmap` when true, `--no-mmap` when false
- `mlock: bool = False` -> `--mlock`
- `no_host_buffer: bool = False` -> `--no-host`

Keep network binding as `--host <bind_address>`. `no_host_buffer` is a separate BeeLlama
runtime flag and must not replace the listen host setting.

### TUI And Config Surfaces

- Add `dflash` to `SPEC_TYPE_CHOICES`.
- Add fields under the collapsed “Speculative decoding” section:
  - draft model path
  - draft Hugging Face repo/quant
  - draft GPU layers
  - DFlash cross context
- Add advanced toggles for:
  - unified KV
  - mmproj offload
  - mmap
  - mlock
  - no-host buffer
- Round-trip all new fields through:
  - global config defaults
  - custom slot profile modal payload
  - custom slot profile TOML store
  - profile-to-server-config resolution
  - dry-run output

## Smaller-Model Acceptance Command Shape

A custom profile should be able to dry-run into this shape with user paths:

```bash
llama-server \
  --model /models/Qwen3.6-27B-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-draft-model /models/Qwen3.6-27B-DFlash-IQ4_XS.gguf \
  --spec-draft-ngl all \
  --spec-dflash-cross-ctx 512 \
  --kv-unified \
  --n-gpu-layers all \
  --batch-size 2048 \
  --ubatch-size 512 \
  --ctx-size 65536 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --no-mmap \
  --mlock \
  --no-host \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --port 8082
```

Alternative draft resolver:

```bash
--spec-draft-hf Anbeeld/Qwen3.6-27B-DFlash-GGUF:IQ4_XS
```

## Test Plan

- Unit tests for `SpeculativeDecodingConfig`:
  - accepts `dflash`
  - rejects unknown spec types
  - requires exactly one draft source for DFlash
  - rejects negative cross context
- Unit tests for `build_server_cmd()`:
  - emits local draft DFlash flags
  - emits HF draft DFlash flags
  - emits smaller-model cache/context/runtime flags
  - preserves existing `ngram-mod` and `draft-mtp` output
- Profile tests:
  - TOML store round-trips every new field
  - TUI payload converts to `SlotProfileSpec`
  - global config defaults prefill new profile fields
- Build tests:
  - upstream flavor keeps existing CMake flags
  - BeeLlama CUDA flavor emits `GGML_CUDA_FA` and `GGML_CUDA_FA_ALL_QUANTS`
  - remote mismatch is reported clearly
- Gates:
  - `uv run pytest`
  - `uv run pre-commit run --all-files`

## Assumptions

- Smaller-model means lower-VRAM Q4/IQ4 Qwen3.6 target plus IQ4/Q4 DFlash draft.
- No model downloader is added in llm-runner; users provide file paths or BeeLlama's
  `--spec-draft-hf` value.
- BeeLlama DFlash is treated as CUDA-first for full acceleration.
- SYCL build flavor support may exist, but it is not the primary DFlash performance path.
