# Qwen3.8-27B PP/TG tuning on 2×RTX 3090

Date: 2026-08-15
Status: approved

## Goal

Raise prompt-processing and token-generation throughput for `Qwen3.8-27B-UD-Q8_K_XL`
on the `qwen35` slot (2× RTX 3090, PCIe, no NVLink), then land the winning settings
as profile defaults.

## Measured baseline

Source: `~/.local/state/llm-runner/logs/llm-runner-20260815-*.log`, live agent session
at ~160k ctx.

| Metric | Mean | Range |
| ------ | ---- | ----- |
| PP (>1k tok) | 452 t/s | 315–714 |
| TG (>200 tok) | 22.5 t/s | 17–32 |
| MTP acceptance | 0.84 | mean draft len 4.5 (`n_max=7`) |

Current profile (`~/.config/llm-runner/slot_profiles.toml`): `ctx_size 262144`,
`ubatch_size 256`, `batch_size 1024`, `parallel 1`, `tensor_split "5,4"`,
`device CUDA:0,1`, `cache_type_k/v q8_0`, `spec_type draft-mtp`,
`spec_draft_n_max 7`, `spec_draft_p_min 0.75`, `poll_ms 0`.

## Architecture facts driving the design

`Qwen3.8-27B` is a **hybrid**, not a dense transformer. Of 65 GGUF blocks:

- 48 SSM / linear-attention layers (`ssm_conv1d`, `ssm_a`, `ssm_beta`, `ssm_dt`)
- 16 full-attention layers (`attn_k`, `attn_v`; 4 KV heads × 256 dim)
- 1 MTP / next-token-prediction head (`blk.64.nextn.*`)

Consequences:

- KV is ~34 KB/token at q8_0 (16 layers only) → 262144 ctx ≈ **8.9 GB**.
  31.4 GB weights + 8.9 GB KV ≈ 40 GB of 48 GB, leaving ~6 GB of headroom.
- PP is dominated by dense FFN and SSM scan, both batch-hungry → `ubatch 256` is
  the primary bottleneck.
- SSM recurrent state cannot be KV-shifted or rolled back, so mid-context cache
  reuse depends entirely on `--ctx-checkpoints` (default 32) and
  `--checkpoint-min-step` (default 8192). At 262k ctx that is exactly 32 — on the
  boundary.

## Blocking defects in the command builder

`src/llama_manager/validation/commands/builder.py`:

1. `ngram-mod` emits `--spec-ngram-size-n`, `--draft-min`, `--draft-max`; this
   llama.cpp build reports all three as removed (replacements:
   `--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-min`, `--spec-ngram-mod-n-max`).
   `spec_type = "ngram-mod"` therefore cannot launch.
2. `_SPEC_TYPE_DFLASH = "dflash"`; the build's `--spec-type` enum is `draft-dflash`.
3. `--split-mode layer` is hardcoded (line 129), so `row` and `none` are untestable.
4. `--spec-type` accepts a comma-separated list; `spec_decode` validation rejects
   combinations such as `draft-mtp,ngram-mod`.

These are fixed first — a sweep on a builder that cannot emit the flags under test
produces nothing.

## Sweep order

Each stage is measured against a fixed replayed agent prompt at ~160k ctx, PP and TG
recorded separately, winner carried into the next stage.

1. `ubatch_size` 256 → 512 → 1024 → 2048, `batch_size` 1024 → 4096
2. `split_mode` layer vs row (vs none where it fits)
3. `spec_draft_p_min` 0.75 → 0.6 → 0.5 × `spec_draft_n_max` 7 → 10
4. `spec_type` `draft-mtp` vs `draft-mtp,ngram-mod`
5. `ctx_checkpoints` / `checkpoint_min_step` against a mid-context divergence replay

`UD-Q6_K_XL` is deferred: it is only worth downloading if the Q8 sweep plateaus, and
it then requires a quality gate (`llama-perplexity --kl-divergence` against Q8 as
reference) rather than a speed number alone.

## Landing

Winning values become `qwen35` profile defaults in
`llama_manager/config/builder.py`, with the measured before/after recorded here.

## Out of scope

- New abstractions for sweeping; the existing `llama_manager/benchmark` module and
  log parsing cover measurement.
- Vision / `mmproj` tuning (`mmproj` is empty on this profile).
- The Intel SYCL slots.
