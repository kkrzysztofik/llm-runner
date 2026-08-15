# Profile load-mode, reasoning_preserve, and modern llama-server options

Date: 2026-08-15
Status: approved

## Goal

Parity with current llama.cpp server flags for profiles and Config defaults, plus a broader option set useful for Qwen3.8-27B-class workloads — without nested config abstractions.

## Decisions

- Scope: kitchen-sink fields (load/memory, reasoning preserve, sampling defaults, fit, ctx-checkpoints, reasoning-budget-message).
- Data model: **flat** fields (ponytail). No `LoadMemoryConfig` / `SamplingDefaultsConfig`.
- Replace `mmap`/`mlock` bools with single `load_mode` string; migrate on read; rewrite on next save.
- `reasoning_preserve` lives on profile/server/defaults as an Advanced/Reasoning field — **not** on `SpeculativeDecodingConfig` / Speculative UI.
- Template kwargs stay freeform JSON (no guided `preserve_thinking` / `reasoning_effort` helpers).
- Full Config modal `server_defaults` parity with profile fields.
- Profile modal: themed Collapsibles (Runtime, Memory, Reasoning, Sampling, Speculative).

## Data model

Remove: `mmap`, `mlock`.

Add (defaults):

| Field | Default | Notes |
|-------|---------|-------|
| `load_mode` | `"auto"` | `auto` \| `none` \| `mmap` \| `mlock` \| `mmap+mlock` \| `dio` |
| `reasoning_preserve` | `"auto"` | `auto` \| `on` \| `off` |
| `reasoning_budget_message` | `""` | omit if empty |
| `fit` | `"auto"` | `auto` \| `on` \| `off` |
| `ctx_checkpoints` | unset | omit if unset |
| `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `repeat_penalty` | unset | omit if unset |
| `no_host_buffer` | unchanged | |

Apply on `SlotProfileSpec`, `ServerConfig`, `ServerDefaultsConfig`, store I/O, and shell helper.

### Legacy mmap/mlock → load_mode

| mmap | mlock | load_mode |
|------|-------|-----------|
| true | false | `mmap` |
| true | true | `mmap+mlock` |
| false | true | `mlock` |
| false | false | `none` |
| missing | missing | `auto` |

## Command emission

In `build_server_cmd` / `_append_optional_server_flags`:

- `load_mode == auto` → omit; else `--load-mode <value>`
- Drop `--mmap` / `--no-mmap` / `--mlock`
- `reasoning_preserve`: on → `--reasoning-preserve`; off → `--no-reasoning-preserve`; auto → omit
- Non-empty `reasoning_budget_message` → `--reasoning-budget-message`
- `fit` on/off → `--fit`; auto → omit
- Set `ctx_checkpoints` → `--ctx-checkpoints N`
- Set sampling fields → `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--repeat-penalty`
- Keep `--no-host` for `no_host_buffer`

`run_opencode_models.sh`: stop hardcoding `--mmap`; follow the same rules.

Skipped until needed: `fit_target`, `fit_ctx`.

## UI

Profile modal collapsibles (collapsed by default):

1. **Runtime** — bin, port, ubatch, ngl, threads, bind, tensor split, main GPU, batch, poll, n-predict, parallel, threads-batch, cache K/V, mmproj, mmproj offload, kv-unified
2. **Memory** — load_mode Select, no-host-buffer, fit Select, ctx_checkpoints
3. **Reasoning** — mode, format, budget, reasoning_preserve, budget-message, use-jinja, chat-template-kwargs
4. **Sampling** — temp, top-k, top-p, min-p, presence, repeat (empty = unset)
5. **Speculative** — draft/MTP only (unchanged)

Config modal: same fields with full `server_defaults` parity. Select for enums; Input for numerics/strings; Checkbox for bools.

## Persistence

- Profiles: write new keys only; on read map legacy mmap/mlock; no background rewrite.
- Config: `server_defaults.load_mode` etc.; remove `server_defaults.mmap` / `.mlock`.
- Dry-run output shows new flags.

## Validation & tests

- Enums must be in allowed sets at save.
- Sampling / ctx_checkpoints: empty OK; else parse as number (checkpoints ≥ 0).
- Tests: cmd builder omit/emit matrix; store migration + round-trip; config persistence; modal payload for new sections.

## Out of scope

- Built-in Qwen3.8 preset profile
- Guided chat-template kwargs UI (`preserve_thinking`, `reasoning_effort`)
- Nested domain dataclasses for memory/sampling
- `fit_target` / `fit_ctx`
