# Qwen 3.8 thinking levels (reasoning_effort)

Date: 2026-08-17
Status: approved

## Goal

Expose Qwen 3.8 thinking levels as a first-class **Thinking level** Select on run profiles and Config `server_defaults`, merged into a single `--chat-template-kwargs` object at launch. Default is `medium`.

This revisits the 2026-08-15 load-mode/reasoning design, which deferred guided `reasoning_effort` UI.

## Decisions

- Dedicated stored field `reasoning_effort` (`xhigh` | `medium` | `low`), default `medium`.
- Live on `LaunchRuntimeFields` (same ponytail as `reasoning_preserve`). Not on `SpeculativeDecodingConfig`.
- Merge into `chat_template_kwargs` at command emission. llama-server does not merge duplicate `--chat-template-kwargs` flags.
- Conflict is an error: if the JSON already contains `reasoning_effort`, reject save and refuse `build_server_cmd`.
- Always emit the key, including for summary profiles with `enable_thinking: false` (templates ignore steering when thinking is off).
- Official-template-safe values only. No `high` alias, no omit, no thinking-off option.
- Full parity: profile field, Config `server_defaults`, TUI Selects, command builder, dry-run, and `run_opencode_models.sh`.
- No new CLI `--thinking-level` flag. Launch modes use the profile/default field.

## Why merge into kwargs

Qwen 3.8 thinking levels are Jinja prompt steering, not a llama.cpp CLI flag. The official chat template injects a system-message instruction for `xhigh` / `low`, and nothing for `medium`. Official templates raise a fatal exception on any other value (including OpenAI's `high`). Community A/B on Qwen3.8-27B showed stock `xhigh` averaging ~30k reasoning characters and truncating outputs, which is why the default here is `medium`.

Server-launch kwargs matter: some clients (OpenCode) drop per-request `chat_template_kwargs`.

## Data model

Add to `LaunchRuntimeFields`:

| Field | Default | Notes |
|-------|---------|-------|
| `reasoning_effort` | `"medium"` | `xhigh` \| `medium` \| `low` |

Applies on `SlotProfileSpec`, `ServerConfig`, and `ServerDefaultsConfig` via the existing mixin. Persist as a top-level profile/config key (same flattening as `reasoning_preserve`).

Invalid persisted values (`"high"`, `"auto"`, garbage) normalize to `medium` at read/builder boundary.

## Command emission

In `_append_optional_server_flags` / the shell `build_server_cmd`:

1. Parse `chat_template_kwargs` as a JSON object (`{}` if empty).
2. If the object contains `reasoning_effort`, raise / exit with an error.
3. Set `reasoning_effort` from the dedicated field.
4. Emit one `--chat-template-kwargs '<json>'`.

Example: `'{"preserve_thinking":true}'` + `medium` becomes
`'{"preserve_thinking":true,"reasoning_effort":"medium"}'`.

## UI

Reasoning collapsible (profile modal and Config `server_defaults`):

- Add **Thinking level** Select (`xhigh` / `medium` / `low`) immediately above Chat template kwargs.
- Use existing `profile-select` / `config-select` styling.
- Dry-run prints `Thinking level: <value>`.

## Validation

Save (profile modal, Config modal, controller):

- Select value must be one of the three strings.
- `chat_template_kwargs` must remain valid JSON (existing check).
- If the JSON object contains `reasoning_effort`, reject with: remove it from JSON and use the Thinking level Select.

Launch (`build_server_cmd`): same JSON conflict is a hard error so hand-edited TOML cannot split-brain.

## Shell

`run_opencode_models.sh`:

- `DEFAULT_REASONING_EFFORT=medium`
- Merge in `build_server_cmd` the same way Python does.
- qwen35 / qwen27b keep `preserve_thinking` and gain `reasoning_effort`.

## Testing

Unit tests only. No GPU, no subprocess llama-server.

- Command builder: default merge, preserve existing keys, all three values, JSON conflict raises, invalid JSON still fails the existing path.
- Persistence: profile store round-trip; invalid persisted value → `medium`; Config default is `medium`.
- TUI: payload includes the Select; save rejects JSON that contains `reasoning_effort`; prefill uses `medium` when unset.
- Dry-run: printed line includes the thinking level.

## Out of scope

- `high` alias (official template crash)
- Omit / template-default option
- Thinking off (`enable_thinking: false` helper)
- Per-request live toggle without restart
- Dedicated Qwen 3.8 preset profile (the `qwen35` slot already runs Qwen3.8-27B)
- CLI `--thinking-level` override
- Guided helpers for other kwargs (`preserve_thinking`, `enable_thinking`)
