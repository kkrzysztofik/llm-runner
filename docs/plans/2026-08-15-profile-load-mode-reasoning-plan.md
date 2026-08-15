# Profile load-mode & modern llama-server options — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deprecated mmap/mlock with `load_mode`, add flat profile/Config fields for reasoning_preserve and other modern llama-server options, emit the correct CLI flags, and restructure the profile modal into themed collapsibles.

**Architecture:** Flat fields on `SlotProfileSpec` / `ServerConfig` / `ServerDefaultsConfig` (no new nested dataclasses). One small `resolve_load_mode()` helper for legacy migration. `build_server_cmd` omits auto/unset flags. UI themed Collapsibles read/write the same flat fields. `reasoning_preserve` is Advanced/Reasoning — not on `SpeculativeDecodingConfig`.

**Tech Stack:** Python 3.12, existing dataclasses, Textual modals, pytest, ruff/pyright.

**Design:** `docs/plans/2026-08-15-profile-load-mode-reasoning-design.md`

## Global Constraints

- No nested `LoadMemoryConfig` / `SamplingDefaultsConfig`.
- Do not put `reasoning_preserve` on `SpeculativeDecodingConfig`.
- No `fit_target` / `fit_ctx`, no guided chat-template kwargs UI, no built-in Qwen3.8 preset.
- `auto` / unset ⇒ omit CLI flag.
- Allowed `load_mode`: `auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`.
- Allowed `reasoning_preserve` / `fit`: `auto`, `on`, `off`.
- Agent gate before commit: `uv run pre-commit run --all-files` and `uv run pytest`.
- Prefer revising existing files; do not delete files without permission.

## File map

| File | Role |
|------|------|
| `src/llama_manager/config/load_mode.py` | `LOAD_MODE_VALUES`, `resolve_load_mode(data) -> str` |
| `src/llama_manager/config/server.py` | Flat fields on `ServerConfig` |
| `src/llama_manager/config/profiles.py` | Flat fields on `SlotProfileSpec` |
| `src/llama_manager/config/defaults.py` | Flat fields on `ServerDefaultsConfig` |
| `src/llama_manager/config/persistence.py` | Config key list mmap→load_mode + new keys |
| `src/llama_manager/config/builder.py` | Profile→ServerConfig field mapping |
| `src/llama_manager/slot_profile_store.py` | Serialize/deserialize + migration |
| `src/llama_manager/common/profile_io.py` | TOML export lines |
| `src/llama_manager/validation/commands/builder.py` | CLI flag emission |
| `src/llama_cli/tui/components/form_widgets.py` | Choice tuples + defaults helpers |
| `src/llama_cli/tui/components/slot_profile_modal.py` | Themed collapsibles + payload |
| `src/llama_cli/tui/components/config_modal.py` | Config defaults parity |
| `run_opencode_models.sh` | Stop hardcoding `--mmap` |
| Tests under `src/tests/` | Builder, store, persistence, modals |

---

### Task 1: `resolve_load_mode` + replace mmap/mlock on models + cmd builder

**Files:**
- Create: `src/llama_manager/config/load_mode.py`
- Modify: `src/llama_manager/config/server.py`, `profiles.py`, `defaults.py`
- Modify: `src/llama_manager/validation/commands/builder.py` (`_append_optional_server_flags`)
- Modify: `src/llama_manager/config/builder.py` (mmap/mlock → load_mode)
- Test: `src/tests/config/test_load_mode.py` (new), `src/tests/server/test_server.py`

**Interfaces:**
- Produces: `LOAD_MODE_VALUES: frozenset[str]`, `resolve_load_mode(data: dict[str, object]) -> str`, `ServerConfig.load_mode: str = "auto"` (and same on profile/defaults); remove `mmap`/`mlock`.

- [ ] **Step 1: Write failing tests for migration + cmd emission**

```python
# src/tests/config/test_load_mode.py
from llama_manager.config.load_mode import resolve_load_mode


def test_resolve_explicit_load_mode() -> None:
    assert resolve_load_mode({"load_mode": "dio"}) == "dio"


def test_resolve_legacy_mmap_mlock() -> None:
    assert resolve_load_mode({"mmap": True, "mlock": False}) == "mmap"
    assert resolve_load_mode({"mmap": True, "mlock": True}) == "mmap+mlock"
    assert resolve_load_mode({"mmap": False, "mlock": True}) == "mlock"
    assert resolve_load_mode({"mmap": False, "mlock": False}) == "none"


def test_resolve_missing_defaults_auto() -> None:
    assert resolve_load_mode({}) == "auto"
```

Replace mmap/mlock tests in `test_server.py` with:

```python
def test_load_mode_auto_omits_flag(self) -> None:
    cmd = build_server_cmd(self._minimal_cfg(load_mode="auto"))
    assert "--load-mode" not in cmd
    assert "--mmap" not in cmd
    assert "--mlock" not in cmd


def test_load_mode_mmap_emits(self) -> None:
    cmd = build_server_cmd(self._minimal_cfg(load_mode="mmap"))
    i = cmd.index("--load-mode")
    assert cmd[i + 1] == "mmap"
```

- [ ] **Step 2: Run tests — expect FAIL** (`load_mode` module / kwargs missing)

```bash
uv run pytest src/tests/config/test_load_mode.py src/tests/server/test_server.py -k "load_mode or mmap or mlock" -v
```

- [ ] **Step 3: Implement `load_mode.py` + model fields + builder emission**

```python
# src/llama_manager/config/load_mode.py
LOAD_MODE_VALUES = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})


def resolve_load_mode(data: dict[str, object]) -> str:
    raw = data.get("load_mode")
    if isinstance(raw, str) and raw in LOAD_MODE_VALUES:
        return raw
    if "mmap" not in data and "mlock" not in data:
        return "auto"
    mmap = bool(data.get("mmap", True))
    mlock = bool(data.get("mlock", False))
    if mmap and mlock:
        return "mmap+mlock"
    if mmap:
        return "mmap"
    if mlock:
        return "mlock"
    return "none"
```

On `ServerConfig` / `SlotProfileSpec` / `ServerDefaultsConfig`: delete `mmap`/`mlock`; add `load_mode: str = "auto"`. Update `__init__` kwargs accordingly.

In `_append_optional_server_flags`:

```python
if cfg.load_mode and cfg.load_mode != "auto":
    cmd.extend(["--load-mode", cfg.load_mode])
# remove mmap/mlock branches
if cfg.no_host_buffer:
    cmd.append("--no-host")
```

Update `config/builder.py` profile→server mapping to pass `load_mode`.

Export `resolve_load_mode` from `llama_manager.config` if that package re-exports helpers (match local style).

- [ ] **Step 4: Run tests — expect PASS** for Task 1 scope

```bash
uv run pytest src/tests/config/test_load_mode.py src/tests/server/test_server.py -k "load_mode or no_host" -v
```

- [ ] **Step 5: Commit** (only if user asked)

```bash
git add src/llama_manager/config/load_mode.py src/llama_manager/config/server.py \
  src/llama_manager/config/profiles.py src/llama_manager/config/defaults.py \
  src/llama_manager/config/builder.py src/llama_manager/validation/commands/builder.py \
  src/tests/config/test_load_mode.py src/tests/server/test_server.py
git commit -m "$(cat <<'EOF'
feat: replace mmap/mlock with load_mode for llama-server

EOF
)"
```

---

### Task 2: reasoning_preserve, budget_message, fit, ctx_checkpoints, sampling on models + cmd

**Files:**
- Modify: `server.py`, `profiles.py`, `defaults.py`, `config/builder.py`, `validation/commands/builder.py`
- Test: `src/tests/server/test_server.py`

**Interfaces:**
- Produces flat fields:
  - `reasoning_preserve: str = "auto"`
  - `reasoning_budget_message: str = ""`
  - `fit: str = "auto"`
  - `ctx_checkpoints: int | None = None`
  - `temperature: float | None = None`
  - `top_k: int | None = None`
  - `top_p: float | None = None`
  - `min_p: float | None = None`
  - `presence_penalty: float | None = None`
  - `repeat_penalty: float | None = None`

- [ ] **Step 1: Write failing cmd tests**

```python
def test_reasoning_preserve_on(self) -> None:
    cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="on"))
    assert "--reasoning-preserve" in cmd


def test_reasoning_preserve_off(self) -> None:
    cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="off"))
    assert "--no-reasoning-preserve" in cmd


def test_reasoning_preserve_auto_omits(self) -> None:
    cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="auto"))
    assert "--reasoning-preserve" not in cmd
    assert "--no-reasoning-preserve" not in cmd


def test_fit_and_sampling_emit_when_set(self) -> None:
    cmd = build_server_cmd(
        self._minimal_cfg(
            fit="off",
            ctx_checkpoints=64,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            presence_penalty=0.0,
            repeat_penalty=1.0,
            reasoning_budget_message="stop thinking",
        )
    )
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "64"
    assert cmd[cmd.index("--temp") + 1] == "1.0"
    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert "--reasoning-budget-message" in cmd
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest src/tests/server/test_server.py -k "reasoning_preserve or fit_and_sampling" -v
```

- [ ] **Step 3: Add fields + emission**

```python
# in _append_optional_server_flags
if cfg.reasoning_preserve == "on":
    cmd.append("--reasoning-preserve")
elif cfg.reasoning_preserve == "off":
    cmd.append("--no-reasoning-preserve")
if cfg.reasoning_budget_message:
    cmd.extend(["--reasoning-budget-message", cfg.reasoning_budget_message])
if cfg.fit in ("on", "off"):
    cmd.extend(["--fit", cfg.fit])
if cfg.ctx_checkpoints is not None:
    cmd.extend(["--ctx-checkpoints", str(cfg.ctx_checkpoints)])
if cfg.temperature is not None:
    cmd.extend(["--temp", str(cfg.temperature)])
if cfg.top_k is not None:
    cmd.extend(["--top-k", str(cfg.top_k)])
if cfg.top_p is not None:
    cmd.extend(["--top-p", str(cfg.top_p)])
if cfg.min_p is not None:
    cmd.extend(["--min-p", str(cfg.min_p)])
if cfg.presence_penalty is not None:
    cmd.extend(["--presence-penalty", str(cfg.presence_penalty)])
if cfg.repeat_penalty is not None:
    cmd.extend(["--repeat-penalty", str(cfg.repeat_penalty)])
```

Wire the same fields through `SlotProfileSpec`, `ServerDefaultsConfig`, and `config/builder.py`.

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest src/tests/server/test_server.py -k "reasoning_preserve or fit_and_sampling or load_mode" -v
```

- [ ] **Step 5: Commit** (if requested)

---

### Task 3: Persistence — profile store, profile_io, config keys

**Files:**
- Modify: `src/llama_manager/slot_profile_store.py`, `common/profile_io.py`, `config/persistence.py`
- Test: `src/tests/config/test_slot_profile_store.py`, config persistence tests if present

**Interfaces:**
- Consumes: `resolve_load_mode`
- `_profile_to_dict` writes `load_mode` + new fields; never `mmap`/`mlock`
- `_profile_from_dict` calls `resolve_load_mode(data)` and reads new fields with defaults

- [ ] **Step 1: Failing store migration test**

```python
def test_profile_from_dict_migrates_mmap_mlock() -> None:
    p = _profile_from_dict(
        {
            "profile_id": "t",
            "model": "/m.gguf",
            "alias": "t",
            "device": "cuda:0",
            "port": 8080,
            "ctx_size": 4096,
            "ubatch_size": 512,
            "threads": 8,
            "mmap": False,
            "mlock": True,
        }
    )
    assert p.load_mode == "mlock"
    d = _profile_to_dict(p)
    assert d["load_mode"] == "mlock"
    assert "mmap" not in d and "mlock" not in d
```

(Import `_profile_from_dict` / `_profile_to_dict` the same way existing store tests do — if private, test via public save/load round-trip.)

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Update store + profile_io + persistence key lists**

- Replace mmap/mlock serialization with `load_mode` and new keys.
- `persistence.py`: remove `server_defaults.mmap` / `.mlock`; add `server_defaults.load_mode`, `reasoning_preserve`, `reasoning_budget_message`, `fit`, `ctx_checkpoints`, sampling keys.
- Config load path: use `resolve_load_mode` when reading server_defaults blob if legacy keys appear.

- [ ] **Step 4: Run store + related tests — expect PASS**

```bash
uv run pytest src/tests/config/test_slot_profile_store.py -v
```

- [ ] **Step 5: Commit** (if requested)

---

### Task 4: form_widgets choices + Config modal parity

**Files:**
- Modify: `src/llama_cli/tui/components/form_widgets.py`, `config_modal.py`
- Test: `src/tests/tui/test_config_modal.py`

**Interfaces:**
- Produces:

```python
LOAD_MODE_CHOICES = (
    ("auto", "auto"),
    ("none", "none"),
    ("mmap", "mmap"),
    ("mlock", "mlock"),
    ("mmap+mlock", "mmap+mlock"),
    ("dio", "dio"),
)
REASONING_PRESERVE_CHOICES = (("auto", "auto"), ("on", "on"), ("off", "off"))
FIT_CHOICES = (("auto", "auto"), ("on", "on"), ("off", "off"))
```

- Config payload fields mirror `ServerDefaultsConfig` new flats; `to_overrides()` / apply path updated; drop mmap/mlock checkboxes.

- [ ] **Step 1: Extend config modal tests for load_mode + reasoning_preserve defaults**

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Wire Selects/Inputs in config modal (group near existing server defaults); validate enums on save**

- [ ] **Step 4: Run config modal tests — PASS**

```bash
uv run pytest src/tests/tui/test_config_modal.py -v
```

- [ ] **Step 5: Commit** (if requested)

---

### Task 5: Profile modal themed collapsibles

**Files:**
- Modify: `src/llama_cli/tui/components/slot_profile_modal.py`
- Test: `src/tests/tui/test_slot_profile_modal.py`

**Interfaces:**
- Replace `_build_advanced_fields` with builders:
  - `_build_runtime_fields`
  - `_build_memory_fields` (load_mode, no_host_buffer, fit, ctx_checkpoints)
  - `_build_reasoning_fields` (mode, format, budget, reasoning_preserve, budget_message, jinja, chat_template_kwargs)
  - `_build_sampling_fields`
  - keep `_build_speculative_fields` (no preserve)
- `SlotProfileFormPayload` + collect/prefill/save mapping include all new flats.
- Empty sampling/ctx inputs → `None` / omit; invalid enum → reject save (same pattern as chat_template_kwargs validation).

- [ ] **Step 1: Tests for Memory/Reasoning payload fields and absence of mmap/mlock checkboxes**

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement collapsibles + payload wiring**

- [ ] **Step 4:**

```bash
uv run pytest src/tests/tui/test_slot_profile_modal.py -v
```

- [ ] **Step 5: Commit** (if requested)

---

### Task 6: Shell helper + dry-run display + full gate

**Files:**
- Modify: `run_opencode_models.sh` (remove hardcoded `--mmap`; optional `--load-mode` if you thread a variable — default omit for auto)
- Modify: dry-run printers if they list mmap (`src/llama_cli/commands/dry_run.py` or equivalent)
- Sweep remaining `mmap`/`mlock` references in `src/` (tests, docs in ARCHITECTURE only if you touch docs — prefer code/tests only unless asked)

- [ ] **Step 1: Grep for leftover mmap/mlock profile fields**

```bash
rg -n 'mmap|mlock|--no-mmap' src run_opencode_models.sh
```

Expected: only migration helper / comments / historical notes, no `cfg.mmap`.

- [ ] **Step 2: Fix leftovers**

- [ ] **Step 3: Full gate**

```bash
uv run pre-commit run --all-files
uv run pytest
```

Expected: both green.

- [ ] **Step 4: Commit** (if requested)

---

## Self-review

1. **Spec coverage:** load_mode migration ✓; reasoning_preserve not on spec_decode ✓; fit/ctx/sampling/budget-message ✓; themed UI ✓; Config parity ✓; shell ✓; fit_target/fit_ctx out of scope ✓.
2. **Placeholders:** none intentional — Task 4/5 tests should copy the concrete assertion style from existing modal tests in-repo when implementing.
3. **Types:** `load_mode`/`fit`/`reasoning_preserve` are `str`; sampling and `ctx_checkpoints` are `None`-able numerics.
