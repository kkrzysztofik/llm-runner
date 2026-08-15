# Qwen3.8-27B PP/TG Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the four command-builder defects that block PP/TG experiments, add a repeatable sweep harness, then land the winning `ubatch`/`split-mode`/MTP settings as `qwen35` profile defaults.

**Architecture:** Phase 1 is TDD bug-fixing inside `llama_manager/validation/commands/builder.py` plus two new `LaunchRuntimeFields` (`split_mode`, `checkpoint_min_step`), following the existing `ctx_checkpoints` field as the template for every touch point. Phase 2 is a single script that builds real commands through `resolve_profile_config` + `build_server_cmd`, launches each variant, replays one fixed long prompt, and records `timings` from the `/completion` response. Phase 3 records results and updates profile defaults.

**Tech Stack:** Python 3.12, `uv`, pytest, dataclasses, llama.cpp `llama-server` (build 879, commit `adb55e5`), CUDA on 2× RTX 3090.

**Design doc:** `docs/plans/2026-08-15-qwen38-pp-tg-tuning-design.md`

---

## Background the executor needs

`Qwen3.8-27B` is a **hybrid** model: 48 SSM/linear layers, 16 full-attention layers,
1 MTP head. Do not reason about it as a dense transformer. KV is ~34 KB/token at
q8_0, so the configured `ctx_size = 262144` genuinely fits in 48 GB alongside
31.4 GB of weights.

Measured baseline to beat (from `~/.local/state/llm-runner/logs/llm-runner-20260815-*.log`):

- PP: **452 t/s** mean (315–714)
- TG: **22.5 t/s** mean (17–32)
- MTP acceptance 0.84, mean draft length 4.5

The live profile lives in `~/.config/llm-runner/slot_profiles.toml` under
`profile_id = "qwen35"` and points at `Qwen3.8-27B-UD-Q8_K_XL.gguf`. The
`qwen35` name is historical — do not rename it in this plan.

**Flag mapping confirmed against `llama-server --help` in this build:**

| Config field | Old (removed) flag | Correct flag |
| ------------ | ------------------ | ------------ |
| `spec_ngram_size_n` | `--spec-ngram-size-n` | `--spec-ngram-mod-n-match` |
| `draft_min` | `--draft-min` | `--spec-ngram-mod-n-min` |
| `draft_max` | `--draft-max` | `--spec-ngram-mod-n-max` |

The existing defaults (24 / 48 / 64) match this build's defaults for
`--spec-ngram-mod-n-match` / `-n-min` / `-n-max` exactly, which confirms the mapping.

---

## Task 1: Fix `ngram-mod` removed-flag emission

**Files:**
- Modify: `src/llama_manager/validation/commands/builder.py:243-255`
- Test: `src/tests/server/test_server.py:258` (existing `test_ngram_spec_flags`)

**Step 1: Update the existing test to assert the current flag names**

Replace the body of `test_ngram_spec_flags` so it asserts the new flags and
explicitly asserts the removed ones are absent:

```python
    def test_ngram_spec_flags(self) -> None:
        cmd = build_server_cmd(
            make_server_config(
                spec_type="ngram-mod",
                spec_ngram_size_n=12,
                draft_min=8,
                draft_max=32,
            )
        )
        assert "ngram-mod" in cmd
        assert cmd[cmd.index("--spec-ngram-mod-n-match") + 1] == "12"
        assert cmd[cmd.index("--spec-ngram-mod-n-min") + 1] == "8"
        assert cmd[cmd.index("--spec-ngram-mod-n-max") + 1] == "32"
        assert "--spec-ngram-size-n" not in cmd
        assert "--draft-min" not in cmd
        assert "--draft-max" not in cmd
```

Keep the surrounding `make_server_config` helper call shape used by the existing
test in that file — read lines 250-275 first and match it.

**Step 2: Run the test to verify it fails**

Run: `uv run pytest src/tests/server/test_server.py::TestBuildServerCmd::test_ngram_spec_flags -v`
Expected: FAIL with `ValueError: '--spec-ngram-mod-n-match' is not in list`

**Step 3: Fix the emission**

In `builder.py`, replace `_append_ngram_speculative_flags`:

```python
def _append_ngram_speculative_flags(cmd: list[str], spec: Any) -> None:
    cmd.extend(
        [
            "--spec-ngram-mod-n-match",
            str(spec.spec_ngram_size_n),
            "--spec-ngram-mod-n-min",
            str(spec.draft_min),
            "--spec-ngram-mod-n-max",
            str(spec.draft_max),
        ]
    )
```

Note the `--spec-type` pair is removed from this function — Task 3 emits it once
for all types. Until Task 3 lands, the test above will fail on `"ngram-mod" in cmd`,
so for this task keep `_SPEC_TYPE_FLAG, _SPEC_TYPE_NGRAM_MOD,` as the first two
entries of the list and remove them in Task 3.

**Step 4: Run the test to verify it passes**

Run: `uv run pytest src/tests/server/test_server.py -v -k ngram`
Expected: PASS

**Step 5: Commit**

```bash
rtk git add src/llama_manager/validation/commands/builder.py src/tests/server/test_server.py
rtk git commit -m "fix: emit current ngram-mod spec flags"
```

---

## Task 2: Fix the dflash spec-type value

**Files:**
- Modify: `src/llama_manager/validation/commands/builder.py:23`
- Modify: `src/llama_manager/config/spec_decode.py:69-72`
- Test: `src/tests/config/test_spec_decode.py`

**Step 1: Write the failing test**

Add to `src/tests/config/test_spec_decode.py`:

```python
def test_dflash_spec_type_uses_build_enum_name() -> None:
    config = SpeculativeDecodingConfig(
        spec_type="draft-dflash",
        spec_draft_model="/models/draft.gguf",
    )
    assert config.spec_type == "draft-dflash"


def test_legacy_dflash_spec_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="spec_type"):
        SpeculativeDecodingConfig(spec_type="dflash", spec_draft_model="/m.gguf")
```

**Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/config/test_spec_decode.py -v -k dflash`
Expected: FAIL — `"draft-dflash"` raises `ValueError`

**Step 3: Rename the value**

In `spec_decode.py`, change the allowed tuple and both `"dflash"` comparisons to
`"draft-dflash"`:

```python
    if config.spec_type not in ("", "ngram-mod", "draft-mtp", "draft-dflash"):
        raise ValueError("spec_type must be '', 'ngram-mod', 'draft-mtp', or 'draft-dflash'")
    if config.spec_type == "draft-dflash":
        _validate_dflash_config(config)
```

In `builder.py`, change `_SPEC_TYPE_DFLASH: Final = "draft-dflash"`.

**Step 4: Run the full spec-decode suite**

Run: `uv run pytest src/tests/config/test_spec_decode.py -v`
Expected: PASS. If other tests in the repo still pass `"dflash"`, update them — per
`AGENTS.md` there is no back-compat requirement, so do not add a migration shim.

**Step 5: Commit**

```bash
rtk git add -A src/
rtk git commit -m "fix: use draft-dflash spec type name from llama.cpp"
```

---

## Task 3: Allow comma-separated spec-type combinations

**Files:**
- Modify: `src/llama_manager/config/spec_decode.py:69-70`
- Modify: `src/llama_manager/validation/commands/builder.py:229-241`
- Test: `src/tests/config/test_spec_decode.py`, `src/tests/server/test_server.py`

**Step 1: Write the failing tests**

In `test_spec_decode.py`:

```python
def test_comma_separated_spec_types_are_accepted() -> None:
    config = SpeculativeDecodingConfig(spec_type="draft-mtp,ngram-mod")
    assert config.spec_type == "draft-mtp,ngram-mod"


def test_unknown_member_of_spec_type_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="spec_type"):
        SpeculativeDecodingConfig(spec_type="draft-mtp,bogus")
```

In `test_server.py`:

```python
    def test_combined_spec_types_emit_both_flag_groups(self) -> None:
        cmd = build_server_cmd(
            make_server_config(
                spec_type="draft-mtp,ngram-mod",
                spec_draft_n_max=8,
                spec_ngram_size_n=12,
                draft_min=8,
                draft_max=32,
            )
        )
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp,ngram-mod"
        assert cmd.count("--spec-type") == 1
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "8"
        assert cmd[cmd.index("--spec-ngram-mod-n-match") + 1] == "12"
```

**Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/config/test_spec_decode.py src/tests/server/test_server.py -v -k "spec_type or combined"`
Expected: FAIL — `ValueError` on the comma value

**Step 3: Validate per member**

In `spec_decode.py`, replace the membership check:

```python
_VALID_SPEC_TYPES: frozenset[str] = frozenset(
    {"ngram-mod", "draft-mtp", "draft-dflash"}
)


def _spec_type_members(spec_type: str) -> list[str]:
    """Split a comma-separated ``--spec-type`` value into its members."""
    return [part.strip() for part in spec_type.split(",") if part.strip()]
```

and in `_validate_speculative_decoding`:

```python
    members = _spec_type_members(config.spec_type)
    if config.spec_type and not members:
        raise ValueError("spec_type must not be blank")
    for member in members:
        if member not in _VALID_SPEC_TYPES:
            raise ValueError(
                f"spec_type members must be one of {sorted(_VALID_SPEC_TYPES)}, got: {member}"
            )
    if "draft-dflash" in members:
        _validate_dflash_config(config)
```

Export `_spec_type_members` as `spec_type_members` (no leading underscore) so the
command builder can import it.

**Step 4: Emit `--spec-type` once, then each member's flags**

In `builder.py`, replace `_append_speculative_flags`:

```python
def _append_speculative_flags(cmd: list[str], cfg: ServerConfig) -> None:
    """Append llama-server speculative decoding flags when configured."""
    spec = cfg.spec_decode
    members = spec_type_members(spec.spec_type)
    if not members:
        return
    cmd.extend([_SPEC_TYPE_FLAG, spec.spec_type])
    if _SPEC_TYPE_NGRAM_MOD in members:
        _append_ngram_speculative_flags(cmd, spec)
    if _SPEC_TYPE_DRAFT_MTP in members:
        _append_draft_mtp_flags(cmd, spec)
    if _SPEC_TYPE_DFLASH in members:
        _append_dflash_flags(cmd, spec)
```

Then remove the now-duplicated `_SPEC_TYPE_FLAG, <type>,` entries from the head of
`_append_ngram_speculative_flags`, `_append_draft_mtp_flags`, and
`_append_dflash_flags`.

**Step 5: Run the full builder suite**

Run: `uv run pytest src/tests/server/test_server.py src/tests/config/test_spec_decode.py -v`
Expected: PASS

**Step 6: Commit**

```bash
rtk git add -A src/
rtk git commit -m "feat: support comma-separated spec-type combinations"
```

---

## Task 4: Stop emitting `--spec-draft-n-max 0`

**Files:**
- Modify: `src/llama_manager/validation/commands/builder.py:258-261`
- Test: `src/tests/server/test_server.py`

**Step 1: Write the failing test**

```python
    def test_unset_spec_draft_n_max_is_omitted(self) -> None:
        cmd = build_server_cmd(make_server_config(spec_type="draft-mtp"))
        assert "--spec-draft-n-max" not in cmd
```

**Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/server/test_server.py -v -k n_max`
Expected: FAIL — the flag is present with value `0`

**Step 3: Guard the emission**

```python
def _append_draft_mtp_flags(cmd: list[str], spec: Any) -> None:
    if spec.spec_draft_n_max > 0:
        cmd.extend(["--spec-draft-n-max", str(spec.spec_draft_n_max)])
    if spec.spec_draft_p_min > 0:
        cmd.extend(["--spec-draft-p-min", str(spec.spec_draft_p_min)])
    ...
```

**Step 4: Run tests**

Run: `uv run pytest src/tests/server/test_server.py -v`
Expected: PASS

**Step 5: Commit**

```bash
rtk git add -A src/
rtk git commit -m "fix: omit zero spec-draft-n-max"
```

---

## Task 5: Add `split_mode` as a launch-runtime field

`--split-mode layer` is hardcoded at `builder.py:129`, so `row` cannot be tested.
Follow `ctx_checkpoints` as the template — it is already wired through every layer.

**Files:**
- Modify: `src/llama_manager/config/launch_runtime.py:13-46` (both `LaunchRuntimeOverrides` and `LaunchRuntimeFields`)
- Modify: `src/llama_manager/validation/commands/builder.py:129-130`
- Modify: `src/llama_manager/config/server.py` (validation, near line 178)
- Modify: `src/llama_manager/config/builder.py:163` and `:230` and `:622`
- Modify: `src/llama_manager/config/persistence.py:88`, `:264`, `:297`
- Modify: `src/llama_manager/slot_profile_store.py:162`, `:224`
- Modify: `src/llama_manager/common/profile_io.py:132`
- Modify: `src/llama_cli/tui/components/slot_profile_modal.py` and `config_modal.py` and `form_widgets.py` (mirror every `ctx_checkpoints` occurrence)
- Test: `src/tests/server/test_server.py`, `src/tests/config/test_config_builders.py`

**Step 1: Write the failing tests**

```python
    def test_split_mode_defaults_to_layer(self) -> None:
        cmd = build_server_cmd(make_server_config())
        assert cmd[cmd.index("--split-mode") + 1] == "layer"

    def test_split_mode_override_is_emitted(self) -> None:
        cmd = build_server_cmd(make_server_config(split_mode="row"))
        assert cmd[cmd.index("--split-mode") + 1] == "row"
        assert cmd.count("--split-mode") == 1

    def test_invalid_split_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="split_mode"):
            make_server_config(split_mode="sideways")
```

**Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/server/test_server.py -v -k split_mode`
Expected: FAIL — `TypeError: unexpected keyword argument 'split_mode'`

**Step 3: Add the field**

In `launch_runtime.py`, add to `LaunchRuntimeOverrides`:

```python
    split_mode: str | None
```

and to `LaunchRuntimeFields`:

```python
    split_mode: str = field(default="layer", kw_only=True)
```

In `config/server.py`, next to the `ctx_checkpoints` check:

```python
        if self.split_mode not in ("none", "layer", "row", "tensor"):
            raise ValueError(f"split_mode must be none/layer/row/tensor, got: {self.split_mode}")
```

In `validation/commands/builder.py`, replace the hardcoded pair in the `cmd` list:

```python
        "--split-mode",
        cfg.split_mode,
```

**Step 4: Wire the remaining layers**

Grep for the template field and mirror every hit:

```bash
/usr/bin/grep -rn "ctx_checkpoints" src/ --include=*.py | /usr/bin/grep -v "^src/tests"
```

`split_mode` is a plain `str` with a non-empty default, so use the `load_mode`
pattern (always present) rather than the `ctx_checkpoints` optional-int pattern for
TOML round-tripping and the TUI Input widgets.

**Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (existing dry-run snapshot tests in `src/tests/cli/test_dry_run_schema.py` may need their expected command lists regenerated — inspect diffs before accepting)

**Step 6: Commit**

```bash
rtk git add -A src/
rtk git commit -m "feat: make split-mode configurable per profile"
```

---

## Task 6: Add `checkpoint_min_step` as a launch-runtime field

Hybrid SSM state cannot be rolled back, so checkpoint spacing governs mid-context
cache reuse. `--ctx-checkpoints` is already plumbed; `--checkpoint-min-step`
(default 8192) is not.

**Files:** same set as Task 5.

**Step 1: Write the failing test**

```python
    def test_checkpoint_min_step_is_emitted_when_set(self) -> None:
        cmd = build_server_cmd(make_server_config(checkpoint_min_step=4096))
        assert cmd[cmd.index("--checkpoint-min-step") + 1] == "4096"

    def test_checkpoint_min_step_omitted_when_unset(self) -> None:
        assert "--checkpoint-min-step" not in build_server_cmd(make_server_config())
```

**Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/server/test_server.py -v -k checkpoint_min_step`
Expected: FAIL — unexpected keyword argument

**Step 3: Implement**

Add `checkpoint_min_step: int | None = field(default=None, kw_only=True)` to
`LaunchRuntimeFields` and `checkpoint_min_step: int | None` to
`LaunchRuntimeOverrides`. This one is optional-int, so copy the `ctx_checkpoints`
pattern verbatim, including the non-negative validation in `config/server.py` and
`_append_optional_server_flags`:

```python
    if cfg.checkpoint_min_step is not None:
        cmd.extend(["--checkpoint-min-step", str(cfg.checkpoint_min_step)])
```

**Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
rtk git add -A src/
rtk git commit -m "feat: add checkpoint-min-step profile field"
```

---

## Task 7: Sweep harness

One script. It builds commands through the **real** code path so a sweep also
smoke-tests Tasks 1-6.

**Files:**
- Create: `scripts/sweep_qwen38.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Sweep llama-server settings for one profile and record PP/TG.

Builds each variant's command through the real profile/builder path, launches it,
replays one fixed long prompt twice (cold + warm), and records the server's own
timings from the second /completion response.

Usage: uv run python scripts/sweep_qwen38.py [--profile qwen35] [--out sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from llama_manager.config.builder import create_default_profile_registry, resolve_profile_config
from llama_manager.validation.commands.builder import build_server_cmd

# Each variant is a dict of ServerConfig field overrides. Keep the list short and
# ordered: the winner of each stage is folded into BASE before the next stage.
VARIANTS: list[dict[str, object]] = [
    {},  # control — current profile settings
    {"ubatch_size": 512},
    {"ubatch_size": 1024},
    {"ubatch_size": 2048},
    {"ubatch_size": 1024, "batch_size": 4096},
]

PORT = 18081
PROMPT_TOKENS_TARGET = 160_000
N_PREDICT = 256


def build_prompt(repo_root: Path) -> str:
    """Deterministic long prompt: repo source concatenated to ~160k tokens."""
    chunks: list[str] = []
    total = 0
    for path in sorted(repo_root.glob("src/**/*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"# file: {path}\n{text}\n")
        total += len(text) // 4  # ponytail: chars/4 token estimate, good enough for a fixed prompt
        if total >= PROMPT_TOKENS_TARGET:
            break
    return "".join(chunks) + "\n\nSummarise the module layout above in one sentence.\n"


def wait_for_health(port: int, timeout_s: int = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s")


def complete(port: int, prompt: str) -> dict[str, float]:
    payload = json.dumps(
        {"prompt": prompt, "n_predict": N_PREDICT, "cache_prompt": True, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["timings"]


def run_variant(base_cmd: list[str], overrides: dict[str, object], prompt: str) -> dict[str, object]:
    proc = subprocess.Popen(base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_health(PORT)
        complete(PORT, prompt)  # cold: populates the prompt cache
        timings = complete(PORT, prompt)  # warm: the number we record
    finally:
        proc.terminate()
        proc.wait(timeout=120)
    return {
        "variant": json.dumps(overrides, sort_keys=True),
        "pp_tok_s": round(timings["prompt_per_second"], 2),
        "tg_tok_s": round(timings["predicted_per_second"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen35")
    parser.add_argument("--out", type=Path, default=Path("sweep.csv"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    prompt = build_prompt(repo_root)
    registry = create_default_profile_registry()

    rows: list[dict[str, object]] = []
    for overrides in VARIANTS:
        cfg = resolve_profile_config(registry, args.profile, port=PORT)
        cfg = dataclasses.replace(cfg, **overrides) if overrides else cfg
        row = run_variant(build_server_cmd(cfg), overrides, prompt)
        print(row, flush=True)
        rows.append(row)

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "pp_tok_s", "tg_tok_s"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the command builder path works without launching anything**

Run:

```bash
uv run python -c "
from llama_manager.config.builder import create_default_profile_registry, resolve_profile_config
from llama_manager.validation.commands.builder import build_server_cmd
cfg = resolve_profile_config(create_default_profile_registry(), 'qwen35', port=18081)
print(' '.join(build_server_cmd(cfg)))
"
```

Expected: a full `llama-server` line containing `--split-mode layer`,
`--spec-type draft-mtp`, `--ubatch-size 256`. If `resolve_profile_config`'s
signature differs, read `src/llama_manager/dry_run.py:119-140` and match how it
calls the function there — do not guess.

**Step 3: Check `dataclasses.replace` works on `ServerConfig`**

Run:

```bash
uv run python -c "
import dataclasses
from llama_manager.config.builder import create_default_profile_registry, resolve_profile_config
cfg = resolve_profile_config(create_default_profile_registry(), 'qwen35', port=18081)
print(dataclasses.replace(cfg, ubatch_size=1024).ubatch_size)
"
```

Expected: `1024`. `ServerConfig` mixes in `SpeculativeDecodingFieldsMixin` and
`LaunchRuntimeAttributeMixin`, so if `replace` rejects the mixin-backed fields,
fall back to mutating a copy via `setattr` for those specific keys and note it in
the script.

**Step 4: Commit**

```bash
rtk git add scripts/sweep_qwen38.py
rtk git commit -m "feat: add qwen38 settings sweep harness"
```

---

## Task 8: Stage 1 — ubatch / batch sweep

**Prerequisite:** no other llama-server instance is running (`ps aux | grep llama-server`),
and both 3090s are idle (`nvidia-smi`). Each variant loads 31.4 GB; expect ~2 min
per variant just for load.

**Step 1: Run the sweep**

```bash
uv run python scripts/sweep_qwen38.py --out /tmp/sweep-stage1.csv
```

Expected: five rows. Control should land near PP 452 / TG 22.5 — if it does not,
stop and investigate before trusting any other row.

**Step 2: Record results in the design doc**

Append a results table to
`docs/plans/2026-08-15-qwen38-pp-tg-tuning-design.md` under a new
`## Results` heading. Record the raw CSV values, not a summary.

**Step 3: Fold the winner into the script's `VARIANTS` base and commit**

```bash
rtk git add -A docs/ scripts/
rtk git commit -m "bench: record stage 1 ubatch/batch sweep results"
```

---

## Task 9: Stage 2 — split-mode sweep

**Step 1: Edit `VARIANTS` in `scripts/sweep_qwen38.py`**

```python
VARIANTS = [
    {"ubatch_size": <stage-1 winner>},
    {"ubatch_size": <stage-1 winner>, "split_mode": "row"},
]
```

`split_mode: "none"` is not testable at Q8 — 31.4 GB does not fit one 24 GB card.
Skip it and say so in the results.

**Step 2: Run and record** — same commands as Task 8, `--out /tmp/sweep-stage2.csv`.

**Step 3: Commit results.**

---

## Task 10: Stage 3 — MTP draft sweep

**Step 1: Edit `VARIANTS`** to hold stage 1+2 winners constant and vary the draft:

```python
VARIANTS = [
    {**BASE, "spec_draft_p_min": 0.75, "spec_draft_n_max": 7},   # control
    {**BASE, "spec_draft_p_min": 0.60, "spec_draft_n_max": 7},
    {**BASE, "spec_draft_p_min": 0.50, "spec_draft_n_max": 10},
    {**BASE, "spec_draft_p_min": 0.60, "spec_draft_n_max": 10},
]
```

`spec_draft_p_min` and `spec_draft_n_max` live on the nested
`SpeculativeDecodingConfig`, so `dataclasses.replace(cfg, ...)` may not reach them.
If it fails, build the variant by replacing `cfg.spec_decode` with a new
`SpeculativeDecodingConfig` instead.

**Step 2: Also capture acceptance rate.** The `/completion` timings block does not
include draft acceptance — read it from the server's stderr instead by changing
`stderr=subprocess.DEVNULL` to a temp file per variant and grepping
`draft acceptance` from it. Record acceptance alongside TG; a TG win with collapsed
acceptance is a measurement artefact, not a win.

**Step 3: Run, record, commit.**

---

## Task 11: Stage 4 — spec-type combination

**Step 1: Add one variant:** `{**BASE, "spec_type": "draft-mtp,ngram-mod"}`
with the stage 3 winning draft settings, plus `spec_ngram_size_n=24`,
`draft_min=48`, `draft_max=64` (this build's defaults).

**Step 2: Run, record, commit.** This is the first live exercise of Tasks 1 and 3 —
if the server refuses to start, capture its stderr and fix the emission before
continuing.

---

## Task 12: Stage 5 — context checkpoints

This stage needs a **divergence** replay, not the identical-prompt replay the other
stages use: the point is mid-context cache recovery.

**Step 1: Extend the harness with a divergence probe**

After the warm request, send a third request whose prompt is the same long prompt
with ~2000 tokens **replaced in the middle** (not appended), and record that
request's `prompt_per_second` separately as `pp_diverged_tok_s`.

**Step 2: Sweep** `ctx_checkpoints` ∈ {32, 64} × `checkpoint_min_step` ∈ {8192, 4096},
holding all previous winners constant.

**Step 3: Run, record, commit.** Watch VRAM headroom — checkpoints cost memory, and
the baseline already sits at ~40 GB of 48 GB.

---

## Task 13: Land the winners

**Files:**
- Modify: `src/llama_manager/config/builder.py:500-520` (the `qwen35` `SlotProfileSpec`)
- Modify: `src/llama_manager/config/defaults.py:128-135` (`ubatch_size_qwen35`, etc.)
- Modify: `docs/plans/2026-08-15-qwen38-pp-tg-tuning-design.md` (final before/after)
- Modify: `README.md` if the profile table's stated model is now wrong

**Step 1: Update the built-in profile defaults** to the winning values.

**Step 2: Run the full suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q`
Expected: all PASS. Dry-run snapshot tests will need updating — verify each diff
matches an intended change.

**Step 3: Update the user's live profile**

`~/.config/llm-runner/slot_profiles.toml` is user state and is **not** in the repo.
Tell the user which values changed and let them apply it through the TUI profile
modal, or confirm before editing that file directly.

**Step 4: Commit**

```bash
rtk git add -A src/ docs/ README.md
rtk git commit -m "perf: tune qwen35 profile defaults for Qwen3.8-27B"
```

---

## Notes for the executor

- **Do not add a sweep framework.** One script, one CSV, edited between stages.
- **The control row is the safety check.** Every stage re-runs it; if control drifts
  more than ~10% from the previous stage, the machine state changed and the stage's
  numbers are not comparable.
- **One variable per stage.** Folding two changes in at once makes the result
  unattributable.
- `AGENTS.md` forbids compatibility shims — rename `dflash` outright, no aliasing.
