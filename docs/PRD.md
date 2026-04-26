# Product Requirements Document (PRD)

## Document Metadata

- **Product**: `llm-runner`
- **Version**: MVP draft `v0.3`
- **Owner**: Solo operator (researcher / DevOps profile)
- **Status**: Draft for implementation planning
- **Supported machine**: **Anchored workstation only** (see Appendix A). MVP documentation and defaults target this machine; broader Linux support is **not** a product claim for MVP.

---

## 1) Product Goal

`llm-runner` is a **local-first control plane** for launching and monitoring **two LLM serving processes** on **two distinct GPU slots** of a single mixed-GPU workstation, using **`llama.cpp` as the only MVP inference backend**.

MVP delivers:

- **GPU-slot-first orchestration** (ports and bind addresses belong to **slots**, not to individual model records).
- **TUI-driven** clone/build flows that produce **`llama.cpp` artifacts for Intel SYCL and NVIDIA CUDA** (serialized builds).
- **`doctor` / `setup` / `smoke`** for diagnostics, environment mutation, and **OpenAI-compatible** verification.
- **Manual** (TUI-triggered) profiling and **persisted** preset guidance (`balanced`, `fast`, `quality`) with deterministic user overrides.
- An internal **backend abstraction** so **`vLLM` can be added later without CLI churn**, but **`vLLM` is not implemented in MVP**.

---

## 2) Target User and Context

### Primary Persona

Solo local power user with a mixed-GPU workstation (Intel + NVIDIA), comfortable with CLI tooling, shell environments, and server tuning.

### Operating Context

- Single workstation, **no** cluster orchestration.
- Two complementary models (MVP: **summary-balanced + Qwen 3.5–35B** class), extensible via config + TUI.
- Frequent iteration on **`llama.cpp` flags**, **slot assignment**, and **build flavors** (SYCL vs CUDA).

---

## 3) Problem Statement

Running two local LLM servers across heterogeneous GPUs typically requires manual scripts, backend-specific flags, and one-off tuning. That creates friction, configuration drift, and slow iteration.

`llm-runner` must provide a reliable single pane of glass to:

- launch **one model per GPU slot** with explicit slot mapping,
- observe process / GPU health and logs,
- resolve and inspect **exact** launch commands (**dry-run**),
- run **guided builds** and **setup** when artifacts or toolchains are missing,
- tune configs with **presets + overrides** in a repeatable way,
- verify serving with **`smoke`** (**OpenAI-compatible** API).

---

## 4) Jobs to Be Done (JTBD)

- As a solo operator, I want **one command / TUI flow** to launch two models on **different GPU slots** so I stop juggling scripts.
- As a **`llama.cpp` experimenter**, I want **SYCL and CUDA artifacts** and **flag bundles** I can trust, with **master-tip builds** driven from the TUI when I choose to build.
- As a systems user, I want **`doctor` and `setup`** to tell me **exactly** what failed (toolchain, ports, bind, GPU visibility) and **what to run next**.
- As a tuner, I want **presets plus explicit overrides** with a **deterministic merge order**.
- *(Post-MVP)* As a backend experimenter, I want to add **`vLLM`** behind the same abstraction without rewriting the control plane.

---

## 5) MVP Scope

### In Scope

1. **Two-model orchestration** on **two GPU slots** of the anchored workstation (**one active model per slot**, hard invariant).
2. **MVP backend**: **`llama.cpp` only** at runtime; **backend plugin interface + disabled / experimental `vLLM` path** in config/code as agreed.
3. **TUI-driven** `llama.cpp` **source build** workflow (**`master` branch**), **two targets**: **Intel SYCL** and **NVIDIA CUDA**, **always serialized** (no parallel builds in MVP).
4. **`doctor`**, **`setup`**, **`smoke`** (see §7–§10).
5. **Backend-aware (llama.cpp) presets** and **manual** profiling flows initiated from the TUI; **profile results persisted** locally with **staleness warnings** (`--strict-profiles` **post-MVP**).
6. **GGUF-only** models in MVP; **metadata extracted** via **pinned** PyPI **`gguf`** reader (**KV / header**, no full weight load), with **read caps** and **parse wall-clock timeout**.
7. **TUI + CLI** monitoring: logs, process state, GPU telemetry, selected backend (**always `llama_cpp` in MVP**), build job state, profile warnings.
8. **Config**: `schema_version: 1`, repo-shipped defaults + **user override file**; **TUI “Apply” vs “Save”** for model edits; **ports edited only at slot level** (model editor shows inherited port **read-only**).
9. **OpenAI-compatible smoke**: sequential **`smoke both`** is the **MVP completion gate** (see §10, Appendix B).

### Out of Scope (MVP)

- **`vLLM` runtime**, multi-host scheduling, cloud orchestration.
- Multi-user auth / RBAC, full API gateway / routing layer.
- **Claiming support for arbitrary Linux hardware** beyond the anchored workstation narrative.
- **Automatic** profiling on first model load (**manual / TUI** in MVP).
- **Formal product metrics dashboard**; success is **functional**: models start, respond, **`smoke both` passes**.
- **CI-enforced** README snippet freshness (generation is **ad-hoc before releases**; optional pre-commit only).
- Single **`--pack-report`** / zipped diagnostics bundle (**post-MVP**; MVP uses plain **report directories** only).

---

## 6) Current-State Baseline

The codebase already provides:

- **`llama_manager/`**: config models, command construction, process lifecycle, GPU stats, log buffering.
- **`llama_cli/`**: argument parsing, CLI runner, dry-run, TUI.
- Run modes: `summary-balanced`, `summary-fast`, `qwen35`, `both`, `dry-run`.
- Quality baseline: `ruff`, `pyright`, `pytest`.

**Gaps to close for this PRD**: slot-first config, **`doctor` / `setup` / `smoke`**, TUI build jobs + provenance, GGUF metadata import (+ **test fixtures**), profile cache, slot locks, hardware warning / allowlist flows, **VRAM heuristics**, **OpenAI flag bundles** (incl. **chat template** for Qwen-class), **smoke defaults** (`both` vs **`slot`**), **`setup` venv**, exit-code contracts, README excerpt generation from PRD markers (`<!-- readme:... -->`), gendoc.py tool.

**Milestone Mapping**:
- **M0** (Documentation): gendoc.py, README marker extraction
- **M1** (Slot-first): slot-first config, validation, dry-run, lockfiles, risk acknowledgement
- **M2** (Build/Setup): build pipeline, provenance, setup venv
- **M3** (Profiling): profile cache, staleness warnings
- **M4** (Operational): smoke, TUI monitoring, shutdown, GGUF parsing, hardware acknowledgment, logging, exit codes

---

## 7) Functional Requirements

### FR-001 — Slot-first launch

User launches **up to two models**, each bound to a **`gpu_slot`**; **each slot** owns **bind host + port**. **One model per slot** in MVP.

### FR-002 — Backend selection (MVP semantics)

User selects backend via config / CLI / TUI **without source edits**. MVP **runtime** supports **`llama_cpp`**. **`vllm`** may appear as a **parsed value** but **must not launch** unless an **explicit experimental gate** (e.g. env flag) documents future behavior; **`doctor` blocks** `vllm` models as **not enabled** for MVP.

### FR-003 — Dry-run

Dry-run prints **exact** resolved command line, **binary path**, **model path**, **slot**, **merged environment** (defaults `<` workstation `<` slot overlays `<` process), **OpenAI-compat flag bundle**, **effective ports**, **hardware / compatibility notes**, and a small **compatibility matrix** (e.g. `vllm: not enabled in MVP`). Supports **`dry-run smoke`** showing **sequential smoke order**.

The **anchored workstation OpenAI flag bundle** MUST include whatever **`llama-server` arguments** are required so **Qwen-class** GGUFs behave correctly under the **OpenAI-compatible** surface (e.g. **chat template / `--jinja`** or the **successor flag** required by the pinned upstream CLI). If upstream renames flags, the **workstation profile** is updated first; **`smoke`** remains the gate that catches drift.

**OpenAI Flag Bundle Schema**: A dictionary mapping CLI-style flag names (with leading `--`) to their effective values. For M1, allowed keys are: `--port`, `--host`, `--chat-format`, `--openai`. Bundle is opt-in based on configuration; if empty, `openai_flag_bundle: {}` in dry-run output.

### FR-004 — TUI build pipeline (`llama.cpp`, `master`)

From the TUI, user can start **serialized** builds for:

- **`intel-sycl`** artifact, and
- **`nvidia-cuda`** artifact.

Each target has **preflight**, **progress**, **success/failure state**, and **retry failed only**. **`doctor --repair`** clears **failed-target staging** without deleting **successful** artifacts.

### FR-005 — Toolchain and diagnostics

Missing build tools → **safe failure** with **actionable** OS-level hints. **`setup`** performs mutating steps **only** under **confirmatory UX**; **`sudo` steps** are **opt-in per action**.

**Python isolation**: **`setup` creates or reuses a dedicated virtual environment** at **`$XDG_CACHE_HOME/llm-runner/venv`** (fallback **`~/.cache/llm-runner/venv`**). **Future** `setup` steps that install Python packages (including **post-MVP vLLM** prep) target **that venv only**; **`doctor` verifies** interpreter path and basic import health as applicable. **Runtime `llm-runner`** does not mutate this venv during **serve**.

### FR-006 — Artifacts and provenance

Built artifacts live at **predictable paths** under the selected `llama.cpp` source root. By default, `llm-runner` clones/reuses source at **`$XDG_CACHE_HOME/llm-runner/llama.cpp`** and writes SYCL/CUDA binaries under that root at **`build/bin/llama-server`** and **`build_cuda/bin/llama-server`**. `LLAMA_CPP_ROOT` or `--source-dir` may point at an explicit checkout. Each build records **remote URL**, **checked-out commit SHA (or `master` tip SHA if no `--git-commit` pin was provided)**, and **timestamps** in XDG state. **No “silent auto-build on launch”** in MVP—user initiates builds via TUI (or future explicit CLI parity).

### FR-007 — Manual profiling

User triggers profiling from the TUI; tool invokes native bench tools **as subprocesses** where integrated. Results feed **`balanced` / `fast` / `quality`** guidance.

### FR-008 — Profile persistence

Profiles persist under **`~/.cache/llm-runner/profiles/`** (or XDG-aware equivalent), keyed by **GPU identifiers + backend + flavor**, including **driver / relevant version metadata** where available. **Stale profiles warn**; **`--strict-profiles` deferred**.

### FR-009 — Overrides

Explicit CLI / config overrides **always win** over preset defaults and profile guidance in a **documented, deterministic order**.

### FR-010 — Monitoring (TUI)

TUI shows **per-slot** status, **logs**, **GPU telemetry**, **backend** (`llama_cpp`), **build job state**, **profile warnings**, **UNSAVED** badge when model **Apply** is active.

### FR-011 — CLI scripting

Stable **`--json`** where applicable for health / status. Commands **`doctor`**, **`setup`**, **`smoke`** with **documented exit codes** (see Appendix B; **doctor** and **smoke** use **non-overlapping** code ranges).

### FR-012 — Shutdown

Graceful shutdown: **SIGTERM**, wait, escalate. **Abnormal TUI exit** must **terminate child servers** (same policy as graceful path where possible) to avoid **orphan** GPU processes.

### FR-013 — Config merge and schema

Repo defaults + user file merge by **`model_id` / slot keys**; **user wins** on conflicts at the **same key**. **`schema_version: 1`**. **`doctor migrate-config`**: backup (`*.bak`) before rewrite; fail validation on bad migrate.

### FR-014 — GGUF metadata

**GGUF-only** MVP; parse **metadata** with pinned **`gguf`** PyPI dependency (**`gguf-parser`** remains a **documented alternative** if `gguf` proves unsuitable—pick **one** for CI). Enforce:

- **Prefix read cap** for metadata scanning: default **32 MiB** from file start (tunable in `Config()`); do **not** map full weights.
- **Wall-clock parse timeout**: default **5 seconds** per file (tunable); on timeout treat as **unsupported / corrupt** with a clear error.

Support **unsharded** files or **user-indicated header shard**; **`doctor`** distinguishes **corrupt file** vs **parser / spec mismatch** (e.g. GGUF version).

**Filename normalization** (for display + smoke id inference): deterministic pipeline — take **basename**, apply **Unicode normalization** (NFKC), replace **whitespace** with **`-`**, lowercase optional (config: default **on**); optional **strip quant suffix patterns** (workstation-tuned list, **off by default** unless enabled in profile). **`doctor` dry-run / metadata view** shows **raw path**, **normalized stem**, and **resolved smoke name**.

**Test fixtures policy**: repo contains **small synthetic GGUF** files under a **`tests/fixtures/`** (or equivalent) path, produced **once** by a **documented maintainer script** (e.g. `scripts/generate_gguf_fixtures.py`). **CI consumes committed bytes only**—**no fixture generation in CI**.

**Metadata Fields Extracted**: `general.name`, `general.architecture`, `tokenizer.type`, `llama.embedding_length`, `llama.block_count`, `llama.context_length`, `llama.attention.head_count`, `llama.attention.head_count_kv`. Missing `general.name` falls back to normalized filename stem for smoke model_id.

### FR-015 — Smoke (OpenAI-compatible)

**Commands**:

- **`smoke both`** — **MVP completion gate**; runs **slot A then slot B** in **declaration order** from config (same order as **`dry-run smoke`**). Fails **non-zero** if **either slot** cannot be probed (e.g. only one GPU present → operator uses **`smoke slot <slot_id>`** instead).
- **`smoke slot <slot_id>`** — **Partial** / degraded workstation probing; same probes as **`both`** for **one** slot.

**Pacing**: after a slot finishes successfully, wait **2 seconds** (default **`smoke.inter_slot_delay_s`**, tunable) before starting the **next** slot’s probes—reduces GPU / driver contention.

**Phases** (per slot): **listen / accept** → **`/v1/models` when present** → **minimal chat completion**. Defaults in **Appendix D**; each phase records **which phase failed** in logs / reports.

**Chat probe**: **`temperature 0`**, **`max_tokens` default 16** (allowed workstation range **8–32**). Keep prompts **minimal** (single user message).

**Model id for API**: precedence **`general.name` (GGUF)** → **normalized filename stem** → **catalog `smoke.model_id` / override** → optional **`/v1/models` match** when discovery succeeds. **`smoke` header** prints **resolved id** + **provenance** (binary tip SHA, `llm-runner` version).

**Auth**: **Default no API key** (localhost). If **server auth is enabled** in config, **`smoke` MUST accept** **`--api-key`**, config **`smoke.api_key`**, or env (**exact name in CLI docs**); **never** emit the key into logs (**redaction** per FR-018).

**Repeated failure rule**: if **two consecutive** smokes for the **same slot** fail with **model-not-found / 4xx** attributable to **wrong `model` id**, **`doctor`** / next smoke **requires explicit `smoke.model_id`** (or slot-level override) **before** further attempts—no silent infinite retry.

### FR-016 — Local safety and collisions

Default bind **`127.0.0.1`**, default **no API key**. **Port-in-use** → **refuse** unless explicit **`--force-bind`** (logged as dangerous). **Per-slot lockfiles** under runtime dir; **`doctor --locks`** surfaces holders.

**VRAM heuristics (warn + confirm)**: before launch (and optionally before **`smoke`** when full model is loaded), compare **best-effort free VRAM** (**`nvidia-smi` / Arc equivalent**) against **heuristic footprint** (GGUF size, metadata hints, architecture defaults). If **free memory minus safety margin** looks insufficient, **warn** and require **explicit user confirmation** in TUI or **`--confirm-vram-risk`** (CLI naming in implementation docs). **Not** a hard block in MVP except where **impossible headroom** is unambiguous (implementation-defined edge case with clear message).

### FR-017 — Hardware acknowledgment

Non-anchor or unexpected topology → **warn** (not MVP hard-fail).

- **TUI**: explicit **Continue** after reading warning details (detected vs expected summary).
- **CLI / automation**: **`--ack-nonstandard-hardware`** records consent for **this** launch path (still logs warning).

**Persistent allowlist**: JSON (or equivalent) under **`~/.config/llm-runner/hardware-allowlist.json`**, entries keyed by **`machine_fingerprint` + PCI device IDs**; **invalidated automatically** when fingerprint / PCI set changes (user must re-ack).

**Session snooze**: **ephemeral** file under **`$XDG_RUNTIME_DIR/llm-runner/`** (fallback **`/tmp/llm-runner/`**) — **suppresses duplicate warnings until exit or reboot**, never replaces **allowlist** for permanent suppression.

**Hardware Fingerprint Computation**: Use `lspci` output hash for GPU devices combined with SYCL device enumeration (`sycl-ls`) to create a deterministic machine identifier. Fingerprint changes when GPU hardware changes.

### FR-018 — Logging and reports

**`setup`** / mutating actions append to a **rotating log** with **command, timestamp, exit code, truncated output**; **redact** obvious secret patterns; failure drops a **report directory** under **`~/.local/share/llm-runner/reports/<timestamp>/`** (plain files MVP).

### FR-019 — Documentation excerpt generation

`PRD.md` contains `<!-- readme:... -->` marker blocks; **`gendoc.py`** (ad-hoc, before releases) injects excerpts into **README** (exact mechanism documented in dev docs / M0).

**Marker Format**: `<!-- readme:section-name -->` opens a section and `<!-- /readme:section-name -->` closes it. Content between markers is extracted verbatim. Example:
```markdown
<!-- readme:workstation -->
(Workstation table + prerequisites — keep short)
<!-- /readme:workstation -->
```

**gendoc.py Behavior**: Scans PRD for marker pairs, extracts content, injects into README at `<!-- BEGIN readme:section-name -->` / `<!-- END readme:section-name -->` markers. Ad-hoc tool, not part of MVP runtime.

---

## 8) Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| **NFR-001** | Validation prevents **duplicate slot assignment**, **invalid ports**, **missing models**, **`vllm` without experimental gate**, and **low-confidence GGUF parse** (where defined). **VRAM-risk** paths require **explicit confirmation** (**warn-only** heuristic). |
| **NFR-002** | Failures attribute to **slot → model → process → phase**; include remediation hints. |
| **NFR-003** | **Dry-run / config resolve** feels immediate; long operations show **phase progress**. |
| **NFR-004** | Determinism: **identical config + identical built artifacts + identical environment** ⇒ identical resolved commands. **`master` moves over time**: record **tip SHA** for supportability, not bit-for-bit eternal reproducibility. |
| **NFR-005** | **Runtime** servers: **no hidden network** beyond binds; **no silent package installs** during **serve**. **`setup`** may use network **only** in explicit flows. |
| **NFR-006** | Polling / log buffer defaults keep TUI responsive; caps documented; **`doctor` prints effective values**. |
| **NFR-007** | Diagnostics: optional **`--redact-paths`** for share bundles; default logs favor local debuggability. |

---

## 9) UX Flows (MVP)

### First run / build

1. User opens TUI or CLI and requests **build** for a missing **SYCL / CUDA** target.
2. Wizard runs **serialized** builds with **preflight**; on network loss, offer **offline continue** if local clone exists.
3. **Progress**, success/failure per target; **`doctor --repair`** clears failed staging.

### Profiling (manual)

1. User triggers **profile** from TUI.
2. Bench subprocess runs; results persist; **staleness warns** later.
3. Presets update **guidance**; overrides still win per FR-009.

### Happy path

1. **`doctor`** clean (warnings allowed or acknowledged).
2. Optional **`setup`** completes mutating steps.
3. **`dry-run`** / **`dry-run smoke`** reviewed.
4. Launch **both** models; TUI monitors.
5. **`smoke both`** exits **0** (or **`smoke slot …`** when only one slot is operable—does not satisfy full MVP gate alone).
6. **Graceful shutdown**.

### Failure paths

- **Preflight**: port conflict, lock held, **`vllm`**, missing GGUF path, parse timeout → **blocking** with hints; **hardware mismatch** → **warn** + **`--ack-nonstandard-hardware`** / **allowlist** / **session snooze**; **VRAM risk** → **warn** + explicit confirm (`--confirm-vram-risk` or TUI equivalent).
- **Build**: surfaced **per-target**; **retry failed** only.
- **Runtime**: one model **OOM / crash** → mark **offline**, sibling continues; suggest **profile / limits** tuning.
- **Abnormal TUI exit** → **stop servers** (FR-012).

---

## 10) Acceptance Criteria

| ID | Criterion |
| --- | --- |
| **AC-001** | Two models launch on **distinct slots** with **slot-owned ports**; duplicate slot assignment **rejected**. |
| **AC-002** | **`dry-run`** shows **binary**, **args**, **model path**, **device mapping per slot**, **merged env**, **OpenAI flag bundle**, **matrix** row for **`vllm` not enabled**. |
| **AC-003** | **`vllm`** in config **fails `doctor`** (or experimental gate only) — **no accidental launch**. |
| **AC-004** | **Hardware warning path** works: **non-anchor** topology **warns**; **`--ack-nonstandard-hardware`**, **`~/.config/llm-runner/hardware-allowlist.json`**, and **session snooze** under **`$XDG_RUNTIME_DIR/llm-runner/`** (or **`/tmp/llm-runner/`**); **single-GPU** warns and **does not fake** a second running model. |
| **AC-005** | **Port listener conflict** → **actionable error** unless **`--force-bind`** (explicit). |
| **AC-006** | **TUI build** produces **SYCL + CUDA** artifacts **when both succeed**, with **recorded SHAs**; **serialized** execution enforced. |
| **AC-007** | **Toolchain missing** → **clear** install guidance; **`setup` log** captures failures with **redaction**. |
| **AC-008** | **Manual profile** persists cache; **staleness** surfaces in TUI + **`doctor`**. |
| **AC-009** | **Override precedence** covered by tests (preset < profile guidance < explicit override). |
| **AC-010** | **GGUF metadata** extracted for MVP fields; **timeout / cap** honored; **fixture tests** in CI. |
| **AC-011** | **TUI** shows **per-slot** health, **logs**, **GPU stats**, **`llama_cpp`**, **build state**. |
| **AC-012** | **Graceful shutdown** and **abnormal exit** both end **without orphan servers** (best-effort **SIGTERM** escalation). |
| **AC-013** | Isolated crash / OOM highlights **failed slot** without killing sibling. |
| **AC-014** | **`smoke both`** is **authoritative MVP gate**: sequential probes pass; **distinct exit code family** from **`doctor`** (Appendix B). |
| **AC-015** | **`gendoc.py`** extracts **marker sections** from this PRD into README **when run** (release-time expectation). |
| **AC-016** | **VRAM heuristic** warns when free memory vs estimated load is risky; **launch / smoke** does not proceed without **explicit confirmation** path. |
| **AC-017** | **`smoke both`** and **`smoke slot`** behave per **FR-015** and **Appendix D** (ordering, delays, timeouts, **`max_tokens`**). |
| **AC-018** | **Committed synthetic GGUF fixtures** + **maintainer generator script** documented; **CI** uses **static** fixtures only. |
| **AC-019** | **`setup` venv** path created/verified; **no serve-time** pip drift. |
| **AC-020** | **`dry-run`** for anchored models includes **Qwen-class template / jinja (or successor) flags** in the printed **OpenAI flag bundle**. |

---

## 11) Near-Term MVP Roadmap

### M0 — Documented baseline

Freeze terminology; add **Appendices A–D** to repo docs; wire **README excerpt markers**; align **`AGENTS.md`** / dev docs with **`doctor` / `setup` / `smoke`** commands.

### M1 — Slot-first config + backend abstraction + validation

Implement **`gpu_slot` model**, merge rules, **`doctor` foundation**, **`dry-run` matrix**, **hardware warn / allowlist**, **lockfiles**, **`vllm` guard**. Deliver **FR-001–FR-003**, **FR-013**, **FR-016–FR-017**, parts of **FR-011**.

### M2 — Build wizard + `setup`

**TUI** (and minimal CLI parity if needed) for **FR-004–FR-006**, **FR-018**; **`master`** checkouts; **SYCL/CUDA** serialization; **provenance** files.

### M3 — Profiling + presets

**FR-007–FR-009**, bench integration, **cache** format, staleness warnings.

### M4 — Operational hardening + `smoke`

**FR-010**, **`smoke both` / `smoke slot`**, OpenAI-compat bundles (incl. **Qwen template flags**), **Appendix D** timeouts, report dirs, perf caps, **AC-014–AC-020** as applicable.

---

## 12) Milestone-to-Requirement Traceability

| Milestone | Primary requirements |
| --- | --- |
| **M0** | NFR docs parity, **FR-019**, metadata sections |
| **M1** | **FR-001–FR-003**, **FR-013**, **FR-016–FR-017**, **FR-011** (`doctor`) |
| **M2** | **FR-004–FR-006**, **FR-005**, **FR-018** |
| **M3** | **FR-007–FR-009** |
| **M4** | **FR-010–FR-012**, **FR-011** (`smoke`), **FR-014–FR-017**, **Appendix D** defaults, NFR-002/NFR-003/NFR-006–007 |

---

## 13) Risks

- **GPU index mapping** errors across OS / SYCL / CUDA numbering — mitigate via **slot IDs** + explicit validation + **dry-run** transparency.
- **`llama.cpp` on `master`** moves rapidly → **flag / OpenAI server behavior** drift; mitigate with **pinned flag bundles per workstation revision** and **`smoke` gate**.
- **GGUF metadata inconsistency** (missing `general.name`, odd shards) → mitigate with **filename fallback + warnings + explicit `smoke.model_id` escape hatch**.
- **`gguf` parser** / file edge cases → **caps + timeouts + golden fixtures**.
- **Synthetic fixture drift** if generator script is not re-run when spec changes → **document** regen process; keep fixtures **tiny** and **rarely changed**.
- **Port ownership detection** incomplete on some systems → clear **fallback messaging** and **`--force-bind`** hazard labeling.

---

## 14) Definition of Done (PRD for MVP planning)

This PRD is implementation-ready when:

1. **FR/NFR/AC** above (**through AC-020**) are **numbered, testable**, and **trace** to **M0–M4**.
2. **Scope** clearly states **llama.cpp-only runtime**, **`vLLM` deferred**, **anchored workstation**, and **slot-first networking**.
3. **Operational contracts** exist for **`doctor` / `setup` / `smoke`**, **build provenance**, **locks**, and **shutdown**.
4. Work can be **broken into engineering tasks** without further product ambiguity on **MVP vs future** boundaries.

**Canonical Terminology**:
- **warning**: Non-blocking diagnostic; does not prevent launch.
- **launch-blocking error**: Error that prevents any slot from starting.
- **risky operation**: Launch condition requiring explicit operator acknowledgement (port <1024, non-loopback bind, warning-bypass override).
- **degraded launch**: One-slot launch when only one of two configured slots is available.
- **full-block launch**: Launch outcome where zero configured slots can start.
- **current launch attempt**: Runtime context covering a single invocation; synonymous with "session-only" in M1.

---

## Appendix A — Anchored workstation profile (MVP defaults)

*(Replace placeholders only when the physical machine changes; keep PRD and generated README in sync.)*

| Slot ID | Hardware (reference) | Build flavor | Default bind | Notes |
| --- | --- | --- | --- | --- |
| **`arc_b580`** | Intel Arc **B580** (SYCL) | `intel-sycl` | **`127.0.0.1:8081`** | SYCL env overlays live at **slot** level |
| **`rtx3090`** | NVIDIA **RTX 3090** (CUDA) | `nvidia-cuda` | **`127.0.0.1:8082`** | CUDA device mapping validated per slot |

**MVP catalog models**: **summary-balanced** on **Arc slot**, **Qwen 3.5–35B** class on **NVIDIA slot** (exact GGUF paths remain local to the operator).

---

## Appendix B — Exit code conventions

**`doctor`**: **`0`** warnings-only OK, **`1`** blocking error, **`2`** needs **`setup`** / missing prerequisites.

**`smoke`**: use a **non-overlapping** band, e.g. **`10`** not ready, **`11`** HTTP/API error, **`12`** config validation failure (document final integers in CLI help at implementation time).

---

## Appendix C — README excerpt markers

Embed stable README fragments **between** these markers in **this file** so `gendoc.py` can copy them verbatim:

```markdown
<!-- readme:workstation -->
(Workstation table + prerequisites — keep short)
<!-- /readme:workstation -->
```

Additional marker pairs may be added (e.g. `readme:quickstart`) in M0.

---

## Appendix D — Default `smoke` and GGUF scan parameters

Values are **defaults**; workstation profile or `Config()` may override. Implementation documents the **final** CLI flags to print them via **`doctor`**.

| Parameter | Default | Notes |
| --- | --- | --- |
| **`smoke.inter_slot_delay_s`** | **2** | Pause between **successful** completion of slot *n* and start of slot *n+1*. |
| **Listen / TCP ready timeout** | **120 s** | Per slot; server must accept connections. |
| **First-token timeout** | **1200 s (20 m)** | Wall clock from chat request to first token (or implementation-defined “stream start”). |
| **Total chat completion timeout** | **1500 s (25 m)** | Hard cap per slot smoke including generation. |
| **Smoke `max_tokens`** | **16** | Allowed range in config **8–32**. |
| **GGUF metadata prefix cap** | **32 MiB** | Bytes read from file start for KV scan. |
| **GGUF parse wall timeout** | **5 s** | Per file metadata read path. |

---

*End of PRD*
