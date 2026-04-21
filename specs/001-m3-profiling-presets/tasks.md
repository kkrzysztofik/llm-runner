# Tasks: M3 Profiling + Presets

**Input**: Design documents from `/specs/001-m3-profiling-presets/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: Test tasks are REQUIRED. Every user story MUST include automated tests that validate
independent behavior.

## Phase 0: Setup (Shared Infrastructure)

**Purpose**: Verify project structure and toolchain are ready

- [x] T001 Verify `src/llama_manager/` and `src/llama_cli/` directory structure exists
- [x] T002 Verify CI toolchain: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest` all pass
- [x] T003 Create feature branch `m3-profiling-presets`

---

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Core data model, cache layer, and config extensions that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 [P] [FOUND] Create `src/llama_manager/profile_cache.py` — define `ProfileFlavor` (StrEnum), `ProfileMetrics` (frozen dataclass), `StalenessReason` (StrEnum), `StalenessResult` dataclass, `PROFILE_OVERRIDE_FIELDS` frozenset, `CURRENT_SCHEMA_VERSION` constant
- [x] T005 [P] [FOUND] Implement `ProfileRecord` dataclass in `src/llama_manager/profile_cache.py` with `from_dict()` and `to_dict()` methods (frozen=True, slots=True), using `profiled_at: str` (ISO 8601 UTC timestamp) per spec.md
- [x] T006 [FOUND] Implement `_sanitize_filename_component()` in `src/llama_manager/profile_cache.py` — regex-based sanitization matching `normalize_slot_id()` pattern from `config.py`
- [x] T007 [FOUND] Implement `compute_gpu_identifier(backend, gpu_name, device_index)` in `src/llama_manager/profile_cache.py` — CUDA: `nvidia-{name}-{idx}`, SYCL: `intel-{name}-{idx}`
- [x] T008 [FOUND] Implement `compute_driver_version_hash(driver_version: str) -> str` in `src/llama_manager/profile_cache.py` — SHA-256 truncated to first 16 hex chars
- [x] T009 [FOUND] Implement `ensure_profiles_dir(profiles_dir: Path)` in `src/llama_manager/profile_cache.py` — create with `DIR_MODE_OWNER_ONLY` (0o700), using `process_manager.py` constants
- [x] T010 [FOUND] Implement `get_profile_path(profiles_dir, gpu_identifier, backend, flavor)` in `src/llama_manager/profile_cache.py` — sanitized filename: `{sanitized_gpu_id}-{sanitized_backend}-{sanitized_flavor}.json`, with path traversal protection
- [x] T011 [FOUND] Implement `_atomic_write_json` pattern for profiles in `src/llama_manager/profile_cache.py` — write to temp file, sync, rename, apply `FILE_MODE_OWNER_ONLY` (0o600), verify permissions
- [x] T012 [FOUND] Implement `write_profile(profiles_dir, record)` in `src/llama_manager/profile_cache.py` — atomic write, returns written `Path`
- [x] T013 [FOUND] Implement `read_profile(profiles_dir, gpu_identifier, backend, flavor)` in `src/llama_manager/profile_cache.py` — JSON deserialize, validate required fields, return `ProfileRecord | None` on corrupt/missing/unsupported schema
- [x] T014 [FOUND] Implement `check_staleness(record, current_driver_version, current_binary_version, staleness_days)` in `src/llama_manager/profile_cache.py` — three conditions: driver hash mismatch, binary version mismatch, age > threshold
- [x] T015 [FOUND] Implement `load_profile_with_staleness()` in `src/llama_manager/profile_cache.py` — combines `read_profile` + `check_staleness`, returns `tuple[ProfileRecord | None, StalenessResult | None]`
- [x] T016 [FOUND] Implement `profile_to_override_dict(record)` in `src/llama_manager/profile_cache.py` — filters `record.parameters` through `PROFILE_OVERRIDE_FIELDS` whitelist
- [x] T017 [FOUND] Extend `Config` class in `src/llama_manager/config.py` — add `profiles_dir` property (XDG pattern), `profile_staleness_days: int = 30`, `server_binary_version: str = ""` (from `SERVER_BINARY_VERSION` env var)
- [x] T018 [FOUND] Extend `gpu_stats.py` — add `get_gpu_identifier()` function that parses `nvidia-smi`/`sycl-ls` output for GPU name and device index (injectable collector pattern, matching `GPUStats` design)
- [x] T019 [FOUND] Export new symbols from `src/llama_manager/__init__.py` — `ProfileFlavor`, `ProfileMetrics`, `ProfileRecord`, `StalenessReason`, `StalenessResult`, `PROFILE_OVERRIDE_FIELDS`, `CURRENT_SCHEMA_VERSION`, `compute_gpu_identifier`, `compute_driver_version_hash`, `ensure_profiles_dir`, `get_profile_path`, `read_profile`, `write_profile`, `check_staleness`, `load_profile_with_staleness`, `profile_to_override_dict`, `get_gpu_identifier`
- [x] T020 [P] [FOUND] Write `src/tests/test_profile_cache.py` — ProfileRecord serialization roundtrip, missing fields returns None, unsupported schema returns None, compute_gpu_identifier for CUDA and SYCL, compute_driver_version_hash, staleness (driver mismatch, binary mismatch, age exceeded, fresh), read_profile not found/corrupt returns None, write/read roundtrip (tmp_path), ensure_profiles_dir creates with 0o700 (tmp_path), sanitize_filename_component, sanitize rejects empty

**Checkpoint**: Foundation ready — data model, cache I/O, staleness detection, and config extensions are complete. All tests pass.

---

## Phase 2: Benchmark Module (Pure Library)

**Purpose**: Benchmark command construction and output parsing — pure library, no subprocess at module level

- [x] T021 [P] [FOUND] Create `src/llama_manager/benchmark.py` — define `SubprocessResult` (frozen dataclass: exit_code, stdout, stderr) and `BenchmarkResult` (frozen dataclass: tokens_per_second, avg_latency_ms, peak_vram_mb | None)
- [x] T022 [P] [FOUND] Implement `build_benchmark_cmd(bench_bin, model, port, threads, ctx_size, ubatch_size, cache_type_k, cache_type_v, n_gpu_layers="all")` in `src/llama_manager/benchmark.py` — constructs `llama-bench` command as `list[str]`, subprocess-safe
- [x] T023 [P] [FOUND] Implement `parse_benchmark_output(output: str)` in `src/llama_manager/benchmark.py` — parses benchmark stdout for metrics (tokens/s, latency, VRAM), returns `BenchmarkResult | None` on failure
- [x] T024 [FOUND] Define `BenchmarkRunner = Callable[[list[str]], SubprocessResult]` type alias in `src/llama_manager/benchmark.py`
- [x] T025 [FOUND] Implement `run_benchmark(cmd, runner)` in `src/llama_manager/benchmark.py` — accepts injectable runner callable, returns parsed `BenchmarkResult | None`
- [x] T026 [P] [FOUND] Write `src/tests/test_benchmark.py` — build_benchmark_cmd contains required flags, is list of strings, n_gpu_layers="all"; parse_benchmark_output success/empty/partial; run_benchmark calls runner, returns None on nonzero exit

**Checkpoint**: Benchmark module is complete — pure library functions that construct commands and parse output. Subprocess execution remains in `llama_cli/`.

---

## Phase 3: US1 — Manual Profiling from TUI (Priority: P1) 🎯 MVP

**Goal**: User triggers profiling from TUI for a specific model + GPU slot, benchmark runs, results are persisted and surfaced in TUI monitoring.

**Independent Test**: Can be fully tested by triggering a TUI profile request and verifying that results are persisted in cache with correct metadata keys, even with synthetic benchmark fixtures.

### Tests for US1 (REQUIRED) ⚠️

- [x] T027 [P] [US1] Write `src/tests/test_profile_cache.py` additions — test that `write_profile` + `read_profile` roundtrip preserves all `ProfileRecord` fields (metrics, parameters, driver_version, etc.)
- [x] T028 [P] [US1] Write `src/tests/test_profile_cache.py` additions — test `load_profile_with_staleness` returns (record, None) for fresh profile, (record, stale_result) for stale profile

### Implementation for US1

- [x] T029 [US1] Add `profile` subcommand parser in `src/llama_cli/cli_parser.py` — following `doctor` subcommand pattern (NOT to VALID_MODES), positional args: `<slot_id> <flavor>`
- [x] T030 [US1] Implement `_handle_profile_case(parsed)` in `src/llama_cli/cli_parser.py` — parses `profile <slot_id> <flavor>` with `--json` flag
- [x] T031 [US1] Create `src/llama_cli/profile_cli.py` — implement `_default_subprocess_runner(cmd)` using `subprocess.run(cmd, shell=False)`
- [x] T032 [US1] Implement `cmd_profile(parsed)` in `src/llama_cli/profile_cli.py` — validates slot not running (check lockfile), resolves benchmark binary path from Config, validates with `require_executable()`, constructs benchmark command via `build_benchmark_cmd`, runs benchmark, parses results, writes profile via `write_profile`
- [x] T033 [US1] Wire `profile` subcommand handler in `src/llama_cli/server_runner.py` — add handler in `main()` function for profile mode
- [x] T034 [US1] Add `profile` subcommand tests to `src/tests/test_server_runner.py` — slot running validation, benchmark binary validation, profile write on success, benchmark failure graceful handling
- [x] T035 [US1] Add TUI profile trigger in `src/llama_cli/tui_app.py` — add input polling daemon thread (non-blocking stdin, `queue.Queue`), `P` keybinding when slot focused, confirmation prompt, flavor selection sub-menu
- [x] T036 [US1] Add profile progress display in `src/llama_cli/tui_app.py` — status panel shows "Profiling: <flavor> [running...]", completion badge (✓/✗), abort with Ctrl+C
- [x] T037 [US1] Add stale warning badge in `src/llama_cli/tui_app.py` — yellow warning: "⚠ profile stale — <reason>" in per-slot health row
- [x] T038 [US1] Add profile trigger tests to `src/tests/test_tui_app.py` — profile trigger with stale/fresh/missing cache shows correct state, keypress queue dispatches P key

**Checkpoint**: US1 complete — user can trigger profiling from TUI, benchmark runs, results persist, stale warnings display.

---

## Phase 4: US2 — Profile Persistence and Cache Lookup (Priority: P2)

**Goal**: Profiles persist across sessions. When a user launches a model, the system loads cached profiles to inform preset selection. Profiles carry timestamp and version stamps for staleness detection.

**Independent Test**: Can be fully tested by writing a synthetic profile file to the cache directory and verifying that the config builder correctly loads and merges it at launch time.

### Tests for US2 (REQUIRED) ⚠️

- [x] T039 [P] [US2] Write `src/tests/test_config.py` additions — test `Config.profiles_dir` property returns correct path, `Config.profile_staleness_days` defaults to 30, `Config.server_binary_version` reads from env var and defaults to ""
- [x] T040 [P] [US2] Write `src/tests/test_us3_precedence.py` additions — test `merge_config_overrides` with profile_config applies overrides, profile_config ignores non-whitelisted fields, warnings list populated, stale profile includes warning, all precedence levels

### Implementation for US2

- [x] T041 [US2] Extend `merge_config_overrides()` in `src/llama_manager/config_builder.py` — add optional `warnings: list[str] | None = None` parameter (non-breaking, preserves backward compatibility)
- [x] T042 [US2] Filter `profile_config` through `PROFILE_OVERRIDE_FIELDS` whitelist before merging in `src/llama_manager/config_builder.py`
- [x] T043 [US2] Append staleness warnings to the provided `warnings` list when profile data is stale in `src/llama_manager/config_builder.py`
- [x] T044 [US2] Integrate profile loading into server launch flow — at launch, load cached profile for slot + backend + flavor, pass as `profile_config` layer to `merge_config_overrides`
- [x] T045 [US2] Update doctor output in `src/llama_cli/doctor_cli.py` — add `_check_profiles()` function that lists profile files, checks staleness for each, adds warnings for stale profiles
- [x] T046 [US2] Add `--repair` action to doctor in `src/llama_cli/doctor_cli.py` — offer to remove profiles stale beyond configurable max-age (default 90 days)
- [x] T047 [US2] Add doctor profile staleness tests to `src/tests/test_doctor_cli.py` — stale profiles detected, fresh profiles pass, no profiles dir handled gracefully

**Checkpoint**: US2 complete — profiles persist across sessions, merge precedence works with profile layer, doctor shows profile status.

---

## Phase 5: US3 — Staleness Warnings and Strict Mode (Priority: P3)

**Goal**: When a cached profile's staleness is detected (driver version changed, backend updated, or profile age exceeds threshold), the TUI and doctor display a stale profile warning. Strict profiles mode (`--strict-profiles`) is documented but deferred to post-MVP.

**Independent Test**: Can be fully tested by creating an intentionally stale profile (e.g., mismatched driver version hash) and verifying that the warning surfaces in TUI and doctor output.

### Tests for US3 (REQUIRED) ⚠️

- [x] T048 [P] [US3] Write `src/tests/test_profile_cache.py` additions — test `check_staleness` with all three staleness conditions returns correct `StalenessResult` with appropriate `StalenessReason` values
- [x] T049 [P] [US3] Write `src/tests/test_profile_cache.py` additions — test `StalenessResult.warning_message` property returns human-readable warning string for each staleness reason

### Implementation for US3

- [x] T050 [US3] Ensure TUI stale warning badge displays at refresh cycle (not blocking) — already implemented in US1 phase, verify it works with doctor output
- [x] T051 [US3] Ensure doctor profile staleness output includes actionable guidance (e.g., "Re-profile recommended: driver version changed from X to Y")
- [x] T052 [US3] Document `--strict-profiles` post-MVP deferral in `src/llama_cli/cli_parser.py` comments — stale profiles would block launch in strict mode, MVP only warns
- [x] T053 [US3] Add edge case handling for profile corruption — corrupt/unparsable profiles logged and left on disk (not auto-deleted), treated as non-existent with defaults used

**Checkpoint**: US3 complete — stale profiles surface warnings in TUI and doctor, corruption handled gracefully, strict mode documented.

---

## Phase 6: US4 — Deterministic Override Precedence (Priority: P1)

**Goal**: User provides explicit CLI or config overrides. Override precedence: repo defaults < slot configuration < workstation configuration < profile guidance < explicit override. Explicit overrides always win. The merge is deterministic and documented.

**Independent Test**: Can be fully tested by providing various combinations of defaults, profile values, and explicit overrides, then verifying the final merged ServerConfig reflects correct precedence.

### Tests for US4 (REQUIRED) ⚠️

- [x] T054 [P] [US4] Write `src/tests/test_us3_precedence.py` additions — test explicit override (e.g., `--threads=12`) wins over profile guidance (`threads=8`), profile guidance applied when no explicit override, all five precedence levels produce deterministic output
- [x] T055 [P] [US4] Write `src/tests/test_us3_precedence.py` additions — test `merge_config_overrides` with profile_config filtering — non-whitelisted fields (n_gpu_layers, tensor_split, model, port, server_bin, backend) are NOT applied from profile

### Implementation for US4

- [x] T056 [US4] Verify `merge_config_overrides()` in `src/llama_manager/config_builder.py` correctly implements all five precedence levels: defaults < slot < workstation < profile < override
- [x] T057 [US4] Ensure `PROFILE_OVERRIDE_FIELDS` whitelist is enforced — only threads, ctx_size, ubatch_size, cache_type_k, cache_type_v can be overridden by profile
- [x] T058 [US4] Document merge precedence chain in `config_builder.py` docstring with example showing all five levels
- [x] T059 [US4] Add CLI parser tests to `src/tests/test_cli_parser.py` — profile subcommand parsing with slot_id, flavor, --json flag

**Checkpoint**: US4 complete — override precedence is deterministic, whitelist prevents structural config override, all levels work correctly.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T060 [P] Run `uv run ruff check .` — zero lint errors
- [x] T061 [P] Run `uv run ruff format .` — all files formatted
- [x] T062 [P] Run `uv run pyright` — zero type errors
- [x] T063 [P] Run `uv run pytest` — all tests pass
- [x] T064 [P] Run `uv run pytest --cov --cov-report=term-missing` — no unexpected coverage gaps
- [x] T065 [P] Run `uv run pre-commit run --all-files` — all pre-commit hooks pass
- [x] T066 [P] Verify `llama_manager/` has no subprocess at module level, no Rich imports, no argparse — grep for violations
- [x] T067 [P] Verify `llama_cli/` does not import from `llama_manager` in reverse — one-way dependency preserved
- [x] T068 [US2] Verify atomic write pattern: profile files written with temp file + rename, owner-only permissions (0o600), directory created with 0o700
- [x] T069 [US1] Verify TUI never calls `console.print()` while `Live` is active — use layout updates instead
- [x] T070 [US4] Verify deterministic merge: identical inputs produce identical merged ServerConfig on repeated runs
- [x] T071 [US1] Verify benchmark command constructed as `list[str]` (never shell string), passed to subprocess with `shell=False`
- [x] T072 [US2] Verify profile cache directory path follows XDG pattern: `~/.cache/llm-runner/profiles/`
- [x] T073 [US3] Verify stale profile warning does NOT block model launch in MVP — only warning, no hard block
- [x] T074 [US4] Verify `--strict-profiles` is documented as post-MVP deferral, not implemented in MVP

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 0)**: No dependencies — can start immediately
- **Foundational (Phase 1)**: Depends on Setup completion — BLOCKS all user stories
- **Benchmark Module (Phase 2)**: Depends on Foundational (Phase 1) — pure library, no user story dependencies
- **US1 (Phase 3)**: Depends on Foundational (Phase 1) + Benchmark Module (Phase 2) — MVP delivery
- **US2 (Phase 4)**: Depends on Foundational (Phase 1) + US1 (Phase 3) — profile persistence + merge integration
- **US3 (Phase 5)**: Depends on US2 (Phase 4) — staleness warnings build on persistence
- **US4 (Phase 6)**: Depends on US2 (Phase 4) — precedence builds on merge integration
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational (Phase 1) + Benchmark Module (Phase 2) — MVP, core data-gathering flow
- **US2 (P2)**: Depends on US1 (Phase 3) — persistence builds on profiling data
- **US3 (P3)**: Depends on US2 (Phase 4) — staleness warnings need persistence
- **US4 (P1)**: Depends on US2 (Phase 4) — precedence needs merge integration

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Data model before business logic
- Business logic before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Foundational tasks marked [P] can run in parallel (T004, T005, T021, T022, T023)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members (after Foundational)
- All Polish tasks marked [P] can run in parallel

### Recommended Commit Order

1. Phase 1: Data model + cache layer (foundation, no breaking changes)
2. Phase 2: Benchmark module (pure library, no dependencies)
3. Phase 3: US1 — Manual profiling from TUI (MVP delivery)
4. Phase 4: US2 — Profile persistence and cache lookup
5. Phase 5: US3 — Staleness warnings and strict mode
6. Phase 6: US4 — Deterministic override precedence
7. Phase 7: Polish & validation

---

## CI Gate Checklist

Before any phase is considered complete, verify:

```bash
uv run ruff check .                    # Lint — zero errors
uv run ruff format --check .           # Format — all files formatted
uv run pyright                         # Type check — zero errors
uv run pytest                          # Tests — all pass
uv run pytest --cov --cov-report=term-missing  # Coverage — no unexpected gaps
```

**Pre-commit hooks** (if installed):

```bash
uv run pre-commit run --all-files
```

---

## Risk/Mitigation Summary

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Return type change breaks callers | High | Non-breaking `warnings` side-effect parameter (Phase 4) |
| Benchmark output format drift | Medium | Defensive parsing — returns `None` on failure (Phase 2) |
| Subprocess in llama_manager | Critical | `build_benchmark_cmd` in library, subprocess in `profile_cli.py` only |
| Race condition on profile writes | Medium | Atomic write pattern (temp file + rename) (Phase 1) |
| Path traversal in filename | Critical | Sanitize all filename components (Phase 1) |
| Unvalidated benchmark binary path | Critical | `require_executable()` before execution (Phase 3) |
| CI pyright errors on new types | Medium | Annotate all new functions; run `pyright` before committing |
| ruff lint errors on new code | Low | Run `ruff check --fix` before committing |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
