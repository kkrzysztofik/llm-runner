# Pipeline stages

Implementation: `llama_manager/build_pipeline/pipeline.py` orchestrates stages in `stages/`. Each stage returns a `BuildProgress` dataclass; failures before configure/build complete return an error `BuildResult` without a failure artifact. Configure and build failures call `write_failure_artifact()`.

## Retry wrapper

`_run_with_retry()` runs each stage function up to `config.retry_attempts` times (default **3** from `BuildConfig`, CLI default **2**). On failure:

1. Emit `BuildProgress` with `status="retrying"` to the optional callback (TUI).
2. Sleep `retry_delay * 2^attempt` seconds (default base delay **5s**).
3. Retry until attempts exhausted.

Stages 1–2 (preflight, clone): failure → immediate `BuildResult(success=False)` with no failure report directory.

Stages 3–4 (configure, build): failure → failure artifact + report under `reports/`.

## Stage 1: preflight

**Module:** `stages/preflight.py`  
**Progress:** 0% → 20% on success

Calls `toolchain.detect_toolchain()` and checks backend readiness:

| Backend | Required (`ToolchainStatus`) |
|---------|------------------------------|
| SYCL | `gcc`, `make`, `git`, `cmake`, `sycl_compiler` (first found among `icpx`, `icx`, `dpcpp`) |
| CUDA | `gcc`, `make`, `git`, `cmake`, `cuda_toolkit` (`nvcc`) |

`missing_tools(cuda)` also lists `nvtop` when absent. nvtop is used for GPU monitoring, not compilation; preflight for **build** only checks `is_cuda_ready` (nvcc), not nvtop.

**Dry-run:** Still runs detection (no subprocess build).

**Failure message example:** `Missing SYCL tools: cmake, sycl_compiler`

Workstation install guide: [workstation-setup.md](workstation-setup.md).

## Stage 2: clone

**Module:** `stages/clone.py`  
**Progress:** ~20–30%

### When source already exists

| Condition | Behavior |
|-----------|----------|
| Valid `.git` + `update_sources=True` | `git fetch` / pull workflow via `_update_sources` |
| Valid `.git` + `git_commit` set | Checkout that commit |
| Valid `.git` + no update, no commit pin | **Skipped** — message: sources already exist |
| Exists but not a git repo | **Failed** — user must remove or point `--source-dir` elsewhere |

### Fresh clone

```bash
git clone --branch <git_branch> [--depth 1] <git_remote_url> <source_dir>
```

- Shallow clone default: `--depth 1` unless `--no-shallow-clone`.
- Timeout: `clone_timeout` attribute if set, else **120s** on clone subprocess.
- After clone, optional `git checkout <git_commit>`.

### Offline continue

If clone fails but a non-empty source directory already existed, status becomes **skipped** with “sources already exist” so configure can proceed offline.

### Dry-run — clone

Prints the `git clone` command that would run; does not touch disk.

## Stage 3: configure

**Module:** `stages/configure.py`  
**Progress:** 30% → 50%

### Skip rule

If `build_dir/CMakeCache.txt` exists and `update_sources=False`, configure is **skipped** (“Already configured”). Use `--no-update-sources` only when you intentionally want to reuse a prior CMake cache without re-running cmake.

### Command

```bash
cmake -S <source_dir> -B <build_dir> <backend flags>
```

SYCL builds wrap the command:

```bash
bash -c 'source "/opt/intel/oneapi/setvars.sh" && cmake ...'
```

only when `setvars.sh` exists (`build_pipeline/utils.py`). If missing, cmake runs unwrapped (likely fails without `icx`/`icpx` on PATH).

**Timeout:** `build_timeout_seconds` (default **3600**).

### Dry-run — configure

Returns success with message `Would run: <full command>` without creating `build_dir`.

CMake flags: [cmake-and-backends.md](cmake-and-backends.md).

## Stage 4: build

**Module:** `stages/build.py`  
**Progress:** 50% → 75%

### Command

```bash
cmake --build <build_dir> [-j N]
```

Same SYCL `setvars.sh` wrapper as configure. Uses `subprocess.Popen` with line-by-line streaming to `progress_callback` (TUI log pane).

**Timeout:** Same `build_timeout_seconds` as configure.

### Dry-run — build

Success with `Would run: cmake --build ...`.

## Stage 5: finalize

**Module:** `stages/finalize.py`  
**Progress:** completes after successful build stage

Only runs when the build stage completed with `status="success"`.

1. `git rev-parse HEAD` in `source_dir` → `git_commit_sha` (or `"unknown"`).
2. Look for `<build_dir>/bin/llama-server`; record size.
3. Write consolidated log via `_BuildContext.write_build_log()` → `reports/<timestamp>-<backend>.log`.
4. Build `BuildArtifact` and atomically write `output_dir/build-artifact.json`.

Does not install or copy the binary elsewhere.

## Lock lifecycle

Before stages: `acquire_lock(Config().build_lock_path, backend)` (skipped in dry-run).

After run (success or failure): `release_lock()` in `finally`.

Stale lock: PID not running (psutil) or lock age > **3600s** (`BuildLock.is_stale()`).

## Progress callback (TUI)

`BuildPipeline(..., progress_callback=fn)` receives every stage completion and retry, plus per-line `output_line` during compile. CLI does not pass a callback; it prints stage outcome after `run()` returns.
