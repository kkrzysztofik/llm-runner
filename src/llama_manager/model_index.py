"""Disk cache for scanned GGUF model metadata.

Scans ``Config.models_dir`` for ``*.gguf`` files, extracts lightweight metadata,
and caches results atomically in a JSON file.
"""

import contextlib
import json
import logging
import multiprocessing
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from threading import Event
from typing import Any, cast

from .config.defaults import Config
from .metadata import GGUFMetadataRecord, extract_gguf_metadata

logger = logging.getLogger(__name__)

ModelIndexProgressCallback = Callable[[list["ModelIndexEntry"], int, int, int], None]
_INDEX_SCHEMA_VERSION = 2
_METADATA_WORKER_TIMEOUT_GRACE_S = 1.0


def _extract_metadata_worker(
    model_path: str,
    prefix_cap_bytes: int,
    parse_timeout_s: float,
    result_queue: Any,
) -> None:
    """Run GGUF metadata extraction in a child process for isolation."""
    try:
        result_queue.put(
            (
                "ok",
                extract_gguf_metadata(
                    model_path,
                    prefix_cap_bytes=prefix_cap_bytes,
                    parse_timeout_s=parse_timeout_s,
                ),
            )
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _extract_model_index_metadata(
    model_path: str,
    prefix_cap_bytes: int,
    parse_timeout_s: float,
) -> GGUFMetadataRecord:
    """Extract model-index metadata outside the main process."""
    timeout_s = max(float(parse_timeout_s), 0.1)
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    ctx = cast(Any, multiprocessing.get_context(start_method))
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_extract_metadata_worker,
        args=(model_path, prefix_cap_bytes, timeout_s, result_queue),
    )
    process.start()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join(_METADATA_WORKER_TIMEOUT_GRACE_S)
        if process.is_alive():
            process.kill()
            process.join(_METADATA_WORKER_TIMEOUT_GRACE_S)
        raise TimeoutError(f"GGUF metadata parse timed out after {timeout_s}s for {model_path}")

    try:
        status, payload = result_queue.get(timeout=0.1)
    except Empty:
        raise RuntimeError(
            f"GGUF metadata worker exited without a result for {model_path} "
            f"(exit code {process.exitcode})"
        ) from None

    if status == "ok" and isinstance(payload, GGUFMetadataRecord):
        return payload
    raise RuntimeError(str(payload))


@dataclass
class ModelIndexEntry:
    """One entry in the model index cache."""

    path: str
    normalized_stem: str
    general_name: str | None
    architecture: str | None
    file_type: int | None
    quantization_type: str | None
    context_length: int | None
    embedding_length: int | None
    block_count: int | None
    file_size_bytes: int
    parse_error: str | None
    mtime_iso: str
    max_context_length: int | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> ModelIndexEntry:
        """Deserialize from a JSON-safe dict."""
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields.setdefault("max_context_length", fields.get("context_length"))
        return cls(**fields)


def _file_mtime_iso(path: str) -> str:
    """Return the file mtime as an ISO timestamp string."""
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=UTC).isoformat()


def model_index_path(config: Config) -> Path:
    """Return the path to the model index JSON cache file.

    Returns ``$XDG_CACHE_HOME/llm-runner/model-index.json`` (default
    ``~/.cache/llm-runner/model-index.json``), creating the parent
    directory if needed.

    Args:
        config: The application Config instance.

    Returns:
        Path object pointing at the index JSON file.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    idx_dir = Path(xdg_cache) / "llm-runner"
    idx_dir.mkdir(parents=True, exist_ok=True)
    return idx_dir / "model-index.json"


def load_model_index(config: Config) -> list[ModelIndexEntry]:
    """Load cached model index from disk.

    Returns an empty list if the file is missing, corrupt, or has a
    stale schema version (forcing a full rescan).

    Args:
        config: The application Config instance.

    Returns:
        List of ``ModelIndexEntry`` objects, or ``[]`` on failure.
    """
    idx_path = model_index_path(config)
    if not idx_path.exists():
        return []
    try:
        raw = idx_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            # Old format: plain JSON array
            logger.info(
                "Model index is in legacy format (array); forcing rescan",
            )
            return []
        schema_version = data.get("schema_version", 0)
        if schema_version < _INDEX_SCHEMA_VERSION:
            logger.info(
                "Model index schema version %d is stale (current: %d); forcing rescan",
                schema_version,
                _INDEX_SCHEMA_VERSION,
            )
            return []
        items: list[dict] = data.get("entries", [])
        return [ModelIndexEntry.from_dict(item) for item in items]
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Failed to load model index %s: %s", idx_path, exc)
        return []


def refresh_model_index(
    config: Config,
    cancel_event: Event | None = None,
    progress_callback: ModelIndexProgressCallback | None = None,
    *,
    progressive: bool = False,
) -> tuple[list[ModelIndexEntry], int, int]:
    """Scan ``config.models_dir`` for ``*.gguf`` files and rebuild the index.

    Uses the existing cache to skip files whose ``mtime_iso`` hasn't changed.
    Writes atomically via a temp file + rename.

    Args:
        config: The application Config instance.
        cancel_event: If set and not ``None``, stops scanning early.
        progress_callback: Called after each scanned file with
            ``(entries_so_far, total_scanned, total_models, error_count)``.
        progressive: If ``True``, write the cache after each scanned file so
            other readers can consume partial results while indexing continues.

    Returns:
        A tuple of ``(entries, total_scanned, error_count)`` where
        *entries* is the complete sorted list.
    """
    models_dir = Path(config.models_dir)
    if not models_dir.is_dir():
        logger.debug("models_dir %s not a directory, skipping index", config.models_dir)
        return ([], 0, 0)

    old_index = load_model_index(config)
    # Build lookup by absolute path for fast mtime comparison
    old_lookup: dict[str, ModelIndexEntry] = {e.path: e for e in old_index}

    entries: list[ModelIndexEntry] = []
    total_scanned = 0
    error_count = 0

    # Collect all gguf files (case-insensitive) so we can check cancel before each parse
    gguf_files: list[Path] = sorted(models_dir.rglob("**/*"))
    unique_files: list[Path] = []
    seen: set[str] = set()
    for p in gguf_files:
        if not p.is_file() or p.suffix.lower() != ".gguf":
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(p)

    if not old_lookup:
        logger.info(
            "model index: cache empty — full rescan of %d GGUF file(s) in %s",
            len(unique_files),
            config.models_dir,
        )
    else:
        logger.info(
            "model index: scanning %d GGUF file(s) in %s (cache: %d entries)",
            len(unique_files),
            config.models_dir,
            len(old_lookup),
        )

    for file_path in unique_files:
        if cancel_event is not None and cancel_event.is_set():
            break

        abs_path = str(file_path.resolve())
        mtime_iso = _file_mtime_iso(abs_path)
        file_size = file_path.stat().st_size

        # Skip if mtime matches cached entry
        if abs_path in old_lookup and old_lookup[abs_path].mtime_iso == mtime_iso:
            entries.append(old_lookup[abs_path])
            total_scanned += 1
            logger.debug("model index: cache hit %s", abs_path)
            continue

        # Extract metadata (may fail) — uses subprocess for isolation
        try:
            meta: GGUFMetadataRecord = _extract_model_index_metadata(
                abs_path,
                prefix_cap_bytes=config.gguf_metadata_prefix_cap_bytes,
                parse_timeout_s=config.gguf_metadata_parse_timeout_s,
            )
            entries.append(
                ModelIndexEntry(
                    path=abs_path,
                    normalized_stem=meta.normalized_stem,
                    general_name=meta.general_name,
                    architecture=meta.architecture,
                    file_type=meta.file_type,
                    quantization_type=meta.quantization_type,
                    context_length=meta.context_length,
                    max_context_length=meta.max_context_length or meta.context_length,
                    embedding_length=meta.embedding_length,
                    block_count=meta.block_count,
                    file_size_bytes=file_size,
                    parse_error=None,
                    mtime_iso=mtime_iso,
                )
            )
            logger.info(
                "model index: parsed %s — %s (%s)",
                abs_path,
                meta.general_name or meta.normalized_stem,
                meta.architecture or "unknown",
            )
        except Exception as exc:
            error_count += 1
            logger.warning(
                "model index: parse failed for %s: %s; falling back to filename metadata",
                abs_path,
                exc,
            )
            fallback_meta = _metadata_from_filename(abs_path, file_path.stem)
            entries.append(
                ModelIndexEntry(
                    path=abs_path,
                    normalized_stem=fallback_meta.normalized_stem,
                    general_name=None,
                    architecture=fallback_meta.architecture,
                    file_type=None,
                    quantization_type=fallback_meta.quantization_type,
                    context_length=None,
                    max_context_length=None,
                    embedding_length=None,
                    block_count=None,
                    file_size_bytes=file_size,
                    parse_error=str(exc),
                    mtime_iso=mtime_iso,
                )
            )

        total_scanned += 1
        _emit_model_index_progress(
            config,
            entries,
            total_scanned,
            len(unique_files),
            error_count,
            progress_callback,
            progressive=progressive,
        )

    # Sort by normalized_stem alphabetically
    entries.sort(key=lambda e: e.normalized_stem)

    _write_model_index(config, entries)

    logger.info(
        "model index: done — %d entries, %d scanned, %d errors",
        len(entries),
        total_scanned,
        error_count,
    )
    return (entries, total_scanned, error_count)


def _emit_model_index_progress(
    config: Config,
    entries: list[ModelIndexEntry],
    total_scanned: int,
    total_models: int,
    error_count: int,
    progress_callback: ModelIndexProgressCallback | None,
    *,
    progressive: bool,
) -> None:
    """Publish one incremental scan update."""
    snapshot = sorted(entries, key=lambda e: e.normalized_stem)
    if progressive:
        _write_model_index(config, snapshot)
    if progress_callback is not None:
        progress_callback(snapshot, total_scanned, total_models, error_count)


def _write_model_index(config: Config, entries: list[ModelIndexEntry]) -> None:
    """Write model index entries atomically to disk."""
    idx_path = model_index_path(config)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(idx_path.parent),
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(
                {
                    "schema_version": _INDEX_SCHEMA_VERSION,
                    "entries": [e.to_dict() for e in entries],
                },
                tmp,
                default=str,
                indent=2,
            )
            tmp_path = tmp.name
        os.replace(tmp_path, str(idx_path))
    except OSError as exc:
        logger.warning("Failed to write model index %s: %s", idx_path, exc)
        # Clean up temp file if it exists
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def _metadata_from_filename(path: str, stem: str) -> GGUFMetadataRecord:
    """Build best-effort metadata from a model filename after parse failure."""
    import re

    normalized = stem
    quant_match = re.search(r"(IQ\d_[A-Z]+|Q\d_[A-Z_]+|F16|F32)", stem)
    arch_match = re.search(r"(qwen3|qwen2|qwen|llama|mistral|phi3|phi)", stem, re.I)
    return GGUFMetadataRecord(
        raw_path=path,
        normalized_stem=normalized,
        architecture=arch_match.group(1).lower() if arch_match else None,
        quantization_type=quant_match.group(1) if quant_match else None,
    )
