"""Atomic filesystem helpers used across llama_manager.

These replace the duplicate ``_atomic_write_json`` / ``_fsync_directory``
implementations that previously lived in both ``process_manager`` and
``profile_cache``.
"""

import contextlib
import json
import os
import tempfile
from pathlib import Path

from .constants import FILE_MODE_OWNER_ONLY


def fsync_directory(dir_path: Path) -> None:
    """fsync *dir_path* to make its directory entries durable on disk.

    Best-effort — ``OSError`` is silently suppressed because some
    filesystems (e.g. tmpfs, some network mounts) do not support
    directory-level fsync.

    Args:
        dir_path: Directory to sync.
    """
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_exclusive_create_json(
    path: Path,
    data: dict,
    mode: int = FILE_MODE_OWNER_ONLY,
    *,
    fsync_after: bool = True,
) -> None:
    """Create *path* exclusively with JSON content (``O_CREAT | O_EXCL``).

    Raises ``FileExistsError`` immediately if *path* exists — the correct
    primitive for lock-file creation where existence means another owner.
    """
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        os.write(fd, (json.dumps(data, indent=2) + "\n").encode("utf-8"))
        if fsync_after:
            os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: Path,
    data: dict,
    mode: int = FILE_MODE_OWNER_ONLY,
    *,
    verify_permissions: bool = False,
) -> None:
    """Atomically write *data* as JSON to *path*.

    Implementation:

    1. Create a temp file in the same directory (``tempfile.mkstemp``).
    2. Serialize JSON, flush, and ``fsync`` the file descriptor.
    3. Set file permissions to *mode* before rename.
    4. ``os.replace`` for an atomic rename over *path*.
    5. ``fsync`` the parent directory so the entry is durable.
    6. Optionally re-stat and verify the final permissions match *mode*.

    Args:
        path: Destination file path.  Must be on the same filesystem as
              its parent directory so ``os.replace`` is atomic.
        data: Dictionary to serialize as JSON.
        mode: File permission bits applied before rename (default: ``0o600``).
        verify_permissions: When ``True``, re-stat after rename and raise
                            ``OSError`` if the effective mode differs from
                            *mode*.

    Raises:
        OSError: On write, sync, rename, or (when requested) permission
                 mismatch.
        TypeError: If *data* is not JSON-serializable.
    """
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".json",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.chmod(str(tmp_path), mode)
        os.replace(str(tmp_path), str(path))
        fsync_directory(path.parent)

        if verify_permissions:
            current_mode = os.stat(path).st_mode & 0o777
            if current_mode != mode:
                raise OSError(
                    f"file permissions mismatch after write: "
                    f"expected {mode:o}, got {current_mode:o}",
                )
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(str(tmp_path))
        raise
