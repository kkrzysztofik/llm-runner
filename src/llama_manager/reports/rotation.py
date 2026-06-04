"""Report rotation — clean old report directories by timestamp."""

from __future__ import annotations

import contextlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


def rotate_reports(config: Config | None = None) -> None:
    """Rotate old report directories.

    Scans the reports directory and deletes oldest report directories
    when count exceeds Config.build_max_reports. Uses FIFO rotation
    (oldest first).

    Args:
        config: Optional Config instance. If not provided, creates default.
    """
    from ..config import Config

    cfg = config if config is not None else Config()

    reports_path = cfg.paths.reports_dir

    if not reports_path.exists():
        return

    # Get all report directories (directories starting with timestamp pattern)
    report_dirs: list[Path] = []
    for entry in reports_path.iterdir():
        if entry.is_dir() and entry.name.startswith("20"):
            # Check if it looks like a timestamp directory (YYYYMMDD_HHMMSS)
            try:
                datetime.strptime(entry.name, "%Y%m%d_%H%M%S")
                report_dirs.append(entry)
            except ValueError:
                continue

    # Sort by directory name (timestamp-based, oldest first)
    report_dirs.sort(key=lambda p: p.name)

    # Delete oldest directories if count exceeds max
    max_reports = cfg.build.max_reports
    if len(report_dirs) > max_reports:
        to_delete = report_dirs[: len(report_dirs) - max_reports]
        for report_dir in to_delete:
            with contextlib.suppress(OSError):
                shutil.rmtree(report_dir)


def _rotate_mutating_log(log_path: Path, max_entries: int = 1000) -> None:
    """Rotate mutating action log if it exceeds max entries.

    Deletes the oldest entry when the log file exceeds the maximum number
    of entries. Uses simple line-based rotation.

    Args:
        log_path: Path to the log file
        max_entries: Maximum number of entries before rotation
    """
    if not log_path.exists():
        return

    try:
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) <= max_entries:
            return

        # Keep only the last max_entries lines
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_entries:])

    except OSError:
        # Log but don't fail on rotation errors
        pass
