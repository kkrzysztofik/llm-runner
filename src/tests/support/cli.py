"""CLI test helpers."""

from argparse import Namespace
from pathlib import Path
from typing import Any


def build_command_args(tmp_path: Path, **overrides: Any) -> Namespace:
    """Create a minimal argparse Namespace for build command tests."""
    defaults: dict[str, Any] = {
        "backend": "cuda",
        "source_dir": tmp_path / "source",
        "build_dir": tmp_path / "build",
        "output_dir": tmp_path / "output",
        "git_remote": "https://github.com/ggerganov/llama.cpp",
        "git_branch": "main",
        "git_commit": None,
        "jobs": 1,
        "json": False,
        "dry_run": False,
        "no_shallow_clone": False,
        "no_update_sources": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def namespace(**overrides: Any) -> Namespace:
    """Create a generic argparse Namespace."""
    return Namespace(**overrides)
