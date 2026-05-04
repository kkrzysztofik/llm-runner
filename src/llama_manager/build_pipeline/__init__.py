"""build_pipeline package — public API for the llama.cpp build pipeline.

Re-exports all symbols that were previously available from the
``llama_manager.build_pipeline`` module so that all existing callers
continue to work without changes.
"""

from .lock import acquire_lock, get_lock_error_message, is_lock_stale, release_lock
from .models import (
    BuildArtifact,
    BuildBackend,
    BuildConfig,
    BuildLock,
    BuildProgress,
    BuildResult,
)
from .orchestration import run_build_for_backend
from .pipeline import BuildPipeline

__all__ = [
    # Models
    "BuildArtifact",
    "BuildBackend",
    "BuildConfig",
    "BuildLock",
    "BuildProgress",
    "BuildResult",
    # Pipeline
    "BuildPipeline",
    "run_build_for_backend",
    # Lock utilities
    "acquire_lock",
    "get_lock_error_message",
    "is_lock_stale",
    "release_lock",
]
