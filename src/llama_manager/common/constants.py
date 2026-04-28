"""Shared filesystem permission constants used across llama_manager."""

from typing import Final

# File and directory mode for owner-only access (chmod 600 / 700).
FILE_MODE_OWNER_ONLY: Final[int] = 0o600
DIR_MODE_OWNER_ONLY: Final[int] = 0o700
