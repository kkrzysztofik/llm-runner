# Virtual environment setup for M2 build environment

import os
import venv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VenvResult:
    """Result of virtual environment creation or validation.

    This dataclass captures the outcome of venv setup operations,
    including whether a new environment was created or an existing
    one was reused, and provides the activation command for the user.
    """

    venv_path: Path
    created: bool
    reused: bool
    activation_command: str

    @property
    def was_created(self) -> bool:
        """Check if a new virtual environment was created."""
        return self.created and not self.reused

    @property
    def was_reused(self) -> bool:
        """Check if an existing virtual environment was reused."""
        return self.reused and not self.created


def get_venv_path() -> Path:
    """Return the managed virtual environment path.

    Always returns the managed venv location at $XDG_CACHE_HOME/llm-runner/venv
    (or ~/.cache/llm-runner/venv). Does not honor VIRTUAL_ENV to ensure
    setup/doctor only operate on the managed venv.

    Returns:
        Path to the managed virtual environment.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(xdg_cache) / "llm-runner" / "venv"


def create_venv(path: str | Path) -> VenvResult:
    """Create a virtual environment at the given path.

    Args:
        path: Path where the virtual environment should be created (or reused)

    Returns:
        VenvResult instance with the venv path and creation status
    """
    venv_path = Path(path)
    created = False
    reused = False

    if venv_path.exists() and venv_path.is_dir():
        # Validate venv integrity before reusing
        is_valid, _ = check_venv_integrity(venv_path)
        if is_valid:
            reused = True
        else:
            # Invalid venv, verify it's actually a venv before removing
            import shutil

            if not (venv_path / "pyvenv.cfg").exists():
                raise ValueError(f"Path exists but is not a valid virtual environment: {venv_path}")
            shutil.rmtree(venv_path)
            venv.create(venv_path, with_pip=True, clear=False)
            created = True
    elif venv_path.exists() and not venv_path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {venv_path}")
    else:
        venv.create(venv_path, with_pip=True, clear=False)
        created = True

    activation_script = venv_path / "bin" / "activate"
    activation_command = f"source {activation_script}"

    return VenvResult(
        venv_path=venv_path,
        created=created,
        reused=reused,
        activation_command=activation_command,
    )


def check_venv_integrity(path: str | Path) -> tuple[bool, str | None]:
    """Validate virtual environment integrity.

    Args:
        path: Path to the virtual environment to validate

    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if the venv is valid
        - error_message: Error message if invalid, None if valid
    """
    venv_path = Path(path)

    # Check if venv directory exists first
    if not venv_path.exists():
        return (False, "venv directory missing")

    # Check if pyvenv.cfg exists
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if not pyvenv_cfg.exists():
        return (False, "pyvenv.cfg missing")

    # Check if interpreter exists
    interpreter = venv_path / "bin" / "python"

    if not interpreter.exists():
        return (False, "interpreter not found in venv")

    return (True, None)
