"""Probe provenance — resolve git SHA and package version for smoke probes."""

from dataclasses import dataclass
from importlib.metadata import version as _importlib_version

from ..config import Config


@dataclass
class ProvenanceRecord:
    """Git provenance for the running server binary.

    Attributes:
        sha: Full git SHA of the llama.cpp HEAD at build time.
        version: Package version from ``importlib.metadata``.
    """

    sha: str
    version: str


def resolve_provenance() -> ProvenanceRecord:
    """Resolve git provenance for the running server binary.

    Reads the SHA via ``git rev-parse HEAD`` in the llama.cpp root
    directory and the package version from ``importlib.metadata``.

    Returns:
        A ProvenanceRecord with sha and version.

    """
    sha = _resolve_sha()
    version = _resolve_version()
    return ProvenanceRecord(sha=sha, version=version)


def _resolve_sha() -> str:
    """Resolve the git SHA via ``git rev-parse HEAD``, or 'unknown'."""
    cfg = Config()
    llama_cpp_root = str(cfg.paths.llama_cpp_root)
    from subprocess import TimeoutExpired, run

    try:
        result = run(
            ["git", "-C", llama_cpp_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError, TimeoutExpired:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolve_version() -> str:
    """Resolve the package version from importlib.metadata.

    Returns:
        Package version string, or 'dev' if unavailable.

    """
    try:
        return _importlib_version("llm_runner")
    except Exception:
        return "dev"
