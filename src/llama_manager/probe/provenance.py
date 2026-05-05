"""Probe provenance — resolve git SHA and package version for smoke probes."""

from dataclasses import dataclass
from importlib.metadata import version as _importlib_version
from pathlib import Path
from subprocess import CalledProcessError

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

    Reads the SHA from ``.git/HEAD`` in the llama.cpp root directory
    and the package version from ``importlib.metadata``.

    Returns:
        A ProvenanceRecord with sha and version.

    """
    sha = _resolve_sha()
    version = _resolve_version()
    return ProvenanceRecord(sha=sha, version=version)


def _resolve_sha() -> str:
    """Resolve the git SHA from the llama.cpp repository.

    Reads ``.git/HEAD`` and runs ``git rev-parse`` to get the full SHA.

    Returns:
        Full git SHA, or 'unknown' if resolution fails.

    """
    import subprocess

    cfg = Config()
    llama_cpp_root = cfg.llama_cpp_root
    git_head = Path(llama_cpp_root) / ".git" / "HEAD"
    if not git_head.exists():
        return "unknown"

    try:
        head_content = git_head.read_text().strip()
        # .git/HEAD can contain a ref (ref: refs/heads/main) or a direct SHA
        if head_content.startswith("ref: "):
            ref_path = git_head.parent / head_content[5:]
            if ref_path.exists():
                sha = ref_path.read_text().strip()
                return sha[:7] if len(sha) > 7 else sha
        else:
            # Direct SHA reference
            return head_content[:7] if len(head_content) > 7 else head_content
    except OSError:
        pass

    # Fallback: try git rev-parse
    try:
        result = subprocess.run(
            ["git", "-C", llama_cpp_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha[:7] if len(sha) > 7 else sha
    except (FileNotFoundError, CalledProcessError, subprocess.TimeoutExpired, TimeoutError):
        pass

    return "unknown"


def _resolve_version() -> str:
    """Resolve the package version from importlib.metadata.

    Returns:
        Package version string, or 'dev' if unavailable.

    """
    try:
        return _importlib_version("llm_runner")
    except Exception:
        return "dev"
