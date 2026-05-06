"""Shared toolchain helpers for CLI commands.

Provides backend resolution and toolchain hint retrieval used by
setup and doctor commands. Consumers should import from this module
rather than duplicating backend/hints logic.
"""

from llama_manager.build_pipeline import BuildBackend
from llama_manager.toolchain import (
    ToolchainErrorDetail,
    get_toolchain_hints,
)


def resolve_backend_enum(backend_str: str | None) -> BuildBackend | None:
    """Convert backend string to BuildBackend enum.

    Args:
        backend_str: Backend string (e.g., "sycl", "cuda", "all") or None.

    Returns:
        BuildBackend enum value, or None if input is None or invalid.
    """
    if backend_str is None:
        return None
    try:
        return BuildBackend(backend_str)
    except ValueError:
        return None


def deduplicate_hints(hints: list[ToolchainErrorDetail]) -> list[ToolchainErrorDetail]:
    """Deduplicate hints by how_to_fix + docs_ref.

    Args:
        hints: List of toolchain hints to deduplicate.

    Returns:
        Deduplicated list of hints.
    """
    seen: set[tuple[str, str | None]] = set()
    result: list[ToolchainErrorDetail] = []
    for hint in hints:
        key = (hint.how_to_fix, hint.docs_ref)
        if key not in seen:
            seen.add(key)
            result.append(hint)
    return result


def get_backend_hints(backend: str) -> list[ToolchainErrorDetail]:
    """Get toolchain hints for specified backend with deduplication.

    Args:
        backend: Backend filter ("sycl", "cuda", or "all").

    Returns:
        Deduplicated list of toolchain hints.
    """
    if backend == "sycl":
        return deduplicate_hints(get_toolchain_hints("sycl"))
    if backend == "cuda":
        return deduplicate_hints(get_toolchain_hints("cuda"))
    if backend == "all":
        sycl_hints = get_toolchain_hints("sycl")
        cuda_hints = get_toolchain_hints("cuda")
        return deduplicate_hints(sycl_hints + cuda_hints)
    return []


def filter_optional_tools(
    missing: list[str],
    backend: str | None,
    is_complete: bool,
) -> list[str]:
    """Filter out optional tools when all backends are complete.

    Args:
        missing: List of missing tool names.
        backend: Backend filter ("sycl", "cuda", "all", or None).
        is_complete: Whether the toolchain is fully complete.

    Returns:
        Filtered list of missing tools.
    """
    if (backend == "all" or backend is None) and is_complete:
        return [t for t in missing if t != "nvtop"]
    return list(missing)


def collect_toolchain_repair_actions(
    result_hints: list[ToolchainErrorDetail],
) -> list[ToolchainErrorDetail]:
    """Deduplicate toolchain hints by failed_check for repair actions.

    Used by doctor repair to collect unique toolchain hints.

    Args:
        result_hints: Combined toolchain hints from all backends.

    Returns:
        Deduplicated hints keyed by failed_check.
    """
    seen_tools: set[str] = set()
    hints: list[ToolchainErrorDetail] = []
    for hint in result_hints:
        if hint.failed_check not in seen_tools:
            seen_tools.add(hint.failed_check)
            hints.append(hint)
    return hints
