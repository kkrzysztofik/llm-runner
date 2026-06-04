"""Text normalization helpers shared across llama_manager modules."""

import re

_FILENAME_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_\-\.]")


def sanitize_filename_component(component: str) -> str:
    """Sanitize a string for safe use as one filesystem path component."""
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")

    sanitized = _FILENAME_SANITIZE_PATTERN.sub("_", component.strip().lower())
    if not sanitized:
        raise ValueError(
            "component must contain at least one valid character after "
            f"sanitization, got: {component!r}",
        )
    return sanitized
