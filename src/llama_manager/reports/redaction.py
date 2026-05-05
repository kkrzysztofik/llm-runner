"""Redaction utilities for sensitive information in build output."""

import re

from ..common.security import REDACTED_VALUE


def redact_sensitive(text: str) -> str:
    """Redact sensitive information from text.

    Replaces values for KEY|TOKEN|SECRET|PASSWORD|AUTH with ``[REDACTED]``.
    The key name is preserved for readability; only the value is replaced.
    Uses case-insensitive regex matching.

    Args:
        text: Input text to redact.

    Returns:
        Text with sensitive values replaced by ``[REDACTED]``.

    Examples:
        >>> redact_sensitive("API_KEY=abc123")
        'API_KEY: [REDACTED]'
        >>> redact_sensitive("password: secret123")
        'password: [REDACTED]'
    """

    def replace_key_value(match: re.Match) -> str:
        full_key = match.group(1)
        return f"{full_key}: {REDACTED_VALUE}"

    # First pass: replace key=value and key: value constructs.
    # Match quoted values (single or double quotes) or unquoted values.
    pattern = (
        r"(?<!\w)(\w*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH)\w*)"
        r"([=:]\s*)"
        r'(?:"[^"]*"|'
        r"'[^']*'"
        r"|\S+)"
    )
    result = re.sub(pattern, replace_key_value, text, flags=re.IGNORECASE)

    # Second pass: replace standalone sensitive words that have no value after them.
    result = re.sub(
        r"(?<!\w)(\w*(KEY|TOKEN|SECRET|PASSWORD|AUTH)\w*)(?![=:]\s*\S+)(?![:\s]*"
        + re.escape(REDACTED_VALUE)
        + r")(?!\w)",
        REDACTED_VALUE,
        result,
        flags=re.IGNORECASE,
    )

    return result
