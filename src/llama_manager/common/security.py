"""Centralized sensitive-data patterns and redaction helpers.

All modules that detect or redact credentials, tokens, or other sensitive
environment variable values should import from here rather than defining
their own patterns.  Extend ``_SENSITIVE_KEYWORDS`` to update every caller
at once.
"""

import re
from string.templatelib import Interpolation, Template
from typing import Any, Final

# ---------------------------------------------------------------------------
# Replacement marker
# ---------------------------------------------------------------------------

REDACTED_VALUE: Final[str] = "[REDACTED]"

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches a sensitive keyword appearing anywhere inside a key *name*.
# Used for per-variable checks (is the key itself sensitive?).
# Requires the keyword to be at a word boundary that is not preceded or
# followed by an uppercase letter or digit, so "MONKEY" (which contains "KEY"
# as a trailing substring) is not matched while "OPENAI_API_KEY_1" is.
SENSITIVE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH(?:_HEADER)?)(?:_[0-9A-Z]+)*(?![A-Z0-9])",
    re.IGNORECASE,
)

# Matches ``KEY=value`` constructs in plain text / log lines where the key
# contains a sensitive keyword.  Unquoted values only; see ``redact_text``
# for the fuller quoted-value and bearer-token handling.
# AUTH_HEADER is listed before AUTH so the longer token wins in alternation.
SENSITIVE_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z0-9_]*(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)(?:_[0-9A-Z]+)*(?![A-Z0-9])\s*=\s*\S+",
)

# Fallback: matches bare sensitive key names even without an assignment.
# Used as a second pass after ``SENSITIVE_WORD_PATTERN`` to catch residual hits.
SENSITIVE_KEY_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z0-9_]*(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|AUTH_HEADER)(?:_[0-9A-Z]+)*(?![A-Z0-9])",
)

# Pattern for ``KEY=value`` pairs in log-stream lines — unquoted values.
# Preserves the key name; only the value is replaced.
# Example: ``API_KEY=abc123`` → ``API_KEY=[REDACTED]``
_LOG_SENSITIVE_UNQUOTED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\b[A-Z0-9_]*(?<![A-Z0-9])(?:AUTH_HEADER|AUTH|KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)="
    r"(\S+)",
    re.IGNORECASE,
)

# Pattern for ``KEY="quoted value"`` in log-stream lines.
# Accepts backslash-escaped characters so e.g. ``val\"ue`` is fully captured.
# Example: ``API_KEY="ab c"`` → ``API_KEY=[REDACTED]``
_LOG_SENSITIVE_DOUBLE_QUOTED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\b[A-Z0-9_]*(?<![A-Z0-9])(?:AUTH_HEADER|AUTH|KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)="
    r'"(?:[^"\\]|\\.)*"',
    re.IGNORECASE,
)

# Pattern for ``KEY='single-quoted value'`` in log-stream lines.
# Example: ``API_KEY='ab c'`` → ``API_KEY=[REDACTED]``
_LOG_SENSITIVE_SINGLE_QUOTED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\b[A-Z0-9_]*(?<![A-Z0-9])(?:AUTH_HEADER|AUTH|KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)="
    r"'(?:[^'\\]|\\.)*'",
    re.IGNORECASE,
)

# Matches ``Authorization: Bearer <token>`` headers in log text.
_BEARER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z_]*Authorization\s*:\s*Bearer\s+\S+",
)

# Matches ``KEY="quoted value"`` in plain text (replaces entire match).
_QUOTED_DOUBLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'(?i)\b[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)\s*=\s*"[^"]*"',
)

# Matches ``KEY='single-quoted value'`` in plain text (replaces entire match).
_QUOTED_SINGLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)\s*=\s*'[^']*'",
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_sensitive_key(key: str) -> bool:
    """Return ``True`` if *key* is a name that indicates a sensitive value.

    Args:
        key: Environment variable name or configuration key.

    Returns:
        ``True`` if the key name contains a sensitive keyword.
    """
    return bool(SENSITIVE_KEY_PATTERN.search(key))


def redact_env_value(env_value: str, env_key: str) -> str:
    """Return *env_value* redacted when *env_key* is a sensitive name.

    This is the per-variable check used when building redacted environment
    dicts for dry-run artifacts and audit logs.

    Args:
        env_value: The value associated with the environment variable.
        env_key: The name of the environment variable.

    Returns:
        ``REDACTED_VALUE`` if *env_key* matches a sensitive pattern,
        otherwise *env_value* unchanged.

    Example:
        >>> redact_env_value("my_secret", "API_KEY")
        '[REDACTED]'
        >>> redact_env_value("/path/to/model", "MODEL_PATH")
        '/path/to/model'
    """
    if is_sensitive_key(env_key):
        return REDACTED_VALUE
    return env_value


def _redact_log_key_value(line: str) -> str:
    """Apply the three ``KEY=value`` redaction patterns in order."""
    line = _LOG_SENSITIVE_DOUBLE_QUOTED_PATTERN.sub(rf"\1={REDACTED_VALUE}", line)
    line = _LOG_SENSITIVE_SINGLE_QUOTED_PATTERN.sub(rf"\1={REDACTED_VALUE}", line)
    return _LOG_SENSITIVE_UNQUOTED_PATTERN.sub(rf"\1={REDACTED_VALUE}", line)


def _log_sensitive_match(text: str) -> re.Match[str] | None:
    """Return the first match of any ``KEY=value`` sensitive pattern."""
    return (
        _LOG_SENSITIVE_DOUBLE_QUOTED_PATTERN.search(text)
        or _LOG_SENSITIVE_SINGLE_QUOTED_PATTERN.search(text)
        or _LOG_SENSITIVE_UNQUOTED_PATTERN.search(text)
    )


def redact_log_line(line: str) -> str:
    """Redact sensitive ``KEY=value`` pairs from a single log line.

    Preserves the key name for readability; only the value is replaced.

    Args:
        line: A single log line that may contain ``KEY=value`` pairs.

    Returns:
        Log line with sensitive values replaced by ``[REDACTED]``.
    """
    return _redact_log_key_value(line)


def safe_log(template: Template) -> str:  # pyright: ignore[reportInvalidTypeForm]
    """Render a t-string log message, redacting interpolated sensitive values.

    Unlike ``redact_log_line`` (which applies regex to an already-formed string),
    this function inspects each interpolated value *before* the string is
    assembled.  The variable name exposed by ``Interpolation.expr`` is checked
    against ``SENSITIVE_KEY_PATTERN``, giving structural guarantees that a
    value bound to a sensitive-looking name cannot slip through as part of a
    larger token.

    Args:
        template: A t-string template literal, e.g. ``t"key={api_key}"``.

    Returns:
        Assembled string with any interpolated value whose expression name
        matches a sensitive pattern replaced by ``[REDACTED]``.

    Example::

        api_key = "sk-abc123"
        safe_log(t"Connecting with api_key={api_key}")
        # → 'Connecting with api_key=[REDACTED]'
    """
    parts: list[str] = []
    for part in template:
        if isinstance(part, Interpolation):
            raw = str(part.value)
            expr_name: str = part.expression  # type: ignore[reportAttributeAccessIssue]
            redacted = is_sensitive_key(expr_name) or bool(
                _log_sensitive_match(f"{expr_name}={raw}")
            )
            parts.append(REDACTED_VALUE if redacted else raw)
        else:
            parts.append(part)
    return "".join(parts)


def redact_text(text: str) -> str:
    """Redact sensitive patterns from plain text, replacing entire matches.

    Handles unquoted ``KEY=value``, quoted values (single/double),
    ``Authorization: Bearer <token>`` headers, and bare sensitive key names.

    Args:
        text: Arbitrary text that may contain sensitive values.

    Returns:
        Text with all sensitive patterns replaced by ``[REDACTED]``.
    """
    if not isinstance(text, str):
        return text
    text = SENSITIVE_WORD_PATTERN.sub(REDACTED_VALUE, text)
    text = _QUOTED_DOUBLE_PATTERN.sub(REDACTED_VALUE, text)
    text = _QUOTED_SINGLE_PATTERN.sub(REDACTED_VALUE, text)
    text = _BEARER_PATTERN.sub(REDACTED_VALUE, text)
    text = SENSITIVE_KEY_NAME_PATTERN.sub(REDACTED_VALUE, text)
    return text


def _redact_value_dict(key: str, value: Any, prefix: str) -> Any:
    """Recursively redact a single value based on its accumulated key path.

    Handles dicts (recurse), lists (map over items), and sensitive strings.
    """
    full_key = f"{prefix}_{key}" if prefix else key
    if isinstance(value, dict):
        return redact_dict(value, full_key)
    if isinstance(value, list):
        return [_redact_value_dict(key, item, full_key) for item in value]
    if isinstance(value, str) and is_sensitive_key(full_key):
        return REDACTED_VALUE
    return value


def redact_dict(data: dict, env_key_prefix: str = "") -> dict:
    """Recursively redact sensitive environment variable values in a nested dict.

    Also recurses into lists: dict items are recursed, string items are
    redacted when ``is_sensitive_key(full_key)`` is true, and all other
    list items are preserved as-is.

    Args:
        data: Dictionary that may contain sensitive values.
        env_key_prefix: Accumulated key path for nested structures.

    Returns:
        New dict with sensitive values replaced by ``[REDACTED]``.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        result[key] = _redact_value_dict(key, value, env_key_prefix)
    return result
