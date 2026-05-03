"""Centralized sensitive-data patterns and redaction helpers.

All modules that detect or redact credentials, tokens, or other sensitive
environment variable values should import from here rather than defining
their own patterns.  Extend ``_SENSITIVE_KEYWORDS`` to update every caller
at once.
"""

import re
from typing import Final

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
# contains a sensitive keyword.  Unquoted values only; see ``_redact_sensitive``
# in process_manager for the fuller quoted-value and bearer-token handling.
# AUTH_HEADER is listed before AUTH so the longer token wins in alternation.
SENSITIVE_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z0-9_]*(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)(?:_[0-9A-Z]+)*(?![A-Z0-9])\s*=\s*\S+",
)

# Fallback: matches bare sensitive key names even without an assignment.
# Used as a second pass after ``SENSITIVE_WORD_PATTERN`` to catch residual hits.
SENSITIVE_KEY_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z0-9_]*(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|AUTH_HEADER)(?:_[0-9A-Z]+)*(?![A-Z0-9])",
)

# Pattern for ``KEY=value`` pairs in log-stream lines.
# Preserves the key name; only the value is replaced.
# Handles both quoted values (e.g. API_KEY="ab c") and unquoted values (e.g. API_KEY=abc123).
# Quoted branches accept backslash-escaped characters so e.g. "val\"ue" is fully captured.
# Example: ``API_KEY=abc123`` → ``API_KEY=[REDACTED]``
_LOG_SENSITIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'(\b[A-Z0-9_]*(?<![A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|AUTH_HEADER)[A-Z0-9_]*)=("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\S+)',
    re.IGNORECASE,
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


def redact_log_line(line: str) -> str:
    """Redact sensitive ``KEY=value`` pairs from a single log line.

    Preserves the key name for readability; only the value is replaced.

    Args:
        line: A single log line that may contain ``KEY=value`` pairs.

    Returns:
        Log line with sensitive values replaced by ``[REDACTED]``.
    """
    return _LOG_SENSITIVE_PATTERN.sub(rf"\1={REDACTED_VALUE}", line)
