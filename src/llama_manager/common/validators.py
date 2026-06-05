"""Shared validators — single source of truth for cross-cutting validation rules."""

PORT_MIN = 1024
PORT_MAX = 65535


def is_valid_port(port: int) -> bool:
    """Return ``True`` if *port* is a valid unprivileged TCP port number."""
    return isinstance(port, int) and PORT_MIN <= port <= PORT_MAX


def validate_port_range(port: int) -> str | None:
    """Return error message if port is invalid, None if valid."""
    if not isinstance(port, int) or port < PORT_MIN or port > PORT_MAX:
        return f"port must be between {PORT_MIN} and {PORT_MAX}, got: {port}"
    return None
