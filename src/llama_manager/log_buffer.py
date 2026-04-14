# Thread-safe log buffer with autoscroll support


import re
import threading
from collections import deque

# Precompiled regex patterns for sensitive value redaction
_PATTERN1 = re.compile(
    r"(\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH)[A-Z0-9_]*)=(\S+)",
    re.IGNORECASE,
)
_PATTERN2 = re.compile(r"\b(KEY|TOKEN|SECRET|PASSWORD|AUTH)=(\S+)", re.IGNORECASE)


def _redact_sensitive_values(line: str) -> str:
    """Redact sensitive values from a log line.

    Matches KEY|TOKEN|SECRET|PASSWORD|AUTH patterns (case-insensitive) in
    environment variable names and redacts their values.

    Args:
        line: Log line potentially containing sensitive values

    Returns:
        Log line with sensitive values redacted

    """
    # Pattern 1: Matches env var names containing sensitive keywords followed by = and a value
    # Examples: API_KEY=secret123, TOKEN=abc, PASSWORD="hidden"
    line = _PATTERN1.sub(r"\1=[REDACTED]", line)

    # Pattern 2: Matches standalone keywords followed by = and a value (for cases like 'auth=xyz')
    line = _PATTERN2.sub(r"\1=[REDACTED]", line)

    return line


class LogBuffer:
    """Thread-safe log buffer with autoscroll support.

    Thread Safety:
    - All public methods acquire self.lock before accessing self.lines
    - add_line(), clear(), get_lines(), get_text(), get_stats(), and line_count
      are all thread-safe via the internal threading.Lock
    - The running flag is not protected by the lock and may be toggled by stop();
      consumers should treat it as best-effort state
    - Consumers should not access self.lines directly without holding the lock
    """

    def __init__(self, max_lines: int = 50, redact_sensitive: bool = True) -> None:
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.running = True
        self.auto_scroll = True
        self.redact_sensitive = redact_sensitive

    def add_line(self, line: str) -> None:
        """Add a log line with optional sensitive value redaction"""
        with self.lock:
            if self.redact_sensitive:
                line = _redact_sensitive_values(line)
            self.lines.append(line)

    def clear(self) -> None:
        """Clear all lines"""
        with self.lock:
            self.lines.clear()

    def stop(self) -> None:
        """Stop the buffer"""
        self.running = False

    def get_lines(self) -> list[str]:
        """Get a snapshot of current buffered lines."""
        with self.lock:
            return list(self.lines)

    def get_text(self, empty_message: str = "Waiting for output...") -> str:
        """Get plain-text log output suitable for UI rendering."""
        lines = self.get_lines()
        if not lines:
            return empty_message
        return "\n".join(lines)

    def get_stats(self) -> str:
        """Get buffer stats for display."""
        with self.lock:
            return f"{len(self.lines)} lines"

    @property
    def line_count(self) -> int:
        """Get current line count"""
        with self.lock:
            return len(self.lines)
