# Thread-safe log buffer with autoscroll support


import re
import threading
from collections import deque

from rich.panel import Panel
from rich.text import Text


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
    pattern1 = r"(\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH)[A-Z0-9_]*)=(\S+)"
    line = re.sub(pattern1, r"\1=[REDACTED]", line, flags=re.IGNORECASE)

    # Pattern 2: Matches standalone keywords followed by = and a value (for cases like 'auth=xyz')
    pattern2 = r"\b(KEY|TOKEN|SECRET|PASSWORD|AUTH)=(\S+)"
    line = re.sub(pattern2, r"\1=[REDACTED]", line, flags=re.IGNORECASE)

    return line


class LogBuffer:
    """Thread-safe log buffer with autoscroll support.

    Thread Safety:
    - All public methods acquire self.lock before accessing self.lines
    - add_line(), clear(), get_rich_renderable(), get_stats(), and line_count
      are all thread-safe via the internal threading.Lock
    - The running flag is not protected by the lock (read-only after initialization)
    - Consumers should not access self.lines directly without holding the lock
    """

    def __init__(self, max_lines: int = 50, redact_sensitive: bool = True):
        self.lines: deque = deque(maxlen=max_lines)
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

    def get_rich_renderable(self) -> Panel:
        """Get a Rich renderable for this buffer (Panel auto-scrolls)"""
        with self.lock:
            if not self.lines:
                text = Text("[dim]Waiting for output...[/]")
            else:
                text = Text("\n".join(self.lines))

        return Panel(text, title="Logs", border_style="dim")

    def get_stats(self) -> str:
        """Get buffer stats for display"""
        with self.lock:
            return f"[dim]{len(self.lines)} lines[/]"

    @property
    def line_count(self) -> int:
        """Get current line count"""
        with self.lock:
            return len(self.lines)
