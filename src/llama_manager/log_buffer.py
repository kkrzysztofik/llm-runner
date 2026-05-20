"""Thread-safe log buffer with autoscroll support."""

import threading
from collections import deque

from .common.security import redact_log_line


class LogBuffer:
    """Thread-safe log buffer with autoscroll support.

    Thread Safety:
    - All public methods acquire self.lock before accessing self.lines
    - add_line(), clear(), get_lines(), get_text(), get_stats(), and line_count
      are all thread-safe via the internal threading.Lock
    - The running, auto_scroll, and redact_sensitive flags are not protected by the lock
      and may be toggled concurrently; consumers should treat them as best-effort state
    - Consumers should not access self.lines directly without holding the lock
    """

    def __init__(self, max_lines: int = 50, redact_sensitive: bool = True) -> None:
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.running = True
        self.auto_scroll = True
        self.redact_sensitive = redact_sensitive

    def add_line(self, line: str) -> None:
        """Append a single log line to the buffer.

        If ``redact_sensitive`` is enabled, the line is passed through
        :func:`~.common.security.redact_log_line` before storage.

        Thread-safe: acquires ``self.lock`` before mutating ``self.lines``.

        After ``stop()`` is called, new lines are silently ignored.

        Args:
            line: The log line string to append.
        """
        with self.lock:
            if not self.running:
                return
            if self.redact_sensitive:
                line = redact_log_line(line)
            self.lines.append(line)

    def clear(self) -> None:
        """Remove all buffered lines.

        Thread-safe: acquires ``self.lock`` before clearing ``self.lines``.
        """
        with self.lock:
            self.lines.clear()

    def stop(self) -> None:
        """Signal the buffer to stop accepting new lines.

        Sets ``self.running`` to ``False`` so consumers (e.g. log-reading
        threads) can detect shutdown. Does not clear existing lines.
        """
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
