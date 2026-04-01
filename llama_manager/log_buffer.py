# Thread-safe log buffer with autoscroll support


import threading
from collections import deque

from rich.panel import Panel
from rich.text import Text


class LogBuffer:
    """Thread-safe log buffer with autoscroll support"""

    def __init__(self, max_lines: int = 50):
        self.lines: deque = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.running = True
        self.auto_scroll = True

    def add_line(self, line: str) -> None:
        """Add a log line"""
        with self.lock:
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
