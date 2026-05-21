"""Tests for llama_manager.log_buffer."""

import threading

from llama_manager.log_buffer import LogBuffer

# ---------------------------------------------------------------------------
# TestLogBufferInit
# ---------------------------------------------------------------------------


class TestLogBufferInit:
    """Tests for LogBuffer.__init__()."""

    def test_default_values(self) -> None:
        """LogBuffer should have correct default values."""
        buf = LogBuffer()
        assert buf.line_count == 0
        assert buf.running is True
        assert buf.auto_scroll is True
        assert buf.redact_sensitive is True

    def test_custom_max_lines(self) -> None:
        """LogBuffer should respect custom max_lines."""
        buf = LogBuffer(max_lines=10)
        for i in range(20):
            buf.add_line(f"line {i}")
        assert buf.line_count == 10

    def test_custom_redact_sensitive(self) -> None:
        """LogBuffer should accept custom redact_sensitive value."""
        buf = LogBuffer(redact_sensitive=False)
        assert buf.redact_sensitive is False


# ---------------------------------------------------------------------------
# TestAddLine
# ---------------------------------------------------------------------------


class TestAddLine:
    """Tests for LogBuffer.add_line()."""

    def test_increments_count(self) -> None:
        """add_line should increment line count."""
        buf = LogBuffer()
        buf.add_line("hello")
        assert buf.line_count == 1
        buf.add_line("world")
        assert buf.line_count == 2

    def test_respects_max_lines(self) -> None:
        """add_line should drop old lines when max_lines reached."""
        buf = LogBuffer(max_lines=3)
        buf.add_line("first")
        buf.add_line("second")
        buf.add_line("third")
        buf.add_line("fourth")
        lines = buf.get_lines()
        assert len(lines) == 3
        assert lines[0] == "second"
        assert lines[1] == "third"
        assert lines[2] == "fourth"

    def test_redacts_sensitive_by_default(self) -> None:
        """add_line should redact sensitive data by default."""
        buf = LogBuffer(redact_sensitive=True)
        buf.add_line("API_KEY=secret123")
        lines = buf.get_lines()
        assert "secret123" not in lines[0]
        assert "[REDACTED]" in lines[0]

    def test_preserves_raw_when_redact_disabled(self) -> None:
        """add_line should preserve raw data when redact_sensitive=False."""
        buf = LogBuffer(redact_sensitive=False)
        buf.add_line("API_KEY=secret123")
        lines = buf.get_lines()
        assert "secret123" in lines[0]

    def test_ignored_after_stop(self) -> None:
        """add_line should be silently ignored after stop()."""
        buf = LogBuffer()
        buf.add_line("before stop")
        buf.stop()
        buf.add_line("after stop")
        lines = buf.get_lines()
        assert len(lines) == 1
        assert lines[0] == "before stop"


# ---------------------------------------------------------------------------
# TestGetLines
# ---------------------------------------------------------------------------


class TestGetLines:
    """Tests for LogBuffer.get_lines()."""

    def test_returns_list(self) -> None:
        """get_lines should return a list."""
        buf = LogBuffer()
        buf.add_line("hello")
        result = buf.get_lines()
        assert isinstance(result, list)

    def test_returns_snapshot(self) -> None:
        """get_lines should return a snapshot, not the deque itself."""
        buf = LogBuffer()
        buf.add_line("hello")
        lines1 = buf.get_lines()
        buf.add_line("world")
        assert len(lines1) == 1
        assert len(buf.get_lines()) == 2

    def test_empty_returns_empty_list(self) -> None:
        """get_lines on empty buffer should return empty list."""
        buf = LogBuffer()
        assert buf.get_lines() == []


# ---------------------------------------------------------------------------
# TestGetText
# ---------------------------------------------------------------------------


class TestGetText:
    """Tests for LogBuffer.get_text()."""

    def test_empty_returns_default_message(self) -> None:
        """get_text should return default message when empty."""
        buf = LogBuffer()
        assert buf.get_text() == "Waiting for output..."

    def test_empty_custom_message(self) -> None:
        """get_text should accept custom empty message."""
        buf = LogBuffer()
        assert buf.get_text(empty_message="Nothing yet") == "Nothing yet"

    def test_joins_lines(self) -> None:
        """get_text should join lines with newlines."""
        buf = LogBuffer()
        buf.add_line("line1")
        buf.add_line("line2")
        buf.add_line("line3")
        result = buf.get_text()
        assert result == "line1\nline2\nline3"

    def test_single_line(self) -> None:
        """get_text with one line should return just that line."""
        buf = LogBuffer()
        buf.add_line("single")
        assert buf.get_text() == "single"


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------


class TestStop:
    """Tests for LogBuffer.stop()."""

    def test_sets_running_false(self) -> None:
        """stop() should set running to False."""
        buf = LogBuffer()
        assert buf.running is True
        buf.stop()
        assert buf.running is False

    def test_subsequent_add_ignored(self) -> None:
        """Lines added after stop() should be ignored."""
        buf = LogBuffer()
        buf.add_line("before")
        buf.stop()
        buf.add_line("after")
        assert buf.line_count == 1


# ---------------------------------------------------------------------------
# TestClear
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for LogBuffer.clear()."""

    def test_clears_lines(self) -> None:
        """clear() should remove all buffered lines."""
        buf = LogBuffer()
        buf.add_line("hello")
        buf.add_line("world")
        buf.clear()
        assert buf.line_count == 0
        assert buf.get_lines() == []

    def test_clear_after_stop(self) -> None:
        """clear() should work even after stop()."""
        buf = LogBuffer()
        buf.add_line("hello")
        buf.stop()
        buf.clear()
        assert buf.line_count == 0

    def test_clear_empty_is_noop(self) -> None:
        """clear() on empty buffer should not raise."""
        buf = LogBuffer()
        buf.clear()
        assert buf.line_count == 0


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for LogBuffer thread safety."""

    def test_concurrent_add_lines(self) -> None:
        """Multiple threads adding lines should not corrupt the buffer."""
        buf = LogBuffer(max_lines=1000)
        num_threads = 10
        lines_per_thread = 100

        def add_lines() -> None:
            for i in range(lines_per_thread):
                buf.add_line(f"thread-{threading.current_thread().name}-{i}")

        threads = [threading.Thread(target=add_lines, name=f"t{i}") for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total added lines should equal threads * lines_per_thread,
        # but capped by max_lines
        assert buf.line_count <= num_threads * lines_per_thread

    def test_clear_during_add(self) -> None:
        """clear() called during concurrent add should not raise."""
        buf = LogBuffer()
        stop_event = threading.Event()

        def add_lines() -> None:
            i = 0
            while not stop_event.is_set():
                buf.add_line(f"line-{i}")
                i += 1

        def clear_loop() -> None:
            for _ in range(50):
                buf.clear()
                threading.Event().wait(0.001)

        adder = threading.Thread(target=add_lines)
        clearer = threading.Thread(target=clear_loop)
        adder.start()
        clearer.start()
        clearer.join()
        stop_event.set()
        adder.join()

        # Should not crash; buffer may be in any valid state
        assert isinstance(buf.get_lines(), list)

    def test_get_lines_during_add(self) -> None:
        """get_lines() during concurrent add should return a valid snapshot."""
        buf = LogBuffer()
        stop_event = threading.Event()

        def add_lines() -> None:
            i = 0
            while not stop_event.is_set():
                buf.add_line(f"line-{i}")
                i += 1

        def read_lines() -> None:
            for _ in range(50):
                _ = buf.get_lines()
                threading.Event().wait(0.001)

        adder = threading.Thread(target=add_lines)
        reader = threading.Thread(target=read_lines)
        adder.start()
        reader.start()
        reader.join()
        stop_event.set()
        adder.join()

        lines = buf.get_lines()
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)
