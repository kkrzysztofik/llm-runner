"""Tests for llama_cli.ui_output — stream routing, TTY detection, ANSI styling."""

from unittest.mock import patch

import pytest

from llama_cli.ui_output import (
    emit_error,
    emit_heading,
    emit_info,
    emit_plain,
    emit_success,
    emit_warn,
)

# ---------------------------------------------------------------------------
# Stream routing tests
# ---------------------------------------------------------------------------


class TestStreamRouting:
    """Verify each emit_* function writes to the correct stream."""

    def test_emit_info_goes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_info should write to stdout with 'info:' prefix."""
        emit_info("hello")
        captured = capsys.readouterr()
        assert "info:" in captured.out
        assert "hello" in captured.out

    def test_emit_success_goes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_success should write to stdout with 'ok:' prefix."""
        emit_success("done")
        captured = capsys.readouterr()
        assert "ok:" in captured.out
        assert "done" in captured.out

    def test_emit_warn_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_warn should write to stderr with 'warn:' prefix."""
        emit_warn("caution")
        captured = capsys.readouterr()
        assert "warn:" in captured.err
        assert "caution" in captured.err

    def test_emit_error_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_error should write to stderr with 'error:' prefix."""
        emit_error("fail")
        captured = capsys.readouterr()
        assert "error:" in captured.err
        assert "fail" in captured.err

    def test_emit_plain_default_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_plain should write to stdout by default."""
        emit_plain("raw text")
        captured = capsys.readouterr()
        assert "raw text" in captured.out


# ---------------------------------------------------------------------------
# emit_plain err flag
# ---------------------------------------------------------------------------


class TestEmitPlainErr:
    """Verify emit_plain respects the err=True flag."""

    def test_emit_plain_err_goes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_plain(err=True) should write to stderr."""
        emit_plain("error path", err=True)
        captured = capsys.readouterr()
        assert "error path" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# TTY detection — non-TTY
# ---------------------------------------------------------------------------


class TestNonTTY:
    """Verify output is plain (no ANSI escape codes) when stdout is not a TTY."""

    @patch("llama_cli.ui_output.sys.stdout.isatty", return_value=False)
    def test_no_ansi_codes_when_not_tty(
        self, _mock: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When isatty() is False, output must not contain \\033 escape sequences."""
        emit_info("hello")
        emit_success("done")
        emit_warn("caution")
        emit_error("fail")
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "\033" not in combined


# ---------------------------------------------------------------------------
# emit_heading
# ---------------------------------------------------------------------------


class TestEmitHeading:
    """Verify heading prefix and styling."""

    def test_emit_heading_default_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_heading without level should use '# ' prefix."""
        emit_heading("My Section")
        captured = capsys.readouterr()
        output = captured.out
        # When not a TTY, prefix is plain "# "
        assert output.startswith("# My Section") or output.startswith("#")

    def test_emit_heading_level_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_heading(level=2) should use '## ' prefix."""
        emit_heading("Sub Section", level=2)
        captured = capsys.readouterr()
        output = captured.out
        assert output.startswith("## Sub Section") or output.startswith("##")


# ---------------------------------------------------------------------------
# Message content preservation
# ---------------------------------------------------------------------------


class TestMessagePreservation:
    """Verify the original message text is preserved in output."""

    def test_message_content_preserved(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The full message text must appear in output (even with prefix added)."""
        emit_info("unique message xyz")
        captured = capsys.readouterr()
        assert "unique message xyz" in captured.out

        emit_warn("unique message xyz")
        captured = capsys.readouterr()
        assert "unique message xyz" in captured.err

        emit_error("unique message xyz")
        captured = capsys.readouterr()
        assert "unique message xyz" in captured.err


# ---------------------------------------------------------------------------
# TTY detection — TTY
# ---------------------------------------------------------------------------


class TestTTY:
    """Verify ANSI escape codes are present when stdout is a TTY."""

    @patch("llama_cli.ui_output.sys.stdout.isatty", return_value=True)
    def test_ansi_codes_when_tty(self, _mock: object, capsys: pytest.CaptureFixture[str]) -> None:
        """When isatty() is True, color output must contain \\033 escape codes."""
        emit_info("hello")
        emit_success("done")
        captured = capsys.readouterr()
        assert "\033" in captured.out
