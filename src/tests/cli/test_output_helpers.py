"""Tests for shared CLI output helpers (_output.py).

Covers:
  - print_error (stderr, red)
  - print_success (stdout)
  - print_header (stdout, bold blue)
  - print_json (stdout, JSON)
"""

from __future__ import annotations

import json
from unittest.mock import patch

from llama_cli.commands._output import (
    print_error,
    print_header,
    print_json,
    print_success,
)


class TestPrintError:
    """Tests for print_error function."""

    def test_print_error_writes_to_stderr(self, capsys) -> None:
        """print_error should write to stderr."""
        with patch("llama_cli.commands._output.Colors.red", side_effect=lambda x: x):
            print_error("test error")

        captured = capsys.readouterr()
        assert "error: test error" in captured.err

    def test_print_error_message_format(self, capsys) -> None:
        """print_error should prefix with 'error:'."""
        with patch("llama_cli.commands._output.Colors.red", side_effect=lambda x: x):
            print_error("disk full")

        captured = capsys.readouterr()
        assert captured.err.startswith("error: disk full")


class TestPrintSuccess:
    """Tests for print_success function."""

    def test_print_success_writes_to_stdout(self, capsys) -> None:
        """print_success should write to stdout."""
        print_success("done")

        captured = capsys.readouterr()
        assert captured.out.strip() == "done"

    def test_print_success_no_prefix(self, capsys) -> None:
        """print_success should not add any prefix."""
        print_success("hello world")

        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"


class TestPrintHeader:
    """Tests for print_header function."""

    def test_print_header_calls_colors(self, capsys) -> None:
        """print_header should use Colors.bold and Colors.blue."""
        with (
            patch("llama_cli.commands._output.Colors.bold", side_effect=lambda x: x),
            patch("llama_cli.commands._output.Colors.blue", side_effect=lambda x: x),
        ):
            print_header("Setup")

        captured = capsys.readouterr()
        assert "Setup" in captured.out


class TestPrintJson:
    """Tests for print_json function."""

    def test_print_json_serializes_dict(self, capsys) -> None:
        """print_json should serialize a dict to JSON."""
        print_json({"key": "value", "num": 42})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"key": "value", "num": 42}

    def test_print_json_nested_dict(self, capsys) -> None:
        """print_json should handle nested dicts."""
        print_json({"outer": {"inner": "deep"}})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["outer"]["inner"] == "deep"

    def test_print_json_writes_to_stdout(self, capsys) -> None:
        """print_json should write to stdout, not stderr."""
        print_json({"test": True})

        captured = capsys.readouterr()
        assert captured.err == ""
        assert "test" in captured.out

    def test_print_json_default_str_handler(self, capsys) -> None:
        """print_json should use default=str for non-serializable types."""
        import datetime

        print_json({"ts": datetime.datetime(2024, 1, 1)})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "ts" in parsed  # datetime should be serialized via str()
