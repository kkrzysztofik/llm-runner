"""Tests for llama_cli.colors."""

from llama_cli.colors import Colors, _AnsiCodes

# ---------------------------------------------------------------------------
# TestColorsIsEnabled
# ---------------------------------------------------------------------------


class TestColorsIsEnabled:
    """Tests for Colors.is_enabled()."""

    def setup_method(self) -> None:
        """Reset Colors state before each test."""
        Colors.set_enabled(True)

    def test_default_enabled(self) -> None:
        """Colors should be enabled by default."""
        assert Colors.is_enabled() is True

    def test_enabled_after_disable_and_reenable(self) -> None:
        """is_enabled should reflect current state."""
        Colors.set_enabled(False)
        assert Colors.is_enabled() is False
        Colors.set_enabled(True)
        assert Colors.is_enabled() is True


# ---------------------------------------------------------------------------
# TestColorsWrapping
# ---------------------------------------------------------------------------


class TestColorsWrapping:
    """Tests for color-wrapping methods when enabled."""

    def setup_method(self) -> None:
        """Ensure colors are enabled before each test."""
        Colors.set_enabled(True)

    def test_green_wraps_with_ansi(self) -> None:
        """green() should wrap text with ANSI codes."""
        result = Colors.green("hello")
        assert result == f"{_AnsiCodes.GREEN}hello{_AnsiCodes.RESET}"

    def test_red_wraps_with_ansi(self) -> None:
        """red() should wrap text with ANSI codes."""
        result = Colors.red("error")
        assert result == f"{_AnsiCodes.RED}error{_AnsiCodes.RESET}"

    def test_bold_wraps_with_ansi(self) -> None:
        """bold() should wrap text with ANSI codes."""
        result = Colors.bold("important")
        assert result == f"{_AnsiCodes.BOLD}important{_AnsiCodes.RESET}"

    def test_dim_wraps_with_ansi(self) -> None:
        """dim() should wrap text with ANSI codes."""
        result = Colors.dim("quiet")
        assert result == f"{_AnsiCodes.DIM}quiet{_AnsiCodes.RESET}"

    def test_yellow_wraps_with_ansi(self) -> None:
        """yellow() should wrap text with ANSI codes."""
        result = Colors.yellow("warning")
        assert result == f"{_AnsiCodes.YELLOW}warning{_AnsiCodes.RESET}"

    def test_blue_wraps_with_ansi(self) -> None:
        """blue() should wrap text with ANSI codes."""
        result = Colors.blue("info")
        assert result == f"{_AnsiCodes.BLUE}info{_AnsiCodes.RESET}"

    def test_cyan_wraps_with_ansi(self) -> None:
        """cyan() should wrap text with ANSI codes."""
        result = Colors.cyan("debug")
        assert result == f"{_AnsiCodes.CYAN}debug{_AnsiCodes.RESET}"

    def test_magenta_wraps_with_ansi(self) -> None:
        """magenta() should wrap text with ANSI codes."""
        result = Colors.magenta("purple")
        assert result == f"{_AnsiCodes.MAGENTA}purple{_AnsiCodes.RESET}"

    def test_bright_green_wraps_with_ansi(self) -> None:
        """bright_green() should wrap text with ANSI codes."""
        result = Colors.bright_green("bright")
        assert result == f"{_AnsiCodes.BRIGHT_GREEN}bright{_AnsiCodes.RESET}"

    def test_bright_red_wraps_with_ansi(self) -> None:
        """bright_red() should wrap text with ANSI codes."""
        result = Colors.bright_red("alert")
        assert result == f"{_AnsiCodes.BRIGHT_RED}alert{_AnsiCodes.RESET}"

    def test_bright_yellow_wraps_with_ansi(self) -> None:
        """bright_yellow() should wrap text with ANSI codes."""
        result = Colors.bright_yellow("highlight")
        assert result == f"{_AnsiCodes.BRIGHT_YELLOW}highlight{_AnsiCodes.RESET}"

    def test_bright_blue_wraps_with_ansi(self) -> None:
        """bright_blue() should wrap text with ANSI codes."""
        result = Colors.bright_blue("sky")
        assert result == f"{_AnsiCodes.BRIGHT_BLUE}sky{_AnsiCodes.RESET}"


# ---------------------------------------------------------------------------
# TestColorsDisabled
# ---------------------------------------------------------------------------


class TestColorsDisabled:
    """Tests for color methods when disabled."""

    def setup_method(self) -> None:
        """Disable colors before each test."""
        Colors.set_enabled(False)

    def test_green_returns_plain(self) -> None:
        """green() should return plain text when disabled."""
        result = Colors.green("hello")
        assert result == "hello"

    def test_red_returns_plain(self) -> None:
        """red() should return plain text when disabled."""
        result = Colors.red("error")
        assert result == "error"

    def test_bold_returns_plain(self) -> None:
        """bold() should return plain text when disabled."""
        result = Colors.bold("important")
        assert result == "important"

    def test_dim_returns_plain(self) -> None:
        """dim() should return plain text when disabled."""
        result = Colors.dim("quiet")
        assert result == "quiet"

    def test_bright_green_returns_plain(self) -> None:
        """bright_green() should return plain text when disabled."""
        result = Colors.bright_green("bright")
        assert result == "bright"


# ---------------------------------------------------------------------------
# TestGetCode
# ---------------------------------------------------------------------------


class TestGetCode:
    """Tests for Colors.get_code()."""

    def setup_method(self) -> None:
        """Ensure colors are enabled for get_code tests."""
        Colors.set_enabled(True)

    def test_summary_balanced_returns_blue(self) -> None:
        """summary-balanced should map to 'blue'."""
        assert Colors.get_code("summary-balanced") == "blue"

    def test_summary_fast_returns_yellow(self) -> None:
        """summary-fast should map to 'yellow'."""
        assert Colors.get_code("summary-fast") == "yellow"

    def test_qwen35_coding_returns_green(self) -> None:
        """qwen35-coding should map to 'green'."""
        assert Colors.get_code("qwen35-coding") == "green"

    def test_unknown_returns_none(self) -> None:
        """Unknown server names should return None."""
        assert Colors.get_code("nonexistent") is None

    def test_disabled_returns_none(self) -> None:
        """When colors disabled, get_code returns None for all names."""
        Colors.set_enabled(False)
        assert Colors.get_code("summary-balanced") is None
        assert Colors.get_code("summary-fast") is None
        assert Colors.get_code("qwen35-coding") is None
        assert Colors.get_code("nonexistent") is None
