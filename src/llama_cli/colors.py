# Terminal color utilities (moved from llama_manager)


class _AnsiCodes:
    """ANSI escape codes for terminal colors and styles."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"


class Colors:
    """Terminal color configuration.

    Module-level color state that can be controlled independently of
    Rich library availability. Colors are enabled by default.

    Attributes:
        enabled: Global flag to enable/disable all color output.
                 When False, all color codes are stripped.
        _COLORS: Mapping of server aliases to Rich color names.

    """

    enabled: bool = True

    _COLORS: dict[str, str] = {
        "summary-balanced": "blue",
        "summary-fast": "yellow",
        "qwen35-coding": "green",
    }

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """Enable or disable all color output.

        Args:
            enabled: True to enable colors, False to disable.

        """
        cls.enabled = enabled

    @classmethod
    def get_code(cls, server_name: str) -> str | None:
        """Get Rich color name for a server alias.

        Args:
            server_name: The server alias to get color for.

        Returns:
            Rich color name when colors are enabled and mapping exists, otherwise None.

        """
        if not cls.enabled:
            return None
        return cls._COLORS.get(server_name)

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if colors are enabled.

        Returns:
            True if colors are enabled, otherwise False.

        """
        return cls.enabled

    @classmethod
    def green(cls, text: str) -> str:
        """Wrap text in green color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.GREEN}{text}{_AnsiCodes.RESET}"

    @classmethod
    def bright_green(cls, text: str) -> str:
        """Wrap text in bright green color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BRIGHT_GREEN}{text}{_AnsiCodes.RESET}"

    @classmethod
    def red(cls, text: str) -> str:
        """Wrap text in red color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.RED}{text}{_AnsiCodes.RESET}"

    @classmethod
    def bright_red(cls, text: str) -> str:
        """Wrap text in bright red color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BRIGHT_RED}{text}{_AnsiCodes.RESET}"

    @classmethod
    def yellow(cls, text: str) -> str:
        """Wrap text in yellow color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.YELLOW}{text}{_AnsiCodes.RESET}"

    @classmethod
    def bright_yellow(cls, text: str) -> str:
        """Wrap text in bright yellow color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BRIGHT_YELLOW}{text}{_AnsiCodes.RESET}"

    @classmethod
    def blue(cls, text: str) -> str:
        """Wrap text in blue color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BLUE}{text}{_AnsiCodes.RESET}"

    @classmethod
    def bright_blue(cls, text: str) -> str:
        """Wrap text in bright blue color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BRIGHT_BLUE}{text}{_AnsiCodes.RESET}"

    @classmethod
    def cyan(cls, text: str) -> str:
        """Wrap text in cyan color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.CYAN}{text}{_AnsiCodes.RESET}"

    @classmethod
    def magenta(cls, text: str) -> str:
        """Wrap text in magenta color."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.MAGENTA}{text}{_AnsiCodes.RESET}"

    @classmethod
    def bold(cls, text: str) -> str:
        """Wrap text in bold style."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.BOLD}{text}{_AnsiCodes.RESET}"

    @classmethod
    def dim(cls, text: str) -> str:
        """Wrap text in dim style."""
        if not cls.enabled:
            return text
        return f"{_AnsiCodes.DIM}{text}{_AnsiCodes.RESET}"
