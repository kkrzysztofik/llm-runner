# Terminal color utilities (moved from llama_manager)


class Colors:
    """Terminal color configuration.

    Module-level color state that can be controlled independently of
    Rich library availability. Color detection is disabled by default
    to allow testing without tty requirements.

    Attributes:
        enabled: Global flag to enable/disable all color output.
                 When False, all color codes are stripped.

    """

    enabled: bool = True

    COLORS: dict[str, str] = {
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

    @staticmethod
    def get_code(server_name: str) -> str | None:
        """Get ANSI color code for a server name.

        Args:
            server_name: The server alias to get color for.

        Returns:
            ANSI color code if server_name is in COLORS and Colors.enabled is True,
            otherwise None.

        """
        if not Colors.enabled:
            return None
        return Colors.COLORS.get(server_name)

    @staticmethod
    def is_enabled() -> bool:
        """Check if colors are enabled.

        Returns:
            True if Colors.enabled is True, otherwise False.

        """
        return Colors.enabled
