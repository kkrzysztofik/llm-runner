# Terminal color utilities (moved from llama_manager)


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
