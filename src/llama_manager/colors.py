# Rich color utilities


import sys


class Color:
    """Rich-compatible color names"""

    COLORS: dict[str, str] = {
        "summary-balanced": "blue",
        "summary-fast": "yellow",
        "qwen35-coding": "green",
    }

    @staticmethod
    def get_code(server_name: str) -> str | None:
        return Color.COLORS.get(server_name)

    @staticmethod
    def is_enabled() -> bool:
        """Check if colors are enabled"""
        return sys.stdout.isatty()
