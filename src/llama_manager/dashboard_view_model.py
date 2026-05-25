"""Dashboard view model — profile options and view state for the TUI.

Extracted from the TUI app/controller so that profile options can be
tested and cached independently of Rich, Textual, or other UI libraries.
"""

from typing import Any

from .config import Config
from .config.builder import create_tui_profile_registry


class DashboardViewModel:
    """Holds cached view state and profile options for the dashboard TUI.

    Attributes:
        _profile_options: Cached list of profile IDs, lazily populated
            from the built-in profile registry.
        _last_config_id: Hash of the last ``Config`` used to populate
            ``_profile_options``, used for cache invalidation.
    """

    def __init__(self) -> None:
        self._profile_options: list[str] | None = None
        self._last_config_id: int | None = None

    def profile_options(self, config: Config | None = None) -> list[str]:
        """Return available profile IDs for the TUI dropdown.

        Uses the TUI profile registry (built-in + custom) when available.

        Args:
            config: Optional base configuration. When omitted, defaults are used.

        Returns:
            List of profile ID strings.
        """
        cfg = config or Config()
        if self._profile_options is not None and self._last_config_id == id(cfg):
            return self._profile_options
        registry = create_tui_profile_registry(cfg)
        self._profile_options = [p.profile_id for p in registry.profiles]
        self._last_config_id = id(cfg)
        return self._profile_options

    def clear_cache(self) -> None:
        """Clear all cached view state."""
        self._profile_options = None
        self._last_config_id = None

    def get_state(self) -> dict[str, Any]:
        """Return the current view state as a serialisable dict."""
        return {
            "profile_options": self.profile_options(),
        }
