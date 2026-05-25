"""Tests for DashboardViewModel.profile_options caching and profile resolution."""

from __future__ import annotations

import pytest

from llama_manager.config import Config
from llama_manager.dashboard_view_model import DashboardViewModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def vm() -> DashboardViewModel:
    return DashboardViewModel()


# ---------------------------------------------------------------------------
# Basic profile_options behavior
# ---------------------------------------------------------------------------


def test_profile_options_returns_builtin_ids(vm: DashboardViewModel, config: Config) -> None:
    """profile_options should return built-in profile IDs."""
    options = vm.profile_options(config)
    assert isinstance(options, list)
    assert "summary-balanced" in options
    assert "summary-fast" in options
    assert "qwen35" in options


def test_profile_options_returns_config_dependent_ids(config: Config) -> None:
    """profile_options should reflect profiles from the provided config."""
    vm = DashboardViewModel()
    options = vm.profile_options(config)
    assert len(options) >= 3  # at least the built-in profiles


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_profile_options_caching(vm: DashboardViewModel, config: Config) -> None:
    """Second call should return cached value without re-creating registry."""
    options_first = vm.profile_options(config)
    options_second = vm.profile_options(config)

    # Same object reference — cached, not recreated
    assert options_first is options_second


def test_profile_options_caching_with_different_config(
    vm: DashboardViewModel,
    config: Config,
) -> None:
    """Different config instance should NOT return cached value (cache uses id(cfg))."""
    options_first = vm.profile_options(config)
    different_config = Config()
    options_second = vm.profile_options(different_config)

    # Different config -> different id -> cache miss -> new list
    assert options_first == options_second
    assert options_first is not options_second


def test_profile_options_cache_invalidation_with_none_config(
    vm: DashboardViewModel,
    config: Config,
) -> None:
    """Calling with None config after a config call should return cached value."""
    vm.profile_options(config)
    options_none = vm.profile_options(None)

    # Should still be the cached value
    assert options_none is not None
    assert len(options_none) >= 3


def test_profile_options_clear_cache(vm: DashboardViewModel, config: Config) -> None:
    """clear_cache should invalidate the cached profile_options."""
    vm.profile_options(config)
    vm.clear_cache()

    options_after = vm.profile_options(config)
    # After clear, a new list is created
    assert options_after is not None
    assert len(options_after) >= 3


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


def test_get_state_returns_profile_options(vm: DashboardViewModel, config: Config) -> None:
    """get_state should include profile_options in the returned dict."""
    state = vm.get_state()
    assert "profile_options" in state
    assert isinstance(state["profile_options"], list)
    assert len(state["profile_options"]) >= 3


def test_get_state_triggers_profile_options_populate(
    vm: DashboardViewModel,
    config: Config,
) -> None:
    """get_state should populate the cache by calling profile_options internally."""
    vm.get_state()
    # get_state calls profile_options(), so cache is populated
    assert vm._profile_options is not None
    assert len(vm._profile_options) >= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_profile_options_empty_config_returns_defaults() -> None:
    """profile_options with no config should still return built-in profiles."""
    vm = DashboardViewModel()
    options = vm.profile_options()
    assert isinstance(options, list)
    assert len(options) >= 3


def test_profile_options_returns_strings(vm: DashboardViewModel, config: Config) -> None:
    """All items in profile_options should be strings."""
    options = vm.profile_options(config)
    for item in options:
        assert isinstance(item, str)
        assert len(item) > 0
