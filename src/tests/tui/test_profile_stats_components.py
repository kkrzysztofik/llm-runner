"""Focused tests for profile stats and compact TUI component helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from textual.css.query import NoMatches

from llama_cli.tui.components.gpu_stats import GPUStatsPanel
from llama_cli.tui.components.profile_stats_screen import ProfileStatsScreen, _format_updated_at
from llama_cli.tui.components.server_column import ServerColumnPanel
from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats
from llama_manager.config.profiles import SlotProfileSpec
from llama_manager.slot_stats import ProfileStatsAggregate


def _content(widget: Any) -> str:
    return str(widget._Static__content)  # type: ignore[attr-defined]


def test_profile_stats_rows_use_profile_labels_and_clamp_counters() -> None:
    """ProfileStatsScreen rows should format labels and display-safe counters."""
    profile = SlotProfileSpec(
        profile_id="summary",
        model="/models/summary.gguf",
        alias="summary-alias",
        device="SYCL0",
        port=8080,
        ctx_size=8192,
        ubatch_size=512,
        threads=4,
        description="Summary profile",
    )
    screen = ProfileStatsScreen(
        {
            "summary": ProfileStatsAggregate(
                "summary",
                updated_at=0.0,
                tokens_in=-10,
                tokens_out=25,
                sessions_count=-2,
            )
        },
        [(profile, "builtin")],
    )

    row = screen._stats_rows()[0]
    cells = row._pending_children  # type: ignore[attr-defined]

    assert [_content(cell) for cell in cells] == [
        "summary - Summary profile",
        "0",
        "25",
        "0",
        "--",
    ]


def test_profile_stats_close_actions_dismiss_none() -> None:
    """Escape and close button should both dismiss the modal with None."""
    screen = ProfileStatsScreen({}, [])
    dismiss = MagicMock()
    screen.dismiss = dismiss  # type: ignore[method-assign]

    screen.action_cancel()
    event = MagicMock()
    event.button.id = "close-profile-stats"
    screen.on_button_pressed(event)
    ignored = MagicMock()
    ignored.button.id = "other"
    screen.on_button_pressed(ignored)

    assert dismiss.call_count == 2
    dismiss.assert_called_with(None)


def test_format_updated_at_handles_empty_and_epoch_values() -> None:
    """Profile stats timestamps should be concise and stable."""
    timestamp = 1_700_000_000.0

    assert _format_updated_at(0.0) == "--"
    assert _format_updated_at(-1.0) == "--"
    assert _format_updated_at(timestamp) == datetime.fromtimestamp(timestamp).strftime(
        "%Y-%m-%d %H:%M"
    )


def test_format_updated_at_returns_empty_marker_on_platform_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile stats timestamps should handle out-of-range platform conversions."""

    class BrokenDateTime:
        @staticmethod
        def fromtimestamp(value: float) -> datetime:
            raise OSError("out of range")

    monkeypatch.setattr("llama_cli.tui.components.profile_stats_screen.datetime", BrokenDateTime)

    assert _format_updated_at(1.0) == "--"


def test_gpu_stats_panel_parse_and_meter_helpers() -> None:
    """GPUStatsPanel should normalize valid, invalid, and clamped percentages."""
    assert GPUStatsPanel._parse_percent("87.5%") == 87.5
    assert GPUStatsPanel._parse_percent("N/A") is None
    assert GPUStatsPanel._parse_percent(float("inf")) is None
    assert GPUStatsPanel._usage_meter(None) == "??????????"
    assert GPUStatsPanel._usage_meter(150) == "||||||||||"
    assert GPUStatsPanel._usage_meter(-5) == "          "
    assert GPUStatsPanel._usage_level_class(None) == "gpu-stats-usage-unknown"
    assert GPUStatsPanel._usage_level_class(85) == "gpu-stats-usage-high"
    assert GPUStatsPanel._usage_level_class(60) == "gpu-stats-usage-medium"
    assert GPUStatsPanel._usage_level_class(59.9) == "gpu-stats-usage-low"
    assert GPUStatsPanel._value_class("N/A") == "gpu-stats-muted-value"


def test_gpu_stats_panel_update_stats_refreshes_only_when_changed() -> None:
    """update_stats should avoid unnecessary recomposition for unchanged snapshots."""
    panel = GPUStatsPanel({"gpu_util": "10%"})
    panel.refresh = MagicMock()  # type: ignore[method-assign]

    panel.update_stats({"gpu_util": "10%"})
    panel.update_stats({"gpu_util": "20%"})
    panel.update_stats(None)

    assert panel.refresh.call_count == 2
    panel.refresh.assert_called_with(recompose=True)


def test_gpu_stats_panel_uses_devices_list_when_present() -> None:
    """Aggregated multi-GPU snapshots should render each device snapshot."""
    stats = {
        "device": "CUDA:0 GPU 0",
        "devices": [
            {"device": "CUDA:0 GPU 0", "gpu_util": "10%"},
            {"device": "CUDA:1 GPU 1", "gpu_util": "20%"},
        ],
    }

    devices = GPUStatsPanel._device_stats(stats)

    assert [device["device"] for device in devices] == ["CUDA:0 GPU 0", "CUDA:1 GPU 1"]


def test_server_column_header_includes_stale_warning_when_present() -> None:
    """ServerColumnPanel header should append a warning row only when supplied."""
    state = ServerColumnState(
        alias="summary",
        profile_name="Summary",
        status="running",
        status_label="RUNNING",
        status_class="status-running",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="ctx 8192",
        log_lines=("ready",),
        runtime_stats=SlotRuntimeStats("1.0", "2.0", "3", "4"),
        gpu_stats=None,
        stale_warning="Config changed",
    )

    header = ServerColumnPanel(state)._build_header()
    warning = header._pending_children[-1]  # type: ignore[attr-defined]

    assert _content(warning) == "Config changed"


def test_server_column_on_mount_ignores_missing_log_widget() -> None:
    """ServerColumnPanel.on_mount should ignore a missing log widget during compose churn."""
    state = ServerColumnState(
        alias="summary",
        profile_name="Summary",
        status="running",
        status_label="RUNNING",
        status_class="status-running",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="ctx 8192",
        log_lines=("ready",),
        runtime_stats=SlotRuntimeStats("1.0", "2.0", "3", "4"),
        gpu_stats=None,
        stale_warning=None,
    )
    panel = ServerColumnPanel(state)
    panel.query_one = MagicMock(side_effect=NoMatches("missing"))  # type: ignore[method-assign]

    panel.on_mount()

    panel.query_one.assert_called_once()


def test_server_column_on_mount_logs_unexpected_write_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ServerColumnPanel.on_mount should log unexpected log write failures."""
    state = ServerColumnState(
        alias="summary",
        profile_name="Summary",
        status="running",
        status_label="RUNNING",
        status_class="status-running",
        backend_label="SYCL",
        url="http://127.0.0.1:8080",
        config_summary="ctx 8192",
        log_lines=("ready",),
        runtime_stats=SlotRuntimeStats("1.0", "2.0", "3", "4"),
        gpu_stats=None,
        stale_warning=None,
    )
    log = MagicMock()
    log.write_lines.side_effect = RuntimeError("write failed")
    panel = ServerColumnPanel(state)
    panel.query_one = MagicMock(return_value=log)  # type: ignore[method-assign]

    with caplog.at_level("ERROR"):
        panel.on_mount()

    assert "failed to write initial log lines for Summary" in caplog.text
