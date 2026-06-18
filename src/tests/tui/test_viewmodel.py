"""Tests for DashboardViewModel — derive display state from DashboardModel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from llama_cli.tui.model import DashboardModel
from llama_cli.tui.types import (
    CommandMenuState,
    DashboardSnapshot,
    DateTimeSnapshot,
    MemoryUsageSnapshot,
    ServerColumnState,
    SlotRuntimeStats,
    SystemInfoSnapshot,
)
from llama_cli.tui.viewmodel import BACKEND_LABELS, DashboardViewModel
from llama_manager import ServerConfig, SlotState
from llama_manager.build_pipeline import BuildConfig

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_server_config(
    alias: str = "test-server",
    backend: str = "sycl",
    device: str = "SYCL0",
    port: int = 8080,
    ctx_size: int = 8192,
    ubatch_size: int = 512,
    threads: int = 4,
) -> ServerConfig:
    """Create a minimal ServerConfig for testing."""
    return ServerConfig(
        alias=alias,
        model="/dev/null/model.gguf",
        backend=backend,
        device=device,
        port=port,
        ctx_size=ctx_size,
        ubatch_size=ubatch_size,
        threads=threads,
    )


def _make_viewmodel(
    configs: list[ServerConfig] | None = None,
    gpu_stats: list[Any] | None = None,
    **kwargs: Any,
) -> DashboardViewModel:
    """Build a DashboardViewModel with a mocked DashboardModel."""
    model = MagicMock(spec=DashboardModel)

    if configs is None:
        configs = [_make_server_config()]
    model.configs = configs

    if gpu_stats is None:
        gpu_stats = [MagicMock()]
    model.gpu_stats = gpu_stats

    # Default attribute stubs
    model.risk_prompt = None
    model.build_request = False
    model.build_selected_backends = None
    model.build_in_progress = False
    model.build_result = None
    model.build_error = None
    model.build_selected_backends_options = {}
    model.build_stage = None
    model.build_progress_percent = 0.0
    model.slot_states = {}
    model.server_processes = {}
    model.log_buffers = {cfg.alias: MagicMock() for cfg in configs}
    model.config = MagicMock()
    model.config.deployment.host = "127.0.0.1"
    model.unsaved_slots = set()
    model.stale_warnings = {}
    model.launch_result = None
    default_system_info = SystemInfoSnapshot(
        tasks=0,
        threads=0,
        running=0,
        load_values=None,
        uptime="0:00",
    )
    dashboard_snapshot = kwargs.pop(
        "dashboard_snapshot",
        DashboardSnapshot(
            cpu_percentages=kwargs.pop("cpu_percentages", []),
            memory_usage_rows=kwargs.pop("memory_usage_rows", []),
            system_info=kwargs.pop("system_info_snapshot", default_system_info),
            gpu_stats_by_alias=kwargs.pop("gpu_stats_by_alias", {}),
        ),
    )
    model.dashboard_snapshot.return_value = dashboard_snapshot

    # Slot stats cache
    slot_runtime_stats = kwargs.pop("slot_runtime_stats", {})
    model.slot_stats_snapshot.return_value = slot_runtime_stats

    for key, value in kwargs.items():
        setattr(model, key, value)

    return DashboardViewModel(model)


# ──────────────────────────────────────────────
# Constructor
# ──────────────────────────────────────────────


def test_viewmodel_init() -> None:
    """DashboardViewModel should store the model reference."""
    model = MagicMock(spec=DashboardModel)
    vm = DashboardViewModel(model)
    assert vm.model is model


# ──────────────────────────────────────────────
# command_menu
# ──────────────────────────────────────────────


def test_command_menu_default() -> None:
    """command_menu should return CommandMenuState with None risk_prompt and False build_request."""
    vm = _make_viewmodel()
    result = vm.command_menu()

    assert isinstance(result, CommandMenuState)
    assert result.risk_prompt is None
    assert result.build_request is False


def test_command_menu_with_risk_prompt() -> None:
    """command_menu should pass through the risk_prompt from model."""
    risk = SimpleNamespace(kind="hardware", acknowledged=False)
    vm = _make_viewmodel(risk_prompt=risk)
    result = vm.command_menu()

    assert result.risk_prompt is risk
    assert result.build_request is False


def test_command_menu_with_build_request() -> None:
    """command_menu should reflect build_request=True from model."""
    vm = _make_viewmodel(build_request=True)
    result = vm.command_menu()

    assert result.build_request is True


# ──────────────────────────────────────────────
# gpu_telemetry_lines
# ──────────────────────────────────────────────


def test_gpu_telemetry_lines_empty() -> None:
    """gpu_telemetry_lines should return empty list when no GPUs."""
    vm = _make_viewmodel(gpu_stats=[])
    result = vm.gpu_telemetry_lines()
    assert result == []


def test_gpu_telemetry_lines_single_gpu() -> None:
    """gpu_telemetry_lines should format cached GPU snapshots."""
    vm = _make_viewmodel(gpu_stats_by_alias={"gpu0": {"device": "GPU0", "gpu_util": "50%"}})
    result = vm.gpu_telemetry_lines()

    assert result == ["Device: GPU0\nGPU: 50% | Mem: N/A"]


def test_gpu_telemetry_lines_multiple_gpus() -> None:
    """gpu_telemetry_lines should collect lines from cached GPU snapshots."""
    vm = _make_viewmodel(
        gpu_stats_by_alias={
            "gpu0": {"device": "GPU0", "gpu_util": "45%"},
            "gpu1": {"device": "GPU1", "gpu_util": "72%"},
        }
    )
    result = vm.gpu_telemetry_lines()

    assert result == [
        "Device: GPU0\nGPU: 45% | Mem: N/A",
        "Device: GPU1\nGPU: 72% | Mem: N/A",
    ]


# ──────────────────────────────────────────────
# server_column_count
# ──────────────────────────────────────────────


def test_server_column_count_zero() -> None:
    """server_column_count should return at least 1 when no configs."""
    vm = _make_viewmodel(configs=[])
    assert vm.server_column_count() == 1


def test_server_column_count_one() -> None:
    """server_column_count should return 1 for a single config."""
    vm = _make_viewmodel(configs=[_make_server_config()])
    assert vm.server_column_count() == 1


def test_server_column_count_multiple() -> None:
    """server_column_count should return the number of configs."""
    configs = [_make_server_config(alias=f"server-{i}") for i in range(3)]
    vm = _make_viewmodel(configs=configs)
    assert vm.server_column_count() == 3


# ──────────────────────────────────────────────
# can_select_build_target
# ──────────────────────────────────────────────


def test_can_select_no_request() -> None:
    """can_select_build_target should be False when build_request is False."""
    vm = _make_viewmodel(build_request=False)
    assert vm.can_select_build_target() is False


def test_can_select_no_selection() -> None:
    """can_select_build_target should be True when request is set but no selection."""
    vm = _make_viewmodel(build_request=True, build_selected_backends=None)
    assert vm.can_select_build_target() is True


def test_can_select_with_selection() -> None:
    """can_select_build_target should be False when backends are already selected."""
    vm = _make_viewmodel(
        build_request=True,
        build_selected_backends=["sycl"],
    )
    assert vm.can_select_build_target() is False


# ──────────────────────────────────────────────
# Build properties (pass-through)
# ──────────────────────────────────────────────


def test_build_selected_backends_property() -> None:
    """build_selected_backends should pass through model value."""
    vm = _make_viewmodel(build_selected_backends=["cuda"])
    assert vm.build_selected_backends == ["cuda"]


def test_build_in_progress_property() -> None:
    """build_in_progress should reflect model state."""
    vm = _make_viewmodel(build_in_progress=True)
    assert vm.build_in_progress is True


def test_build_result_property() -> None:
    """build_result should pass through model value."""
    vm = _make_viewmodel(build_result="success")
    assert vm.build_result == "success"


def test_build_error_property() -> None:
    """build_error should pass through model value."""
    vm = _make_viewmodel(build_error="compile failed")
    assert vm.build_error == "compile failed"


def test_build_stage_property() -> None:
    """build_stage should pass through model value."""
    vm = _make_viewmodel(build_stage="cmake")
    assert vm.build_stage == "cmake"


def test_build_progress_percent_property() -> None:
    """build_progress_percent should pass through model value."""
    vm = _make_viewmodel(build_progress_percent=75.5)
    assert vm.build_progress_percent == 75.5


def test_build_selected_backends_options_property() -> None:
    """build_selected_backends_options should pass through model value."""
    options: dict[str, BuildConfig | None] = {"sycl": None}
    vm = _make_viewmodel(build_selected_backends_options=options)
    assert vm.build_selected_backends_options is options


# ──────────────────────────────────────────────
# cpu_usage_rows
# ──────────────────────────────────────────────


def test_cpu_usage_rows_empty() -> None:
    """cpu_usage_rows should return empty list when no CPU data."""
    vm = _make_viewmodel()
    result = vm.cpu_usage_rows()
    assert result == []


def test_cpu_usage_rows_single_core() -> None:
    """cpu_usage_rows with one core should return a single-row single-cell grid."""
    vm = _make_viewmodel(cpu_percentages=[42.5])
    result = vm.cpu_usage_rows()

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].index == 0
    assert result[0][0].percent == 42.5


def test_cpu_usage_rows_multiple_cores_narrow_width() -> None:
    """cpu_usage_rows with narrow width should produce multiple rows."""
    # 10 cores — narrow width (40) → max_cols = 40//16 = 2 → rows = 5
    vm = _make_viewmodel(cpu_percentages=[float(i) for i in range(10)])
    result = vm.cpu_usage_rows(width=40)

    # With width=40: max_cols=2, rows=5, cols=5
    # But only 10 cells total, so some rows will be shorter
    assert len(result) >= 1
    total_cells = sum(len(row) for row in result)
    assert total_cells == 10


def test_cpu_usage_rows_multiple_cores_wide_width() -> None:
    """cpu_usage_rows with wide width should produce a single row."""
    vm = _make_viewmodel(cpu_percentages=[float(i) for i in range(8)])
    result = vm.cpu_usage_rows(width=240)

    # width=240 → _content_width=240 → max_cols=15 → rows=1
    assert len(result) == 1
    assert len(result[0]) == 8


def test_cpu_usage_rows_odd_count() -> None:
    """cpu_usage_rows with odd number of cores should distribute evenly."""
    vm = _make_viewmodel(cpu_percentages=[10.0, 20.0, 30.0])
    result = vm.cpu_usage_rows(width=116)

    assert len(result) >= 1
    total_cells = sum(len(row) for row in result)
    assert total_cells == 3


# ──────────────────────────────────────────────
# memory_usage_rows
# ──────────────────────────────────────────────


def test_memory_usage_rows() -> None:
    """memory_usage_rows should read cached dashboard snapshot."""
    expected = [
        MemoryUsageSnapshot(label="Mem", percent=55.0, value_text="8/16 GB"),
        MemoryUsageSnapshot(label="Swap", percent=10.0, value_text="0.5/2 GB"),
    ]
    vm = _make_viewmodel(memory_usage_rows=expected)
    result = vm.memory_usage_rows()

    assert result == expected


# ──────────────────────────────────────────────
# Snapshots
# ──────────────────────────────────────────────


def test_system_info_snapshot() -> None:
    """system_info_snapshot should read cached dashboard snapshot."""
    expected = SystemInfoSnapshot(
        tasks=150,
        threads=300,
        running=2,
        load_values=(1.5, 2.0, 1.8),
        uptime="10:30",
    )
    vm = _make_viewmodel(system_info_snapshot=expected)
    result = vm.system_info_snapshot()

    assert result is expected


def test_dashboard_model_system_health_reads_cached_snapshots() -> None:
    """DashboardModel system-health reads should not collect live psutil data."""
    model = DashboardModel(configs=[], gpu_indices=[])
    memory = [
        MemoryUsageSnapshot(label="Mem", percent=55.0, value_text="8/16 GB"),
        MemoryUsageSnapshot(label="Swap", percent=10.0, value_text="0.5/2 GB"),
    ]
    system = SystemInfoSnapshot(
        tasks=150,
        threads=300,
        running=2,
        load_values=(1.5, 2.0, 1.8),
        uptime="10:30",
    )
    model.apply_system_health_snapshot([42.5], memory, system)

    with (
        patch("llama_manager.collect_cpu_percentages") as collect_cpu,
        patch("llama_manager.collect_memory_usage") as collect_memory,
        patch("llama_manager.collect_system_info") as collect_system,
    ):
        assert model.cpu_percentages() == [42.5]
        assert model.memory_usage_rows() == memory
        assert model.system_info_snapshot() is system

    collect_cpu.assert_not_called()
    collect_memory.assert_not_called()
    collect_system.assert_not_called()


def test_current_datetime_snapshot() -> None:
    """current_datetime_snapshot should delegate to model."""
    expected = DateTimeSnapshot(date_text="Wed 2026-05-20")
    vm = _make_viewmodel()
    model_mock = cast(MagicMock, vm.model)
    model_mock.current_datetime_snapshot.return_value = expected
    result = vm.current_datetime_snapshot()

    assert result is expected
    model_mock.current_datetime_snapshot.assert_called_once()


# ──────────────────────────────────────────────
# system_notices
# ──────────────────────────────────────────────


def test_system_notices_empty() -> None:
    """system_notices should return empty list when nothing to report."""
    vm = _make_viewmodel(launch_result=None, risk_prompt=None)
    result = vm.system_notices()
    assert result == []


def test_system_notices_blocked() -> None:
    """system_notices should report launch blocked status."""
    launch = MagicMock()
    launch.is_blocked.return_value = True
    launch.is_degraded.return_value = False
    vm = _make_viewmodel(launch_result=launch)
    result = vm.system_notices()

    assert "Launch blocked: no slots could be launched" in result


def test_system_notices_degraded() -> None:
    """system_notices should report launch degraded status."""
    launch = MagicMock()
    launch.is_blocked.return_value = False
    launch.is_degraded.return_value = True
    vm = _make_viewmodel(launch_result=launch)
    result = vm.system_notices()

    assert "Launch degraded: some slots blocked" in result


def test_system_notices_vram_risk() -> None:
    """system_notices should show VRAM risk prompt message."""
    risk = SimpleNamespace(kind="vram", acknowledged=False)
    vm = _make_viewmodel(risk_prompt=risk)
    result = vm.system_notices()

    assert "VRAM risk acknowledgement required [y/n]" in result


def test_system_notices_risk_acknowledged() -> None:
    """system_notices should show acknowledged message for hardware risk."""
    risk = SimpleNamespace(kind="hardware", acknowledged=True)
    vm = _make_viewmodel(risk_prompt=risk)
    result = vm.system_notices()

    assert "Risky operation acknowledged" in result


def test_system_notices_risk_unacknowledged() -> None:
    """system_notices should show hardware risk prompt when not acknowledged."""
    risk = SimpleNamespace(kind="hardware", acknowledged=False)
    vm = _make_viewmodel(risk_prompt=risk)
    result = vm.system_notices()

    assert "Hardware risk acknowledgement required [y/n]" in result


def test_system_notices_multiple() -> None:
    """system_notices should combine launch blocked + risk prompt."""
    launch = MagicMock()
    launch.is_blocked.return_value = True
    launch.is_degraded.return_value = False
    risk = SimpleNamespace(kind="hardware", acknowledged=False)
    vm = _make_viewmodel(launch_result=launch, risk_prompt=risk)
    result = vm.system_notices()

    assert len(result) == 2
    assert "Launch blocked: no slots could be launched" in result
    assert "Hardware risk acknowledgement required [y/n]" in result


# ──────────────────────────────────────────────
# column
# ──────────────────────────────────────────────


def test_column_out_of_range() -> None:
    """column should return None for slot_index beyond configs."""
    vm = _make_viewmodel(configs=[_make_server_config()])
    result = vm.column(99)
    assert result is None


def test_column_valid() -> None:
    """column should return a ServerColumnState with correct fields."""
    cfg = _make_server_config(alias="my-server", backend="sycl", port=9000)
    log_buf = MagicMock()
    log_buf.get_text.return_value = "server log output"
    log_buf.get_lines.return_value = ["server log output"]
    proc = MagicMock()
    proc.poll.return_value = None  # process is alive

    vm = _make_viewmodel(
        configs=[cfg],
        gpu_stats_by_alias={"my-server": {"gpu_util": "45%"}},
        log_buffers={"my-server": log_buf},
        slot_states={"my-server": "running"},
        server_processes={"my-server": proc},
    )
    result = vm.column(0)

    assert isinstance(result, ServerColumnState)
    assert result.alias == "my-server"
    assert result.status == "running"
    assert result.status_class == "server-column-status-running"
    assert result.backend_label == "SYCL"
    assert result.url == "http://127.0.0.1:9000"
    assert result.config_summary == "Device: SYCL0 | Ctx: 8192 | Threads: 4"
    assert result.profile_name == "my-server"
    assert result.status_label == "Running"
    assert result.log_lines == ("server log output",)
    assert result.runtime_stats == SlotRuntimeStats(
        tps="--",
        pp="--",
        tokens_in="0",
        tokens_out="0",
    )
    assert result.gpu_stats == {"gpu_util": "45%"}
    assert result.stale_warning is None


def test_column_shows_cached_slot_runtime_stats() -> None:
    """column should display cached slot stats when present."""
    from llama_manager.slot_stats import SlotStatsSnapshot

    cfg = _make_server_config(alias="my-server", backend="sycl", port=9000)
    log_buf = MagicMock()
    log_buf.get_text.return_value = "server log output"
    log_buf.get_lines.return_value = ["server log output"]
    proc = MagicMock()
    proc.poll.return_value = None

    cached = SlotStatsSnapshot(
        alias="my-server",
        port=9000,
        updated_at=10.0,
        tps=5.25,
        prompt_tps=99.9,
        tokens_in=123,
        tokens_out=45,
    )

    vm = _make_viewmodel(
        configs=[cfg],
        gpu_stats_by_alias={"my-server": {"gpu_util": "45%"}},
        log_buffers={"my-server": log_buf},
        slot_states={"my-server": "running"},
        server_processes={"my-server": proc},
        slot_runtime_stats={"my-server": cached},
    )
    result = vm.column(0)
    assert result is not None

    assert result.runtime_stats == SlotRuntimeStats(
        tps="5.2",
        pp="99.9",
        tokens_in="123",
        tokens_out="45",
    )


def test_column_uses_cached_gpu_snapshot_without_live_probe() -> None:
    """column should not call GPUStats collectors from the render path."""
    cfg = _make_server_config(alias="my-server")
    log_buf = MagicMock()
    log_buf.get_text.return_value = "server log output"
    gpu_mock = MagicMock()
    gpu_mock.get_stats_snapshot.side_effect = AssertionError("live GPU probe")
    gpu_mock.format_stats_text.side_effect = AssertionError("live GPU format")

    vm = _make_viewmodel(
        configs=[cfg],
        gpu_stats=[gpu_mock],
        gpu_stats_by_alias={"my-server": {"gpu_util": "45%"}},
        log_buffers={"my-server": log_buf},
    )

    result = vm.column(0)

    assert result is not None
    assert result.gpu_stats == {"gpu_util": "45%"}
    gpu_mock.get_stats_snapshot.assert_not_called()
    gpu_mock.format_stats_text.assert_not_called()


def test_column_missing_gpu() -> None:
    """column should set gpu_stats=None when GPU stats list is shorter than configs."""
    cfg = _make_server_config(alias="server-0")
    log_buf = MagicMock()
    log_buf.get_text.return_value = ""

    vm = _make_viewmodel(
        configs=[cfg],
        gpu_stats_by_alias={},
        log_buffers={"server-0": log_buf},
    )
    result = vm.column(0)

    assert result is not None
    assert result.gpu_stats is None


# ──────────────────────────────────────────────
# stale_warning
# ──────────────────────────────────────────────


def test_stale_warning_present() -> None:
    """stale_warning should return the cached warning for a config."""
    cfg = _make_server_config(alias="test")
    vm = _make_viewmodel(
        configs=[cfg],
        stale_warnings={"test": "Profile is stale"},
    )
    result = vm.stale_warning(cfg)
    assert result == "Profile is stale"


def test_stale_warning_absent() -> None:
    """stale_warning should return None when no warning is cached."""
    cfg = _make_server_config(alias="test")
    vm = _make_viewmodel(configs=[cfg], stale_warnings={})
    result = vm.stale_warning(cfg)
    assert result is None


# ──────────────────────────────────────────────
# profile_options
# ──────────────────────────────────────────────


def test_profile_options() -> None:
    """profile_options should return label/id pairs from the TUI profile registry."""
    profile1 = SimpleNamespace(profile_id="balanced", description="Balanced")
    profile2 = SimpleNamespace(profile_id="fast", description="Fast")
    registry = SimpleNamespace(profiles=[profile1, profile2])

    with patch(
        "llama_cli.tui.viewmodel.create_tui_profile_registry",
        return_value=registry,
    ) as mock_registry:
        vm = _make_viewmodel()
        result = vm.profile_options()

    mock_registry.assert_called_once()
    assert result == [
        ("balanced - Balanced", "balanced"),
        ("fast - Fast", "fast"),
    ]


# ──────────────────────────────────────────────
# _resolve_slot_status
# ──────────────────────────────────────────────


def test_resolve_slot_status_offline() -> None:
    """_resolve_slot_status should return offline when state is not in model."""
    vm = _make_viewmodel(slot_states={}, server_processes={})
    result = vm._resolve_slot_status("nonexistent")
    assert result == SlotState.OFFLINE.value


def test_resolve_slot_status_custom() -> None:
    """_resolve_slot_status should use resolve_slot_runtime_status for custom states."""
    vm = _make_viewmodel(
        slot_states={"test": "idle"},
        server_processes={},
    )
    result = vm._resolve_slot_status("test")
    # idle is not "running" so it returns unchanged
    assert result == "idle"


def test_resolve_slot_status_running_with_dead_process() -> None:
    """_resolve_slot_status should return crashed when running slot has dead process."""
    proc = MagicMock()
    proc.poll.return_value = 1  # exited
    vm = _make_viewmodel(
        slot_states={"test": "running"},
        server_processes={"test": proc},
    )
    result = vm._resolve_slot_status("test")
    assert result == SlotState.CRASHED.value


# ──────────────────────────────────────────────
# _content_width (static method)
# ──────────────────────────────────────────────


def test_content_width_none() -> None:
    """_content_width with None should return default 116."""
    assert DashboardViewModel._content_width(None) == 116


def test_content_width_zero() -> None:
    """_content_width with 0 should return default 116."""
    assert DashboardViewModel._content_width(0) == 116


def test_content_width_normal() -> None:
    """_content_width with normal value should return the value."""
    assert DashboardViewModel._content_width(80) == 80


def test_content_width_upper_cap() -> None:
    """_content_width should cap at 240."""
    assert DashboardViewModel._content_width(300) == 240


def test_content_width_lower_cap() -> None:
    """_content_width should cap at 40."""
    assert DashboardViewModel._content_width(10) == 40


# ──────────────────────────────────────────────
# BACKEND_LABELS
# ──────────────────────────────────────────────


def test_backend_labels_coverage() -> None:
    """BACKEND_LABELS should contain expected keys."""
    assert BACKEND_LABELS["sycl"] == "SYCL"
    assert BACKEND_LABELS["cuda"] == "CUDA"
    assert BACKEND_LABELS["llama_cpp"] == "CPU"


def test_column_fallback_backend_label() -> None:
    """column should fall back to CPU for unknown backend."""
    cfg = _make_server_config(alias="test", backend="unknown_backend")
    log_buf = MagicMock()
    log_buf.get_text.return_value = ""

    vm = _make_viewmodel(configs=[cfg], log_buffers={"test": log_buf})
    result = vm.column(0)

    assert result is not None
    assert result.backend_label == "CPU"  # fallback to llama_cpp label
