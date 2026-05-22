"""Manager-level smoke service tests.

Tests for resolve_smoke_targets() and run_smoke_probes() in
llama_manager.smoke — pure library functions with no I/O.
"""

from unittest.mock import MagicMock, patch

from llama_manager.config import Config, SmokeProbeConfiguration
from llama_manager.probe import SmokeProbeResult
from llama_manager.smoke import (
    SmokeTarget,
    resolve_smoke_targets,
    run_smoke_probes,
)

# ---------------------------------------------------------------------------
# resolve_smoke_targets — both mode
# ---------------------------------------------------------------------------


class TestResolveSmokeTargetsBoth:
    """Tests for resolve_smoke_targets with mode='both'."""

    def test_both_returns_two_targets(self) -> None:
        """mode 'both' should return summary-balanced and qwen35-coding targets."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "both")

        assert len(targets) == 2
        assert targets[0].slot_id == "summary-balanced"
        assert targets[0].host == "127.0.0.1"
        assert targets[0].port == 8080
        assert targets[1].slot_id == "qwen35-coding"
        assert targets[1].host == "127.0.0.1"
        assert targets[1].port == 8081

    def test_both_targets_have_backend(self) -> None:
        """Both targets should have backend set to 'llama_cpp'."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "both")

        for target in targets:
            assert target.backend == "llama_cpp"

    def test_both_targets_have_model_path(self) -> None:
        """Both targets should have model path set."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "both")

        for target in targets:
            assert isinstance(target.model, str)
            assert len(target.model) > 0

    def test_both_targets_have_correct_port(self) -> None:
        """Targets should have correct ports from Config."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "both")

        summary_target = next(t for t in targets if t.slot_id == "summary-balanced")
        qwen35_target = next(t for t in targets if t.slot_id == "qwen35-coding")

        assert summary_target.port == cfg.summary_balanced_port
        assert qwen35_target.port == cfg.qwen35_port


# ---------------------------------------------------------------------------
# resolve_smoke_targets — slot mode
# ---------------------------------------------------------------------------


class TestResolveSmokeTargetsSlot:
    """Tests for resolve_smoke_targets with mode='slot'."""

    def test_slot_summary_balanced(self) -> None:
        """mode 'slot' with summary-balanced should return one target."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "slot", "summary-balanced")

        assert len(targets) == 1
        assert targets[0].slot_id == "summary-balanced"
        assert targets[0].port == 8080

    def test_slot_qwen35(self) -> None:
        """mode 'slot' with qwen35 should return one target."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "slot", "qwen35")

        assert len(targets) == 1
        assert targets[0].slot_id == "qwen35-coding"
        assert targets[0].port == 8081

    def test_slot_summary_fast(self) -> None:
        """mode 'slot' with summary-fast should return one target."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "slot", "summary-fast")

        assert len(targets) == 1
        assert targets[0].slot_id == "summary-fast"
        assert targets[0].port == 8082

    def test_slot_unknown_returns_empty(self) -> None:
        """mode 'slot' with unknown slot_id should return empty list."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "slot", "nonexistent-slot")

        assert targets == []

    def test_slot_without_id_returns_empty(self) -> None:
        """mode 'slot' without slot_id should return empty list."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "slot", None)

        assert targets == []


# ---------------------------------------------------------------------------
# resolve_smoke_targets — invalid mode
# ---------------------------------------------------------------------------


class TestResolveSmokeTargetsInvalidMode:
    """Tests for resolve_smoke_targets with invalid mode."""

    def test_invalid_mode_returns_empty(self) -> None:
        """Invalid mode should return empty list (no error, no print)."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "bogus-mode")

        assert targets == []

    def test_empty_mode_returns_empty(self) -> None:
        """Empty string mode should return empty list."""
        cfg = Config()
        targets = resolve_smoke_targets(cfg, "")

        assert targets == []


# ---------------------------------------------------------------------------
# run_smoke_probes — basic execution
# ---------------------------------------------------------------------------


class TestRunSmokeProbes:
    """Tests for run_smoke_probes function."""

    def _make_smoke_cfg(self, **overrides: object) -> SmokeProbeConfiguration:
        """Create a SmokeProbeConfiguration for testing."""
        defaults: dict[str, object] = {
            "inter_slot_delay_s": 0,
            "listen_timeout_s": 5,
            "http_request_timeout_s": 10,
            "max_tokens": 16,
            "prompt": "test",
            "skip_models_discovery": True,
            "api_key": "",
            "model_id_override": None,
            "first_token_timeout_s": 1200,
            "total_chat_timeout_s": 1500,
        }
        defaults.update(overrides)
        return SmokeProbeConfiguration(**defaults)  # type: ignore[arg-type]

    def test_single_target(self) -> None:
        """run_smoke_probes should return one result for one target."""
        targets = [
            SmokeTarget(
                slot_id="test",
                model="/models/test.gguf",
                host="127.0.0.1",
                port=8080,
                backend="llama_cpp",
            ),
        ]
        smoke_cfg = self._make_smoke_cfg()

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            report = run_smoke_probes(targets, smoke_cfg)

        assert len(report.results) == 1
        mock_probe.assert_called_once()

    def test_multiple_targets(self) -> None:
        """run_smoke_probes should call probe_slot once per target."""
        targets = [
            SmokeTarget(
                slot_id="t1", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="t2", model="/m2.gguf", host="127.0.0.1", port=8081, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg()

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            report = run_smoke_probes(targets, smoke_cfg)

        assert len(report.results) == 2
        assert mock_probe.call_count == 2

    def test_empty_targets(self) -> None:
        """run_smoke_probes should return report with no results for empty targets."""
        smoke_cfg = self._make_smoke_cfg()

        report = run_smoke_probes([], smoke_cfg)

        assert len(report.results) == 0

    def test_probe_slot_receives_correct_args(self) -> None:
        """probe_slot should be called with target host, port, model, and config."""
        target = SmokeTarget(
            slot_id="test",
            model="/models/test.gguf",
            host="10.0.0.1",
            port=9090,
            backend="llama_cpp",
        )
        smoke_cfg = self._make_smoke_cfg(model_id_override="my-model")

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes([target], smoke_cfg)

        mock_probe.assert_called_once_with(
            host="10.0.0.1",
            port=9090,
            smoke_cfg=smoke_cfg,
            model_path="/models/test.gguf",
            model_id="my-model",
            expected_model_id=None,
        )

    def test_probe_slot_receives_model_id_override(self) -> None:
        """probe_slot should receive model_id_override from smoke_cfg."""
        target = SmokeTarget(
            slot_id="test",
            model="/models/test.gguf",
            host="127.0.0.1",
            port=8080,
            backend="llama_cpp",
        )
        smoke_cfg = self._make_smoke_cfg(model_id_override="override-model")

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes([target], smoke_cfg)

        call_kwargs = mock_probe.call_args.kwargs
        assert call_kwargs["model_id"] == "override-model"


# ---------------------------------------------------------------------------
# run_smoke_probes — inter-slot delay
# ---------------------------------------------------------------------------


class TestRunSmokeProbesInterSlotDelay:
    """Tests for inter-slot delay behavior in run_smoke_probes."""

    def _make_smoke_cfg(self, **overrides: object) -> SmokeProbeConfiguration:
        """Create a SmokeProbeConfiguration for testing."""
        defaults: dict[str, object] = {
            "inter_slot_delay_s": 0,
            "listen_timeout_s": 5,
            "http_request_timeout_s": 10,
            "max_tokens": 16,
            "prompt": "test",
            "skip_models_discovery": True,
            "api_key": "",
            "model_id_override": None,
            "first_token_timeout_s": 1200,
            "total_chat_timeout_s": 1500,
        }
        defaults.update(overrides)
        return SmokeProbeConfiguration(**defaults)  # type: ignore[arg-type]

    def test_sleep_called_between_targets(self) -> None:
        """sleep should be called between targets when delay > 0."""
        targets = [
            SmokeTarget(
                slot_id="t1", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="t2", model="/m2.gguf", host="127.0.0.1", port=8081, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg(inter_slot_delay_s=3)
        sleep_calls: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes(targets, smoke_cfg, sleep=fake_sleep)  # type: ignore[arg-type]

        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 3

    def test_no_sleep_for_single_target(self) -> None:
        """sleep should not be called for a single target."""
        targets = [
            SmokeTarget(
                slot_id="t1", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg(inter_slot_delay_s=3)
        sleep_calls: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes(targets, smoke_cfg, sleep=fake_sleep)  # type: ignore[arg-type]

        assert len(sleep_calls) == 0

    def test_no_sleep_when_delay_is_zero(self) -> None:
        """sleep should not be called when inter_slot_delay_s is 0."""
        targets = [
            SmokeTarget(
                slot_id="t1", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="t2", model="/m2.gguf", host="127.0.0.1", port=8081, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg(inter_slot_delay_s=0)
        sleep_calls: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes(targets, smoke_cfg, sleep=fake_sleep)  # type: ignore[arg-type]

        assert len(sleep_calls) == 0

    def test_three_targets_two_delays(self) -> None:
        """Three targets should have two inter-slot delays."""
        targets = [
            SmokeTarget(
                slot_id="t1", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="t2", model="/m2.gguf", host="127.0.0.1", port=8081, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="t3", model="/m3.gguf", host="127.0.0.1", port=8082, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg(inter_slot_delay_s=2)
        sleep_calls: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch("llama_manager.smoke.probe_slot") as mock_probe:
            mock_probe.return_value = SmokeProbeResult(
                slot_id="test",
                status=MagicMock(),
                phase_reached=MagicMock(),
            )
            run_smoke_probes(targets, smoke_cfg, sleep=fake_sleep)  # type: ignore[arg-type]

        assert len(sleep_calls) == 2
        assert all(call == 2 for call in sleep_calls)

    def test_probe_order_respects_target_order(self) -> None:
        """Probes should be called in target order."""
        targets = [
            SmokeTarget(
                slot_id="first", model="/m1.gguf", host="127.0.0.1", port=8080, backend="llama_cpp"
            ),
            SmokeTarget(
                slot_id="second", model="/m2.gguf", host="127.0.0.1", port=8081, backend="llama_cpp"
            ),
        ]
        smoke_cfg = self._make_smoke_cfg()
        call_order: list[str] = []

        def fake_probe(*args: object, **kwargs: object) -> SmokeProbeResult:
            host = str(kwargs.get("host", "unknown"))
            call_order.append(host)
            return SmokeProbeResult(
                slot_id=host,
                status=MagicMock(),
                phase_reached=MagicMock(),
            )

        with patch("llama_manager.smoke.probe_slot", side_effect=fake_probe):
            run_smoke_probes(targets, smoke_cfg)

        assert call_order == ["127.0.0.1", "127.0.0.1"]


# ---------------------------------------------------------------------------
# SmokeTarget dataclass
# ---------------------------------------------------------------------------


class TestSmokeTarget:
    """Tests for SmokeTarget dataclass."""

    def test_dataclass_creation(self) -> None:
        """SmokeTarget should be creatable with all required fields."""
        target = SmokeTarget(
            slot_id="test",
            model="/models/test.gguf",
            host="127.0.0.1",
            port=8080,
            backend="llama_cpp",
        )

        assert target.slot_id == "test"
        assert target.model == "/models/test.gguf"
        assert target.host == "127.0.0.1"
        assert target.port == 8080
        assert target.backend == "llama_cpp"

    def test_is_dataclass(self) -> None:
        """SmokeTarget should be a dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(SmokeTarget)
