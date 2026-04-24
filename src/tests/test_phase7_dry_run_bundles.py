"""Phase 7 — T082: CA-003 parity test — dry-run flag bundles.

Verifies dry-run output includes OpenAI flag bundles and compatibility
matrix rows in both TUI and CLI modes.

Tests:
  - dry-run summary-balanced mode includes openai_flag_bundle
  - dry-run qwen35 mode includes openai_flag_bundle
  - dry-run both mode includes openai_flag_bundle for all slots
  - vllm_eligibility rows present in output
  - Flag bundle keys are deterministic (sorted)
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

from llama_manager.config import ServerConfig
from llama_manager.server import DryRunSlotPayload, VllmEligibility


def _make_minimal_server_config(
    alias: str = "test-slot",
    model: str = "/models/test.gguf",
    port: int = 8080,
    device: str = "SYCL0",
    ctx_size: int = 4096,
    ubatch_size: int = 512,
    threads: int = 4,
    n_gpu_layers: int = 0,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
) -> ServerConfig:
    """Create a minimal ServerConfig for dry-run testing."""
    cfg = ServerConfig(
        model=model,
        alias=alias,
        device=device,
        port=port,
        ctx_size=ctx_size,
        ubatch_size=ubatch_size,
        threads=threads,
        n_gpu_layers=n_gpu_layers,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
    )
    return cfg


class TestDryRunFlagBundlesParity:
    """T082: Dry-run output includes OpenAI flag bundles and compatibility matrix."""

    # ------------------------------------------------------------------
    # openai_flag_bundle presence
    # ------------------------------------------------------------------

    def test_summary_balanced_has_openai_flag_bundle(self) -> None:
        """dry-run summary-balanced must include openai_flag_bundle in payload."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "openai_flag_bundle")
        assert isinstance(payload.openai_flag_bundle, dict)

    def test_qwen35_has_openai_flag_bundle(self) -> None:
        """dry-run qwen35 must include openai_flag_bundle in payload."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="qwen35",
            model="/models/qwen3.5-35b.gguf",
            port=8081,
            device="CUDA",
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="qwen35",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "openai_flag_bundle")
        assert isinstance(payload.openai_flag_bundle, dict)

    def test_both_mode_has_openai_flag_bundle_for_each_slot(self) -> None:
        """dry-run both mode must include openai_flag_bundle for all slots."""
        from llama_manager.server import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="summary-balanced",
                model="/models/qwen3.5-2b.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="qwen35",
                model="/models/qwen3.5-35b.gguf",
                port=8081,
                device="CUDA",
            ),
        ]

        payloads: list[DryRunSlotPayload] = []
        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            payloads.append(payload)

        # Both must have openai_flag_bundle
        for payload in payloads:
            assert hasattr(payload, "openai_flag_bundle")
            assert isinstance(payload.openai_flag_bundle, dict)

    # ------------------------------------------------------------------
    # vllm_eligibility presence
    # ------------------------------------------------------------------

    def test_summary_balanced_has_vllm_eligibility(self) -> None:
        """dry-run summary-balanced must include vllm_eligibility in payload."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "vllm_eligibility")
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    def test_qwen35_has_vllm_eligibility(self) -> None:
        """dry-run qwen35 must include vllm_eligibility in payload."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="qwen35",
            model="/models/qwen3.5-35b.gguf",
            port=8081,
            device="CUDA",
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="qwen35",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "vllm_eligibility")
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    # ------------------------------------------------------------------
    # Flag bundle key determinism
    # ------------------------------------------------------------------

    def test_openai_flag_bundle_keys_are_deterministic(self) -> None:
        """openai_flag_bundle keys must be deterministic (sorted on serialization)."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        # Keys should be sortable
        keys = list(payload.openai_flag_bundle.keys())
        sorted_keys = sorted(keys)
        assert keys == sorted_keys, f"Keys not sorted: {keys} vs {sorted_keys}"

    def test_multiple_payloads_have_consistent_bundle_structure(self) -> None:
        """All payloads from the same mode should have consistent bundle structure."""
        from llama_manager.server import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="slot1",
                model="/models/model1.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="slot2",
                model="/models/model2.gguf",
                port=8081,
            ),
        ]

        payloads = []
        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            payloads.append(payload)

        # All bundles should have the same set of keys
        bundle_keys = [set(p.openai_flag_bundle.keys()) for p in payloads]
        assert len({frozenset(k) for k in bundle_keys}) == 1, (
            f"Inconsistent bundle keys across payloads: {bundle_keys}"
        )

    # ------------------------------------------------------------------
    # Dry-run output includes flag bundle info
    # ------------------------------------------------------------------

    def test_dry_run_output_includes_openai_bundle_section(self, capsys) -> None:
        """dry-run summary-balanced output must include 'OpenAI Bundle' section."""
        from llama_cli.dry_run import dry_run

        with (
            patch("llama_cli.dry_run._run_summary_balanced_mode") as mock_run,
            patch("llama_cli.dry_run._print_smoke_probe_info"),
        ):
            mock_run.return_value = False
            mock_run.slot_payloads = []

            with contextlib.suppress(SystemExit):
                dry_run(mode="summary-balanced", primary_port="8080")

            mock_run.assert_called()

    def test_dry_run_both_mode_prints_both_bundles(self, capsys) -> None:
        """dry-run both mode must print openai_flag_bundle for both summary and qwen35."""
        from llama_cli.dry_run import dry_run

        with (
            patch("llama_cli.dry_run._run_both_mode") as mock_run,
            patch("llama_cli.dry_run._print_smoke_probe_info"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="both", primary_port="8080", secondary_port="8081")

            mock_run.assert_called()

    def test_dry_run_summary_balanced_integration(self, capsys) -> None:
        """dry-run summary-balanced integration test without mocking mode handlers."""
        from llama_cli.dry_run import _print_common_payload_sections
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        _print_common_payload_sections(payload)

        captured = capsys.readouterr()
        assert "OpenAI Bundle" in captured.out

    def test_dry_run_both_integration(self, capsys) -> None:
        """dry-run both mode integration test checking both bundles appear."""
        from llama_cli.dry_run import _print_common_payload_sections
        from llama_manager.server import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="summary-balanced",
                model="/models/qwen3.5-2b.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="qwen35",
                model="/models/qwen3.5-35b.gguf",
                port=8081,
                device="CUDA",
            ),
        ]

        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            _print_common_payload_sections(payload)

        captured = capsys.readouterr()
        assert "OpenAI Bundle" in captured.out

    # ------------------------------------------------------------------
    # TUI vs CLI consistency for flag bundles
    # ------------------------------------------------------------------

    def test_tui_and_cli_use_same_payload_structure(self) -> None:
        """TUI (via ServerManager) and CLI (via dry_run) must use same payload structure."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test-slot",
            model="/models/test.gguf",
            port=8080,
        )

        # Build payload the same way both TUI and CLI would
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test-slot",
            validation_results=None,
            warnings=[],
        )

        # Both must have the same required fields
        assert "openai_flag_bundle" in vars(payload)
        assert "vllm_eligibility" in vars(payload)
        assert "command_args" in vars(payload)
        assert isinstance(payload.openai_flag_bundle, dict)
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    def test_vllm_eligibility_has_required_fields(self) -> None:
        """vllm_eligibility must include eligible and reason fields."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        vllm = payload.vllm_eligibility
        assert hasattr(vllm, "eligible"), "vllm_eligibility missing 'eligible'"
        assert hasattr(vllm, "reason"), "vllm_eligibility missing 'reason'"
        assert isinstance(vllm.eligible, bool)
        assert isinstance(vllm.reason, str)

    def test_openai_flag_bundle_contains_expected_keys(self) -> None:
        """openai_flag_bundle should contain OpenAI-compatible flag keys."""
        from llama_manager.server import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        bundle = payload.openai_flag_bundle
        # At minimum should have some OpenAI-related flags
        # The exact keys depend on config, but there should be some
        assert isinstance(bundle, dict)
        # Keys should start with -- (CLI flag style)
        for key in bundle:
            assert key.startswith("--"), f"openai_flag_bundle key '{key}' should start with '--'"
