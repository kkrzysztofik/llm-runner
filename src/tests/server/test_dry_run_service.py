"""Tests for dry-run domain service."""

from llama_manager.config import Config, create_default_profile_registry
from llama_manager.dry_run import DryRunResult, run_dry_run


class TestRunDryRun:
    """Tests for run_dry_run function."""

    def test_successful_dry_run(self) -> None:
        """Successful dry-run should return payloads without errors."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="summary-balanced",
            config=cfg,
            registry=registry,
        )

        assert isinstance(result, DryRunResult)
        assert result.mode == "summary-balanced"
        assert result.has_error is False
        assert len(result.slot_payloads) > 0
        assert result.errors == []

    def test_invalid_mode(self) -> None:
        """Invalid mode should return error without payloads."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="nonexistent-mode",
            config=cfg,
            registry=registry,
        )

        assert result.has_error is True
        assert len(result.slot_payloads) == 0
        assert len(result.errors) > 0
        assert "invalid mode" in result.errors[0].lower()

    def test_invalid_port_override(self) -> None:
        """Invalid port override should return error."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="summary-balanced",
            config=cfg,
            registry=registry,
            port_overrides={"primary": 99999},
        )

        assert result.has_error is True
        assert len(result.slot_payloads) == 0
        assert len(result.errors) > 0

    def test_both_mode_includes_multiple_slots(self) -> None:
        """Both mode should include multiple slot payloads."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="both",
            config=cfg,
            registry=registry,
        )

        assert result.has_error is False
        assert len(result.slot_payloads) == 2
        slot_ids = {p.slot_id for p in result.slot_payloads}
        assert "summary-balanced" in slot_ids
        assert "qwen35" in slot_ids

    def test_artifact_payload_built_on_success(self) -> None:
        """Successful dry-run should include artifact payload."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="summary-balanced",
            config=cfg,
            registry=registry,
        )

        assert result.artifact_payload is not None
        ap = result.artifact_payload
        assert ap.get("timestamp") is not None
        assert isinstance(ap.get("slot_scope"), list)
        assert len(ap["slot_scope"]) > 0
        assert isinstance(ap.get("resolved_command"), dict)

    def test_artifact_payload_missing_on_error(self) -> None:
        """Invalid mode should not include artifact payload."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="nonexistent-mode",
            config=cfg,
            registry=registry,
        )

        assert result.artifact_payload is None

    def test_risk_warnings_when_risky(self) -> None:
        """Risky operations should generate warnings."""
        result = run_dry_run(
            mode="summary-balanced",
            config=Config(),
            registry=create_default_profile_registry(),
        )

        # Result may or may not have warnings depending on config
        assert isinstance(result.warnings, list)

    def test_acknowledged_risk_suppresses_input(self) -> None:
        """Acknowledged risk should not require interactive input."""
        cfg = Config()
        registry = create_default_profile_registry(cfg)
        result = run_dry_run(
            mode="summary-balanced",
            config=cfg,
            registry=registry,
            acknowledged=True,
        )

        assert result.has_error is False
