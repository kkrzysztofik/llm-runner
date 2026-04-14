"""Tests for llama_manager.server — validation and command building."""


from llama_manager.config import ErrorCode, ServerConfig, ValidationResult
from llama_manager.server import (
    build_server_cmd,
    sort_validation_errors,
    validate_port,
    validate_ports,
    validate_threads,
)


class TestValidatePort:
    def test_valid_port_passes(self) -> None:
        result = validate_port(8080)
        assert result is None
        result = validate_port(1)
        assert result is None
        result = validate_port(65535)
        assert result is None

    def test_zero_returns_error_detail(self) -> None:
        result = validate_port(0)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID
        assert result.failed_check == "port_validation"
        assert "port" in result.why_blocked

    def test_negative_returns_error_detail(self) -> None:
        result = validate_port(-1)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID

    def test_above_max_returns_error_detail(self) -> None:
        result = validate_port(65536)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID

    def test_custom_name_in_error_detail(self) -> None:
        result = validate_port(0, "summary-balanced port")
        assert result is not None
        assert "summary-balanced port" in result.why_blocked

    def test_error_detail_has_structured_fields(self) -> None:
        """FR-005: Validation errors should have structured fields."""
        result = validate_port(0, "port")
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID
        assert result.failed_check == "port_validation"
        assert result.why_blocked is not None
        assert result.how_to_fix is not None


class TestValidatePorts:
    def test_different_ports_pass(self) -> None:
        result = validate_ports(8080, 8081)
        assert result is None

    def test_same_ports_returns_error_detail(self) -> None:
        result = validate_ports(8080, 8080, "port1", "port2")
        assert result is not None
        assert result.error_code == ErrorCode.PORT_CONFLICT
        assert "port1" in result.why_blocked
        assert "port2" in result.why_blocked


class TestValidateThreads:
    def test_valid_threads_pass(self) -> None:
        result = validate_threads(1)
        assert result is None
        result = validate_threads(32)
        assert result is None

    def test_zero_returns_error_detail(self) -> None:
        result = validate_threads(0)
        assert result is not None
        assert result.error_code == ErrorCode.THREADS_INVALID

    def test_negative_returns_error_detail(self) -> None:
        result = validate_threads(-4)
        assert result is not None
        assert result.error_code == ErrorCode.THREADS_INVALID

    def test_custom_name_in_error_detail(self) -> None:
        result = validate_threads(0, "qwen35 threads")
        assert result is not None
        assert "qwen35 threads" in result.why_blocked

    def test_error_detail_has_structured_fields(self) -> None:
        """FR-005: Validation errors should have structured fields."""
        result = validate_threads(0, "threads")
        assert result is not None
        assert result.error_code == ErrorCode.THREADS_INVALID
        assert result.failed_check == "thread_validation"
        assert result.why_blocked is not None
        assert result.how_to_fix is not None


class TestBuildServerCmd:
    def _minimal_cfg(self, **kwargs: object) -> ServerConfig:
        defaults = {
            "model": "/models/test.gguf",
            "alias": "test",
            "device": "SYCL0",
            "port": 8080,
            "ctx_size": 4096,
            "ubatch_size": 512,
            "threads": 4,
            "server_bin": "/usr/bin/llama-server",
        }
        defaults.update(kwargs)
        return ServerConfig(**defaults)  # type: ignore[arg-type]

    def test_required_flags_present(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--model" in cmd
        assert "/models/test.gguf" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "--threads" in cmd
        assert "4" in cmd

    def test_server_bin_is_first_element(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(server_bin="/custom/llama-server"))
        assert cmd[0] == "/custom/llama-server"

    def test_device_included(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(device="CUDA0"))
        assert "--device" in cmd
        assert "CUDA0" in cmd

    def test_empty_device_excluded(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(device=""))
        assert "--device" not in cmd

    def test_jinja_flag_when_enabled(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(use_jinja=True))
        assert "--jinja" in cmd

    def test_jinja_flag_absent_by_default(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--jinja" not in cmd

    def test_tensor_split_included(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(tensor_split="0.5,0.5"))
        assert "--tensor-split" in cmd
        assert "0.5,0.5" in cmd

    def test_tensor_split_excluded_when_empty(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(tensor_split=""))
        assert "--tensor-split" not in cmd

    def test_chat_template_kwargs_included(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(chat_template_kwargs='{"enable_thinking":false}'))
        assert "--chat-template-kwargs" in cmd
        assert '{"enable_thinking":false}' in cmd

    def test_reasoning_budget_excluded_when_empty(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(reasoning_budget=""))
        assert "--reasoning-budget" not in cmd

    def test_n_gpu_layers_in_command(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(n_gpu_layers=42))
        assert "--n-gpu-layers" in cmd
        assert "42" in cmd

    def test_ctx_size_in_command(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(ctx_size=16384))
        idx = cmd.index("--ctx-size")
        assert cmd[idx + 1] == "16384"

    def test_no_webui_flag_present(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--no-webui" in cmd

    def test_cmd_is_list_of_strings(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert all(isinstance(part, str) for part in cmd)


class TestSortValidationErrors:
    """Tests for deterministic sorting of validation errors (T003)."""

    def _result(
        self,
        slot_id: str,
        passed: bool,
        failed_check: str = "",
        error_code: ErrorCode | None = None,
    ) -> ValidationResult:
        """Helper to create ValidationResult."""
        return ValidationResult(
            slot_id=slot_id,
            passed=passed,
            failed_check=failed_check,
            error_code=error_code,
        )

    def test_sort_preserves_slot_order(self) -> None:
        """sort_validation_errors should preserve input slot sequence order."""
        results = [
            self._result("slot3", False, "check_a"),
            self._result("slot1", False, "check_b"),
            self._result("slot2", True),
            self._result("slot3", False, "check_c"),
        ]
        sorted_results = sort_validation_errors(results)
        # Sort order: (slot_order, failed_check)
        # slot3:0, check_a -> (0, "check_a")
        # slot3:0, check_c -> (0, "check_c")
        # slot1:1, check_b -> (1, "check_b")
        # slot2:2, "" -> (2, "")
        # Result: slot3, slot3, slot1, slot2 (check_a < check_c, "" < "check_b")
        assert sorted_results[0].slot_id == "slot3"
        assert sorted_results[1].slot_id == "slot3"
        assert sorted_results[2].slot_id == "slot1"
        assert sorted_results[3].slot_id == "slot2"

    def test_sort_within_slot_by_failed_check(self) -> None:
        """sort_validation_errors should sort failed_check alphabetically within slot."""
        results = [
            self._result("slot1", False, "zebra_check"),
            self._result("slot1", False, "alpha_check"),
            self._result("slot1", False, "beta_check"),
        ]
        sorted_results = sort_validation_errors(results)
        checks = [r.failed_check for r in sorted_results]
        assert checks == ["alpha_check", "beta_check", "zebra_check"]

    def test_sorted_success_items_first(self) -> None:
        """Sort should place successful results (empty failed_check) first within slot."""
        results = [
            self._result("slot1", False, "zebra"),
            self._result("slot1", True),
            self._result("slot1", False, "alpha"),
        ]
        sorted_results = sort_validation_errors(results)
        # Empty failed_check sorts first (empty string < "alpha" < "zebra")
        assert sorted_results[0].failed_check == ""
        assert sorted_results[0].passed is True
        assert sorted_results[1].failed_check == "alpha"
        assert sorted_results[2].failed_check == "zebra"

    def test_empty_results(self) -> None:
        """sort_validation_errors should handle empty input."""
        assert sort_validation_errors([]) == []

    def test_single_result(self) -> None:
        """sort_validation_errors should return single result unchanged."""
        results = [self._result("only_slot", True)]
        sorted_results = sort_validation_errors(results)
        assert len(sorted_results) == 1
        assert sorted_results[0].slot_id == "only_slot"

    def test_all_same_slot(self) -> None:
        """sort_validation_errors should sort by failed_check when all same slot."""
        results = [
            self._result("slot1", False, "gamma"),
            self._result("slot1", False, "alpha"),
            self._result("slot1", True),
            self._result("slot1", False, "beta"),
        ]
        sorted_results = sort_validation_errors(results)
        checks = [r.failed_check for r in sorted_results]
        assert checks == ["", "alpha", "beta", "gamma"]

    def test_deterministic_repeated_calls(self) -> None:
        """sort_validation_errors should produce identical output on repeated calls."""
        results = [
            self._result("slot_c", False, "check_2"),
            self._result("slot_a", False, "check_1"),
            self._result("slot_b", True),
            self._result("slot_c", False, "check_1"),
        ]
        sorted1 = sort_validation_errors(results)
        sorted2 = sort_validation_errors(results)
        # Verify identical structure
        for r1, r2 in zip(sorted1, sorted2, strict=True):
            assert r1.slot_id == r2.slot_id
            assert r1.failed_check == r2.failed_check
            assert r1.passed == r2.passed
