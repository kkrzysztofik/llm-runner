"""Tests for llama_manager.server — validation and command building."""

import os
from unittest.mock import MagicMock

import pytest

from llama_manager.config import ErrorCode, ServerConfig, ValidationResult
from llama_manager.validation import (
    build_server_cmd,
    sort_validation_errors,
    validate_port,
    validate_ports,
    validate_threads,
)
from tests.support.helpers import make_server_config


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
        defaults: dict[str, object] = {"alias": "test", "server_bin": "/usr/bin/llama-server"}
        defaults.update(kwargs)
        return make_server_config(**defaults)

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


class TestComputeMachineFingerprint:
    """T058: Tests for machine fingerprint computation."""

    def _make_mock_subprocess_result(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> MagicMock:
        result = MagicMock()
        result.stdout = stdout
        result.stderr = stderr
        result.returncode = returncode
        return result

    def test_fingerprint_with_all_hardware_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_machine_fingerprint should include GPU, CPU, and OS info."""
        from llama_manager.validation import compute_machine_fingerprint

        mock_result = self._make_mock_subprocess_result(
            stdout="00:01.0 VGA compatible controller: Intel Corporation Arc B580\n"
            "00:02.0 VGA compatible controller: NVIDIA Corporation RTX 3090\n",
        )
        monkeypatch.setitem(os.environ, "LSPCI_OUTPUT", mock_result.stdout)

        mock_result2 = self._make_mock_subprocess_result(
            stdout="model name\t: Intel(R) Core(TM) i9-13900K\n",
        )

        mock_result3 = self._make_mock_subprocess_result(
            stdout="NAME=Ubuntu\nVERSION_ID=24.04\n",
        )

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            if cmd == ["lspci"]:
                return mock_result
            if cmd == ["cat", "/proc/cpuinfo"]:
                return mock_result2
            if cmd == ["cat", "/etc/os-release"]:
                return mock_result3
            return self._make_mock_subprocess_result()

        monkeypatch.setattr("subprocess.run", fake_run)

        fp = compute_machine_fingerprint()

        assert fp is not None
        assert len(fp) > 0
        # Should be a hex string (SHA-256 based)
        int(fp, 16)  # Should not raise

    def test_fingerprint_with_no_lspci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_machine_fingerprint should return partial fingerprint when lspci fails."""
        from llama_manager.validation import compute_machine_fingerprint

        mock_result = self._make_mock_subprocess_result(
            stdout="",
            stderr="lspci: command not found",
            returncode=127,
        )
        mock_result2 = self._make_mock_subprocess_result(
            stdout="model name\t: Intel(R) Core(TM) i9-13900K\n",
        )
        mock_result3 = self._make_mock_subprocess_result(
            stdout="NAME=Ubuntu\nVERSION_ID=24.04\n",
        )

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            if cmd == ["lspci"]:
                return mock_result
            if cmd == ["cat", "/proc/cpuinfo"]:
                return mock_result2
            if cmd == ["cat", "/etc/os-release"]:
                return mock_result3
            return self._make_mock_subprocess_result()

        monkeypatch.setattr("subprocess.run", fake_run)

        fp = compute_machine_fingerprint()

        assert fp is not None
        assert len(fp) > 0

    def test_fingerprint_all_tools_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_machine_fingerprint should return None when all tools fail."""
        from llama_manager.validation import compute_machine_fingerprint

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            return self._make_mock_subprocess_result(
                stdout="",
                stderr="command not found",
                returncode=127,
            )

        monkeypatch.setattr("subprocess.run", fake_run)

        fp = compute_machine_fingerprint()

        assert fp is None

    def test_fingerprint_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_machine_fingerprint should produce the same output for the same hardware."""
        from llama_manager.validation import compute_machine_fingerprint

        mock_result = self._make_mock_subprocess_result(
            stdout="00:01.0 VGA compatible controller: Intel Corporation Arc B580\n",
        )
        mock_result2 = self._make_mock_subprocess_result(
            stdout="model name\t: Intel(R) Core(TM) i9-13900K\n",
        )
        mock_result3 = self._make_mock_subprocess_result(
            stdout="NAME=Ubuntu\nVERSION_ID=24.04\n",
        )

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            if cmd == ["lspci"]:
                return mock_result
            if cmd == ["cat", "/proc/cpuinfo"]:
                return mock_result2
            if cmd == ["cat", "/etc/os-release"]:
                return mock_result3
            return self._make_mock_subprocess_result()

        monkeypatch.setattr("subprocess.run", fake_run)

        fp1 = compute_machine_fingerprint()
        fp2 = compute_machine_fingerprint()

        assert fp1 == fp2
        assert fp1 is not None


class TestCheckHardwareAllowlist:
    """T059: Tests for hardware allowlist check."""

    def test_allowlist_match(self) -> None:
        """check_hardware_allowlist should return 'match' when fingerprint is in allowlist."""
        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "abc123def456"
        allowlist = ["abc123def456", "other_hash"]

        result = check_hardware_allowlist(fingerprint, allowlist)

        assert result == "match"

    def test_allowlist_mismatch(self) -> None:
        """check_hardware_allowlist should return 'mismatch' when fingerprint is not in allowlist."""
        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "unknown_hash"
        allowlist = ["abc123def456", "other_hash"]

        result = check_hardware_allowlist(fingerprint, allowlist)

        assert result == "mismatch"

    def test_allowlist_invalidated(self) -> None:
        """check_hardware_allowlist should return 'invalidated' when allowlist is empty."""
        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "abc123def456"
        allowlist: list[str] = []

        result = check_hardware_allowlist(fingerprint, allowlist)

        assert result == "invalidated"

    def test_allowlist_single_entry_match(self) -> None:
        """check_hardware_allowlist should handle single-entry allowlist."""
        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "only_hash"
        allowlist = ["only_hash"]

        result = check_hardware_allowlist(fingerprint, allowlist)

        assert result == "match"

    def test_allowlist_single_entry_mismatch(self) -> None:
        """check_hardware_allowlist should handle single-entry allowlist mismatch."""
        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "different_hash"
        allowlist = ["only_hash"]

        result = check_hardware_allowlist(fingerprint, allowlist)

        assert result == "mismatch"

    def test_allowlist_env_whitespace_stripped(self) -> None:
        """check_hardware_allowlist should strip whitespace from env-var allowlist entries."""
        import os

        from llama_manager.validation import check_hardware_allowlist

        fingerprint = "fp2"
        os.environ["LLM_RUNNER_HARDWARE_ALLOWLIST"] = "fp1, fp2, fp3"
        try:
            result = check_hardware_allowlist(fingerprint, None)
        finally:
            del os.environ["LLM_RUNNER_HARDWARE_ALLOWLIST"]

        assert result == "match"

    def test_allowlist_env_empty_entries_ignored(self) -> None:
        """check_hardware_allowlist should ignore empty entries from env-var allowlist."""
        import os

        from llama_manager.validation import check_hardware_allowlist

        os.environ["LLM_RUNNER_HARDWARE_ALLOWLIST"] = "fp1,,fp2"
        try:
            result = check_hardware_allowlist("fp2", None)
        finally:
            del os.environ["LLM_RUNNER_HARDWARE_ALLOWLIST"]

        assert result == "match"


class TestAssessVramRisk:
    """T060: Tests for VRAM risk heuristic (AC-016)."""

    def test_vram_sufficient_proceed(self) -> None:
        """assess_vram_risk should return PROCEED when free VRAM >= 1.5x model size."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        result = assess_vram_risk(vram_free_gb=20, model_size_gb=10)

        assert result == VRamRecommendation.PROCEED

    def test_vram_boundary_1_5x(self) -> None:
        """assess_vram_risk should return PROCEED at exactly 1.5x ratio."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        # 15 / 10 = 1.5x exactly
        result = assess_vram_risk(vram_free_gb=15, model_size_gb=10)

        assert result == VRamRecommendation.PROCEED

    def test_vram_warn_1_45x(self) -> None:
        """assess_vram_risk should return WARN when free VRAM >= 1.411x model size (spec FR-013)."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        # 14.5 / 10 = 1.45x (between 1.411 warn threshold and 1.5 proceed threshold)
        result = assess_vram_risk(vram_free_gb=14.5, model_size_gb=10)

        assert result == VRamRecommendation.WARN

    def test_vram_boundary_warn_threshold(self) -> None:
        """assess_vram_risk should return WARN at exactly 1.2/0.85 ≈ 1.411x ratio (spec FR-013)."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        # 14.12 / 10 = 1.412x (just above 1.2/0.85 ≈ 1.41176)
        result = assess_vram_risk(vram_free_gb=14.12, model_size_gb=10)

        assert result == VRamRecommendation.WARN

    def test_vram_insufficient_confirm_required(self) -> None:
        """assess_vram_risk should return CONFIRM_REQUIRED when free VRAM < 1.1x model size."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        # 10 / 10 = 1.0x
        result = assess_vram_risk(vram_free_gb=10, model_size_gb=10)

        assert result == VRamRecommendation.CONFIRM_REQUIRED

    def test_vram_zero_free(self) -> None:
        """assess_vram_risk should return CONFIRM_REQUIRED when no free VRAM."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        result = assess_vram_risk(vram_free_gb=0, model_size_gb=10)

        assert result == VRamRecommendation.CONFIRM_REQUIRED

    def test_vram_large_buffer(self) -> None:
        """assess_vram_risk should return PROCEED with large VRAM buffer."""
        from llama_manager.config import VRamRecommendation
        from llama_manager.validation import assess_vram_risk

        # 20 / 10 = 2.0x
        result = assess_vram_risk(vram_free_gb=20, model_size_gb=10)

        assert result == VRamRecommendation.PROCEED


"""US2 FR-005 multi-error schema and ordering tests.

Test Tasks:
- T023: Add FR-005 single/multi-error schema and ordering tests verifying:
  (1) MultiValidationError has errors: list[ErrorDetail] with error_count,
  (2) ordering by slot configuration sequence (slot_id iteration order);
      when tie-breaking, use failed_check ascending within slot,
  (3) each ErrorDetail has error_code, failed_check, why_blocked, how_to_fix,
      optional docs_ref fields,
  (4) SC-002 denominator counts all errors[n] entries across runs

Contract:
- FR-005: Actionable error schema with error_code, failed_check, why_blocked, how_to_fix
- MultiValidationError: Container for multiple errors with sort_errors() method
- SC-002: Denominator-style counting across error lists
"""


from llama_manager.config import (
    ErrorDetail,
    MultiValidationError,
)


class TestFR005SingleErrorSchema:
    """FR-005: Single ErrorDetail schema assertions."""

    def test_error_detail_required_fields_present(self) -> None:
        """ErrorDetail must have error_code, failed_check, why_blocked, how_to_fix."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value in range 1-65535",
        )
        assert hasattr(error, "error_code")
        assert hasattr(error, "failed_check")
        assert hasattr(error, "why_blocked")
        assert hasattr(error, "how_to_fix")

    def test_error_detail_optional_docs_ref_field(self) -> None:
        """ErrorDetail should support optional docs_ref field."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value in range 1-65535",
            docs_ref="https://docs.example.com/port-validation",
        )
        assert hasattr(error, "docs_ref")
        assert error.docs_ref == "https://docs.example.com/port-validation"

    def test_error_detail_with_none_docs_ref(self) -> None:
        """ErrorDetail should work with docs_ref=None."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value",
            docs_ref=None,
        )
        assert error.docs_ref is None

    def test_error_detail_error_code_is_valid_enum(self) -> None:
        """ErrorDetail.error_code should be a valid ErrorCode enum value."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        assert isinstance(error.error_code, ErrorCode)
        assert error.error_code == ErrorCode.PORT_INVALID

    def test_error_detail_all_fields_populated(self) -> None:
        """ErrorDetail should work with all fields including docs_ref."""
        error = ErrorDetail(
            error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
            failed_check="vllm_launch_eligibility",
            why_blocked="vllm is not launch-eligible in PRD M1",
            how_to_fix="change backend to 'llama_cpp' for M1",
            docs_ref="https://docs.example.com/backend-eligibility",
        )
        assert error.error_code == ErrorCode.BACKEND_NOT_ELIGIBLE
        assert error.failed_check == "vllm_launch_eligibility"
        assert "vllm is not launch-eligible" in error.why_blocked
        assert "llama_cpp" in error.how_to_fix
        assert error.docs_ref == "https://docs.example.com/backend-eligibility"


class TestFR005MultiValidationErrorSchema:
    """FR-005: MultiValidationError container schema assertions."""

    def test_multi_validation_error_has_errors_field(self) -> None:
        """MultiValidationError must have errors field (list[ErrorDetail])."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error1])
        assert hasattr(multi, "errors")
        assert isinstance(multi.errors, list)
        assert len(multi.errors) == 1

    def test_multi_validation_error_error_count_property(self) -> None:
        """MultiValidationError should have error_count property."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error2 = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_validation",
            why_blocked="file missing",
            how_to_fix="add file",
        )
        multi = MultiValidationError(errors=[error1, error2])
        assert hasattr(multi, "error_count")
        assert multi.error_count == 2

    def test_multi_validation_error_empty_list(self) -> None:
        """MultiValidationError should handle empty errors list."""
        multi = MultiValidationError(errors=[])
        assert multi.errors == []
        assert multi.error_count == 0

    def test_multi_validation_error_multiple_errors(self) -> None:
        """MultiValidationError should support multiple errors."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="model_validation",
                why_blocked="file missing",
                how_to_fix="add file",
            ),
            ErrorDetail(
                error_code=ErrorCode.DUPLICATE_SLOT,
                failed_check="duplicate_detection",
                why_blocked="duplicate",
                how_to_fix="rename",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        assert len(multi.errors) == 3
        assert multi.error_count == 3


class TestFR005ErrorOrdering:
    """FR-005: Error ordering and sorting semantics."""

    def test_sort_errors_orders_by_slot_id_first(self) -> None:
        """sort_errors should order by slot_id iteration sequence first."""
        # Create errors in non-sequential slot order
        error2 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_port_validation",
            why_blocked="slot2 invalid",
            how_to_fix="fix slot2",
        )
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port_validation",
            why_blocked="slot1 invalid",
            how_to_fix="fix slot1",
        )
        error3 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot3_port_validation",
            why_blocked="slot3 invalid",
            how_to_fix="fix slot3",
        )
        multi = MultiValidationError(errors=[error2, error1, error3])
        multi.sort_errors()
        # After sorting, errors should be in slot_id order: slot1, slot2, slot3
        assert multi.errors[0].failed_check == "slot_slot1_port_validation"
        assert multi.errors[1].failed_check == "slot_slot2_port_validation"
        assert multi.errors[2].failed_check == "slot_slot3_port_validation"

    def test_sort_errors_tie_breaks_by_failed_check_ascending(self) -> None:
        """sort_errors should tie-break by failed_check ascending within same slot."""
        # Same slot, different failed_check values
        error_b = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_b_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error_a = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_a_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error_b, error_a])
        multi.sort_errors()
        # Should tie-break by failed_check ascending: a_port_validation before b_port_validation
        assert multi.errors[0].failed_check == "slot_slot1_a_port_validation"
        assert multi.errors[1].failed_check == "slot_slot1_b_port_validation"

    def test_sort_errors_mixed_slots_and_checks(self) -> None:
        """sort_errors should handle mixed slot ordering with tie-breaking."""
        # Multiple slots, some with multiple errors
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        # Expected order: slot1_a, slot1_b, slot2_a, slot2_b
        assert multi.errors[0].failed_check == "slot_slot1_a_port_validation"
        assert multi.errors[1].failed_check == "slot_slot1_b_port_validation"
        assert multi.errors[2].failed_check == "slot_slot2_a_port_validation"
        assert multi.errors[3].failed_check == "slot_slot2_b_port_validation"

    def test_sort_errors_does_not_modify_original_order_without_sort(self) -> None:
        """Errors should remain in original order until sort_errors is called."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error2 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error1, error2])
        # Before sorting - original order preserved
        assert multi.errors[0] is error1
        assert multi.errors[1] is error2
        # Sort to reorder
        multi.sort_errors()
        # After sorting - order changed
        assert multi.errors[0] is error1  # slot1 comes first
        assert multi.errors[1] is error2


class TestSC002DenominatorCounting:
    """SC-002: Denominator-style counting across error lists."""

    def test_error_count_as_denominator(self) -> None:
        """error_count should serve as denominator for percentage calculations."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation2",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation3",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        total_errors = multi.error_count
        # Verify we can use error_count as denominator
        assert total_errors == 3
        # Simulate counting successful validations vs total
        failed = 2
        passed = total_errors - failed
        # This is the SC-002 pattern: denominator is error_count
        assert passed / total_errors == 1 / 3

    def test_error_count_increases_with_more_errors(self) -> None:
        """error_count should increase as more errors are added."""
        multi = MultiValidationError(errors=[])
        assert multi.error_count == 0
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi.errors.append(error1)
        assert multi.error_count == 1
        error2 = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_validation",
            why_blocked="file missing",
            how_to_fix="add file",
        )
        multi.errors.append(error2)
        assert multi.error_count == 2

    def test_error_count_stable_after_sorting(self) -> None:
        """error_count should remain stable after sort_errors call."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        initial_count = multi.error_count
        multi.sort_errors()
        assert multi.error_count == initial_count
        assert multi.error_count == 2

    def test_error_count_counts_all_errors_n_entries(self) -> None:
        """error_count should count all entries in errors[n] across runs."""
        # Simulate multiple runs
        all_counts = []
        for i in range(5):
            errors = [
                ErrorDetail(
                    error_code=ErrorCode.PORT_INVALID,
                    failed_check=f"port_{i}_validation",
                    why_blocked=f"invalid {i}",
                    how_to_fix=f"fix {i}",
                )
            ]
            multi = MultiValidationError(errors=errors)
            all_counts.append(multi.error_count)
        # Each run should have counted correctly
        assert all(count == 1 for count in all_counts)
        assert len(all_counts) == 5

    def test_error_count_for_empty_validation(self) -> None:
        """error_count should be 0 when no validation failures."""
        multi = MultiValidationError(errors=[])
        assert multi.error_count == 0
        # No errors means denominator would cause division by zero in percentage calc
        # This is expected behavior - caller must handle zero denominator


class TestMultiValidationErrorFieldTypes:
    """FR-005: MultiValidationError field type assertions."""

    def test_errors_is_list_of_error_detail(self) -> None:
        """MultiValidationError.errors should be list[ErrorDetail]."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error])
        assert isinstance(multi.errors, list)
        assert all(isinstance(e, ErrorDetail) for e in multi.errors)

    def test_error_detail_fields_are_strings(self) -> None:
        """ErrorDetail string fields should be strings."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="this is why",
            how_to_fix="this is how",
        )
        assert isinstance(error.error_code, ErrorCode)
        assert isinstance(error.failed_check, str)
        assert isinstance(error.why_blocked, str)
        assert isinstance(error.how_to_fix, str)


class TestFR005DeterministicOrdering:
    """FR-005: Deterministic error ordering for reproducible output."""

    def test_sort_errors_produces_deterministic_result(self) -> None:
        """sort_errors should produce deterministic ordering across runs."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_z_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        # First run
        run1_order = [e.failed_check for e in multi.errors]
        # Create new instance and sort again
        multi2 = MultiValidationError(errors=errors.copy())
        multi2.sort_errors()
        run2_order = [e.failed_check for e in multi2.errors]
        # Orders should match (deterministic)
        assert run1_order == run2_order
        # Expected order: slot_a, slot_b, slot_z
        assert run1_order == [
            "slot_a_port_validation",
            "slot_b_port_validation",
            "slot_z_port_validation",
        ]

    def test_sort_errors_handles_none_slot_id_gracefully(self) -> None:
        """sort_errors should handle ErrorDetail without slot_id pattern gracefully."""
        # Error without slot_id pattern in failed_check
        error_without_slot = ErrorDetail(
            error_code=ErrorCode.CONFIG_ERROR,
            failed_check="config_validation",
            why_blocked="config invalid",
            how_to_fix="fix config",
        )
        error_with_slot = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error_without_slot, error_with_slot])
        multi.sort_errors()
        # Errors should be ordered with slot errors first, then others
        # slot1 comes before config (based on slot_id extraction logic)
        assert multi.errors[0].failed_check == "slot_slot1_port_validation"
        assert multi.errors[1].failed_check == "config_validation"
