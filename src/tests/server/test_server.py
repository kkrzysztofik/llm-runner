"""Tests for llama_manager.server — validation and command building."""

from pathlib import Path

import pytest

from llama_manager.config import ErrorCode, ErrorDetail, ServerConfig
from llama_manager.validation import (
    build_server_cmd,
    validate_port,
    validate_ports,
)
from tests.support.helpers import make_server_config


class TestValidatePort:
    def test_valid_port_passes(self) -> None:
        result = validate_port(8080)
        assert result is None
        result = validate_port(1024)
        assert result is None
        result = validate_port(65535)
        assert result is None

    def test_privileged_port_returns_error(self) -> None:
        result = validate_port(1)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID
        result = validate_port(80)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID
        result = validate_port(1023)
        assert result is not None
        assert result.error_code == ErrorCode.PORT_INVALID

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

    def test_cuda_device_uses_nvidia_default_when_server_bin_blank(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        llama_root = tmp_path / "llama.cpp"
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(llama_root))

        cmd = build_server_cmd(self._minimal_cfg(device="CUDA:0", server_bin=""))

        assert cmd[0] == str(llama_root / "build_cuda" / "bin" / "llama-server")

    def test_non_cuda_device_uses_intel_default_when_server_bin_blank(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        llama_root = tmp_path / "llama.cpp"
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(llama_root))

        cmd = build_server_cmd(self._minimal_cfg(device="SYCL0", server_bin=""))

        assert cmd[0] == str(llama_root / "build" / "bin" / "llama-server")

    def test_device_included(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(device="CUDA0"))
        assert "--device" in cmd
        assert cmd[cmd.index("--device") + 1] == "CUDA0"

    def test_cuda_colon_device_normalized_for_server_arg(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(device="CUDA:0,1"))
        assert cmd[cmd.index("--device") + 1] == "CUDA0,CUDA1"

    def test_sycl_device_kept_for_server_arg(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(device="SYCL0"))
        assert cmd[cmd.index("--device") + 1] == "SYCL0"

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

    def test_main_gpu_excluded_when_zero(self) -> None:
        """--main-gpu should not appear when main_gpu is 0 (default)."""
        cmd = build_server_cmd(self._minimal_cfg(main_gpu=0))
        assert "--main-gpu" not in cmd

    def test_main_gpu_included_when_nonzero(self) -> None:
        """--main-gpu should appear when main_gpu is non-zero."""
        cmd = build_server_cmd(self._minimal_cfg(main_gpu=1))
        assert "--main-gpu" in cmd
        assert "1" in cmd

    def test_ctx_size_in_command(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(ctx_size=16384))
        idx = cmd.index("--ctx-size")
        assert cmd[idx + 1] == "16384"

    def test_no_ui_flag_present_by_default(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--no-ui" in cmd

    def test_ui_enabled_omits_no_ui_flag(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(ui=True))
        assert "--no-ui" not in cmd

    def test_metrics_flag_present_for_runtime_stats(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--metrics" in cmd

    def test_cmd_is_list_of_strings(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert all(isinstance(part, str) for part in cmd)

    def test_launch_throughput_flags(self) -> None:
        cmd = build_server_cmd(
            self._minimal_cfg(
                batch_size=1024,
                poll_ms=0,
                n_predict=8192,
                parallel=4,
            )
        )
        assert cmd[cmd.index("--batch-size") + 1] == "1024"
        assert cmd[cmd.index("--poll") + 1] == "0"
        assert cmd[cmd.index("--n-predict") + 1] == "8192"
        assert cmd[cmd.index("--parallel") + 1] == "4"

    def test_threads_batch_omitted_when_zero(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(threads_batch=0))
        assert "--threads-batch" not in cmd

    def test_threads_batch_included_when_positive(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(threads_batch=8))
        assert cmd[cmd.index("--threads-batch") + 1] == "8"

    def test_mmproj_included_when_set(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(mmproj="/models/mmproj.gguf"))
        assert cmd[cmd.index("--mmproj") + 1] == "/models/mmproj.gguf"

    def test_ngram_spec_flags(self) -> None:
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="ngram-mod",
                spec_ngram_size_n=12,
                draft_min=8,
                draft_max=32,
            )
        )
        assert "--spec-type" in cmd
        assert "ngram-mod" in cmd
        assert cmd[cmd.index("--spec-ngram-mod-n-match") + 1] == "12"
        assert cmd[cmd.index("--spec-ngram-mod-n-min") + 1] == "8"
        assert cmd[cmd.index("--spec-ngram-mod-n-max") + 1] == "32"
        assert "--spec-ngram-size-n" not in cmd
        assert "--draft-min" not in cmd
        assert "--draft-max" not in cmd

    @pytest.mark.parametrize(
        "flag",
        ["--spec-ngram-mod-n-match", "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max"],
    )
    def test_unset_ngram_fields_are_omitted(self, flag: str) -> None:
        cmd = build_server_cmd(self._minimal_cfg(spec_type="ngram-mod"))
        assert flag not in cmd

    def test_normalized_spec_type_emits_single_argv_token(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(spec_type="draft-mtp, ngram-mod"))
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp,ngram-mod"

    def test_unset_spec_draft_n_max_is_omitted(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(spec_type="draft-mtp"))
        assert "--spec-draft-n-max" not in cmd

    def test_combined_spec_types_emit_both_flag_groups(self) -> None:
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-mtp,ngram-mod",
                spec_draft_n_max=8,
                spec_ngram_size_n=12,
                draft_min=8,
                draft_max=32,
            )
        )
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp,ngram-mod"
        assert cmd.count("--spec-type") == 1
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "8"
        assert cmd[cmd.index("--spec-ngram-mod-n-match") + 1] == "12"

    def test_draft_mtp_spec_flags(self) -> None:
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-mtp",
                spec_draft_n_max=16,
                spec_draft_p_min=0.5,
                spec_draft_cache_type_k="q8_0",
                spec_draft_cache_type_v="f16",
                spec_draft_device="CUDA:1",
            )
        )
        assert "draft-mtp" in cmd
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "16"
        assert cmd[cmd.index("--spec-draft-p-min") + 1] == "0.5"
        assert cmd[cmd.index("--spec-draft-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--spec-draft-type-v") + 1] == "f16"
        assert cmd[cmd.index("--spec-draft-device") + 1] == "CUDA:1"

    def test_dflash_local_draft_flags(self) -> None:
        """DFlash with local draft model emits correct flags."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-dflash",
                spec_draft_model="/models/draft.gguf",
                spec_draft_ngl="all",
                spec_dflash_cross_ctx=512,
            )
        )
        assert "--spec-type" in cmd
        assert "draft-dflash" in cmd
        assert "--spec-draft-model" in cmd
        assert "/models/draft.gguf" in cmd
        assert "--spec-draft-ngl" in cmd
        assert "all" in cmd
        assert "--spec-dflash-cross-ctx" in cmd
        assert "512" in cmd

    def test_dflash_hf_draft_flags(self) -> None:
        """DFlash with HF draft emits --spec-draft-hf."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-dflash",
                spec_draft_hf="Anbeeld/Qwen3.6-27B-DFlash-GGUF:IQ4_XS",
            )
        )
        assert "--spec-draft-hf" in cmd
        assert "Anbeeld/Qwen3.6-27B-DFlash-GGUF:IQ4_XS" in cmd

    def test_dflash_int_draft_ngl(self) -> None:
        """DFlash with integer spec_draft_ngl emits numeric value."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-dflash",
                spec_draft_model="/models/draft.gguf",
                spec_draft_ngl=42,
            )
        )
        assert cmd[cmd.index("--spec-draft-ngl") + 1] == "42"

    def test_dflash_omits_zero_cross_ctx(self) -> None:
        """DFlash omits --spec-dflash-cross-ctx when value is 0."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-dflash",
                spec_draft_model="/models/draft.gguf",
                spec_dflash_cross_ctx=0,
            )
        )
        assert "--spec-dflash-cross-ctx" not in cmd

    def test_dflash_reasoning_flags(self) -> None:
        """DFlash with reasoning mode/format emits reasoning flags."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                spec_type="draft-dflash",
                spec_draft_model="/models/draft.gguf",
                reasoning_mode="on",
                reasoning_format="deepseek",
            )
        )
        assert "--reasoning" in cmd
        assert "on" in cmd
        assert "--reasoning-format" in cmd
        assert "deepseek" in cmd

    def test_smaller_model_kv_unified_flag(self) -> None:
        """--kv-unified flag is emitted when kv_unified is True."""
        cmd = build_server_cmd(self._minimal_cfg(kv_unified=True))
        assert "--kv-unified" in cmd

    def test_smaller_model_no_kv_unified_by_default(self) -> None:
        """--kv-unified flag is absent by default."""
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--kv-unified" not in cmd

    def test_load_mode_auto_omits_flag(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(load_mode="auto"))
        assert "--load-mode" not in cmd
        assert "--mmap" not in cmd
        assert "--mlock" not in cmd

    def test_load_mode_mmap_emits(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(load_mode="mmap"))
        i = cmd.index("--load-mode")
        assert cmd[i + 1] == "mmap"

    def test_reasoning_preserve_on(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="on"))
        assert "--reasoning-preserve" in cmd

    def test_reasoning_preserve_off(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="off"))
        assert "--no-reasoning-preserve" in cmd

    def test_reasoning_preserve_auto_omits(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(reasoning_preserve="auto"))
        assert "--reasoning-preserve" not in cmd
        assert "--no-reasoning-preserve" not in cmd

    def test_fit_and_sampling_emit_when_set(self) -> None:
        cmd = build_server_cmd(
            self._minimal_cfg(
                fit="off",
                ctx_checkpoints=64,
                temperature=1.0,
                top_k=20,
                top_p=0.95,
                min_p=0.0,
                presence_penalty=0.0,
                repeat_penalty=1.0,
                reasoning_budget_message="stop thinking",
            )
        )
        assert cmd[cmd.index("--fit") + 1] == "off"
        assert cmd[cmd.index("--ctx-checkpoints") + 1] == "64"
        assert cmd[cmd.index("--temp") + 1] == "1.0"
        assert cmd[cmd.index("--top-k") + 1] == "20"
        assert cmd[cmd.index("--top-p") + 1] == "0.95"
        assert cmd[cmd.index("--min-p") + 1] == "0.0"
        assert cmd[cmd.index("--presence-penalty") + 1] == "0.0"
        assert cmd[cmd.index("--repeat-penalty") + 1] == "1.0"
        assert cmd[cmd.index("--reasoning-budget-message") + 1] == "stop thinking"

    def test_split_mode_defaults_to_layer(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg())
        assert cmd[cmd.index("--split-mode") + 1] == "layer"

    def test_split_mode_override_is_emitted(self) -> None:
        cmd = build_server_cmd(self._minimal_cfg(split_mode="row"))
        assert cmd[cmd.index("--split-mode") + 1] == "row"
        assert cmd.count("--split-mode") == 1

    def test_invalid_split_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="split_mode"):
            self._minimal_cfg(split_mode="sideways")

    def test_smaller_model_no_host_buffer_flag(self) -> None:
        """--no-host flag is emitted when no_host_buffer is True."""
        cmd = build_server_cmd(self._minimal_cfg(no_host_buffer=True))
        assert "--no-host" in cmd

    def test_smaller_model_no_host_buffer_absent_by_default(self) -> None:
        """--no-host flag is absent by default."""
        cmd = build_server_cmd(self._minimal_cfg())
        assert "--no-host" not in cmd

    def test_smaller_model_mmproj_offload_true(self) -> None:
        """--no-mmproj-offload is absent when mmproj_offload is True."""
        cmd = build_server_cmd(self._minimal_cfg(mmproj_offload=True))
        assert "--no-mmproj-offload" not in cmd

    def test_smaller_model_mmproj_offload_false(self) -> None:
        """--no-mmproj-offload is emitted when mmproj_offload is False."""
        cmd = build_server_cmd(self._minimal_cfg(mmproj_offload=False))
        assert "--no-mmproj-offload" in cmd

    def test_smaller_model_all_flags_combined(self) -> None:
        """All smaller-model flags can be emitted together."""
        cmd = build_server_cmd(
            self._minimal_cfg(
                kv_unified=True,
                mmproj_offload=True,
                load_mode="mlock",
                no_host_buffer=True,
            )
        )
        assert "--kv-unified" in cmd
        assert cmd[cmd.index("--load-mode") + 1] == "mlock"
        assert "--no-host" in cmd
        assert "--no-mmproj-offload" not in cmd


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
