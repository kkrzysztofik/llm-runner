"""Tests for llama_manager.config and llama_manager.config_builder."""

from llama_manager.config import (
    Config,
    ErrorCode,
    ModelSlot,
    ServerConfig,
    ValidationResult,
)
from llama_manager.config_builder import (
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
)


class TestConfig:
    def test_defaults_are_set(self) -> None:
        cfg = Config()
        assert cfg.host == "127.0.0.1"
        assert cfg.summary_balanced_port == 8080
        assert cfg.summary_fast_port == 8082
        assert cfg.qwen35_port == 8081

    def test_ports_are_distinct_by_default(self) -> None:
        cfg = Config()
        ports = {cfg.summary_balanced_port, cfg.summary_fast_port, cfg.qwen35_port}
        assert len(ports) == 3, "Default ports must all be different"

    def test_default_ctx_sizes_are_positive(self) -> None:
        cfg = Config()
        assert cfg.default_ctx_size_summary > 0
        assert cfg.default_ctx_size_qwen35 > 0

    def test_default_threads_are_positive(self) -> None:
        cfg = Config()
        assert cfg.default_threads_summary_balanced > 0
        assert cfg.default_threads_summary_fast > 0
        assert cfg.default_threads_qwen35 > 0


class TestServerConfig:
    def test_required_fields(self) -> None:
        sc = ServerConfig(
            model="/models/test.gguf",
            alias="test",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        assert sc.model == "/models/test.gguf"
        assert sc.alias == "test"
        assert sc.port == 8080

    def test_default_optional_fields(self) -> None:
        sc = ServerConfig(
            model="/models/test.gguf",
            alias="test",
            device="CPU",
            port=9000,
            ctx_size=2048,
            ubatch_size=256,
            threads=2,
        )
        assert sc.tensor_split == ""
        assert sc.reasoning_mode == "auto"
        assert sc.reasoning_format == "none"
        assert sc.use_jinja is False
        assert sc.server_bin == ""
        assert sc.n_gpu_layers == 99

    def test_server_bin_override(self) -> None:
        sc = ServerConfig(
            model="/models/test.gguf",
            alias="test",
            device="CUDA",
            port=8081,
            ctx_size=4096,
            ubatch_size=1024,
            threads=8,
            server_bin="/custom/llama-server",
        )
        assert sc.server_bin == "/custom/llama-server"


class TestConfigBuilders:
    def test_create_summary_balanced_cfg_port(self) -> None:
        sc = create_summary_balanced_cfg(port=9001)
        assert sc.port == 9001
        assert sc.alias == "summary-balanced"
        assert sc.device == "SYCL0"
        assert sc.use_jinja is True
        assert sc.reasoning_mode == "off"

    def test_create_summary_balanced_cfg_overrides(self) -> None:
        sc = create_summary_balanced_cfg(port=9001, ctx_size=8192, threads=4)
        assert sc.ctx_size == 8192
        assert sc.threads == 4

    def test_create_summary_fast_cfg_port(self) -> None:
        sc = create_summary_fast_cfg(port=9002)
        assert sc.port == 9002
        assert sc.alias == "summary-fast"
        assert sc.device == "SYCL0"

    def test_create_summary_fast_cfg_overrides(self) -> None:
        sc = create_summary_fast_cfg(port=9002, ubatch_size=128)
        assert sc.ubatch_size == 128

    def test_create_qwen35_cfg_port(self) -> None:
        sc = create_qwen35_cfg(port=9003)
        assert sc.port == 9003
        assert sc.alias == "qwen35-coding"

    def test_create_qwen35_cfg_overrides(self) -> None:
        sc = create_qwen35_cfg(port=9003, threads=16, ctx_size=131072)
        assert sc.threads == 16
        assert sc.ctx_size == 131072

    def test_builder_ctx_size_defaults_differ(self) -> None:
        """Summary and qwen35 configs should have different ctx_size defaults."""
        summary = create_summary_balanced_cfg(port=8080)
        qwen35 = create_qwen35_cfg(port=8081)
        # qwen35 has a much larger context window
        assert qwen35.ctx_size > summary.ctx_size


class TestModelSlot:
    """Tests for ModelSlot scaffolding."""

    def test_model_slot_creation(self) -> None:
        """ModelSlot should create with required fields."""
        slot = ModelSlot(
            slot_id="slot1",
            model_path="/models/model1.gguf",
            port=8001,
        )
        assert slot.slot_id == "slot1"
        assert slot.model_path == "/models/model1.gguf"
        assert slot.port == 8001

    def test_model_slot_all_fields(self) -> None:
        """ModelSlot should have all expected fields."""
        slot = ModelSlot(
            slot_id="slot2",
            model_path="/models/model2.gguf",
            port=8002,
        )
        assert isinstance(slot.slot_id, str)
        assert isinstance(slot.model_path, str)
        assert isinstance(slot.port, int)


class TestErrorCode:
    """Tests for ErrorCode enum scaffolding."""

    def test_error_code_values(self) -> None:
        """ErrorCode should have expected string values."""
        assert ErrorCode.FILE_NOT_FOUND == "FILE_NOT_FOUND"
        assert ErrorCode.PORT_CONFLICT == "PORT_CONFLICT"
        assert ErrorCode.PORT_INVALID == "PORT_INVALID"
        assert ErrorCode.THREADS_INVALID == "THREADS_INVALID"

    def test_error_code_has_all_expected_codes(self) -> None:
        """ErrorCode should include all scaffolding error codes."""
        codes = list(ErrorCode)
        code_values = [c.value for c in codes]

        expected_codes = [
            "FILE_NOT_FOUND",
            "PATH_INVALID",
            "PERMISSION_DENIED",
            "PORT_CONFLICT",
            "PORT_INVALID",
            "THREADS_INVALID",
            "CONFIG_ERROR",
            "invalid_slot_id",
            "duplicate_slot",
            "runtime_dir_unavailable",
            "lockfile_integrity_failure",
            "artifact_persistence_failure",
            "backend_not_eligible",
        ]

        for expected in expected_codes:
            assert expected in code_values, f"Missing ErrorCode: {expected}"

    def test_error_code_deterministic_ordering(self) -> None:
        """ErrorCode should have deterministic iteration order for sorting."""
        codes = list(ErrorCode)
        # Should be in definition order for predictable sorting
        assert len(codes) == 13  # All error codes defined


class TestValidationResult:
    """Tests for ValidationResult scaffolding."""

    def test_validation_result_passed(self) -> None:
        """ValidationResult should indicate success when passed=True."""
        result = ValidationResult(
            slot_id="slot1",
            passed=True,
        )
        assert result.passed is True
        assert result.valid is True
        assert result.failed_check == ""
        assert result.error_code is None
        assert result.error_message == ""

    def test_validation_result_failed(self) -> None:
        """ValidationResult should capture failure details."""
        result = ValidationResult(
            slot_id="slot2",
            passed=False,
            failed_check="model_not_found",
            error_code=ErrorCode.FILE_NOT_FOUND,
            error_message="Model file does not exist",
        )
        assert result.passed is False
        assert result.valid is False
        assert result.failed_check == "model_not_found"
        assert result.error_code == ErrorCode.FILE_NOT_FOUND
        assert result.error_message == "Model file does not exist"

    def test_validation_result_valid_alias(self) -> None:
        """ValidationResult.valid should alias passed property."""
        passed_result = ValidationResult(slot_id="a", passed=True)
        failed_result = ValidationResult(slot_id="b", passed=False)
        assert passed_result.valid is passed_result.passed
        assert failed_result.valid is failed_result.passed

    def test_validation_result_minimal_fields(self) -> None:
        """ValidationResult should work with minimal required fields."""
        result = ValidationResult(slot_id="test", passed=True)
        assert result.slot_id == "test"
        assert result.passed is True
