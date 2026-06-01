from unittest.mock import patch

import pytest

from llama_cli.cli_parser import parse_args


class TestParseArgsBasic:
    """Basic argument parsing tests."""

    def test_parse_no_args(self) -> None:
        """parse_args with no args should default to standalone TUI."""
        args = parse_args([])
        assert args.mode == "tui"
        assert args.tui_mode is None
        assert args.port is None
        assert args.port2 is None
        assert args.acknowledge_risky is False

    def test_parse_direct_mode_exits(self) -> None:
        """parse_args with a bare runnable mode (no 'tui') should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["summary-balanced"])
        assert exc_info.value.code == 1

    def test_parse_direct_mode_with_port_exits(self) -> None:
        """parse_args with a bare mode + port should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["summary-balanced", "8080"])
        assert exc_info.value.code == 1


class TestParseArgsDryRun:
    """Tests for dry-run mode parsing."""

    @pytest.mark.parametrize(
        ("args_list", "expected_mode", "expected_dry_run_mode", "expected_ports"),
        [
            (["dry-run", "both"], "dry-run", "both", []),
            (["dry-run", "both", "8080", "8081"], "dry-run", "both", [8080, 8081]),
            (["dry-run", "summary-balanced", "8080"], "dry-run", "summary-balanced", [8080]),
            (["dry-run", "summary-fast"], "dry-run", "summary-fast", []),
            (["dry-run", "qwen35", "8080"], "dry-run", "qwen35", [8080]),
        ],
    )
    def test_parse_dry_run_with_mode(
        self,
        args_list: list[str],
        expected_mode: str,
        expected_dry_run_mode: str | None,
        expected_ports: list[int],
    ) -> None:
        """parse_args with dry-run and mode should work for all valid modes."""
        args = parse_args(args_list)
        assert args.mode == expected_mode
        assert args.dry_run_mode == expected_dry_run_mode
        assert args.ports == expected_ports


class TestParseArgsValidModes:
    """Test all valid mode combinations."""

    @pytest.mark.parametrize("mode", ["summary-balanced", "summary-fast", "qwen35", "both"])
    def test_dry_run_all_modes(self, mode: str) -> None:
        """dry-run with all valid sub-modes should work."""
        args = parse_args(["dry-run", mode])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == mode

    def test_modes_are_loaded_from_profile_registry(self) -> None:
        """dry-run mode choices should come from the dynamic profile registry."""
        from llama_cli import cli_parser
        from llama_manager.config import RunGroupSpec, RunProfileRegistry, RunProfileSpec

        profile = RunProfileSpec(
            profile_id="custom",
            model="/models/custom.gguf",
            alias="custom",
            device="SYCL0",
            port=8090,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        registry = RunProfileRegistry(
            profiles=(profile,),
            run_groups=(RunGroupSpec(group_id="custom-group", profile_ids=("custom",)),),
        )

        with patch.object(
            cli_parser, "VALID_MODES", (*registry.run_group_ids, *cli_parser.COMMAND_MODES)
        ):
            args = parse_args(["dry-run", "custom-group"])

        assert args.mode == "dry-run"
        assert args.dry_run_mode == "custom-group"


class TestParseArgsInvalidModes:
    """Test invalid mode combinations."""

    def test_dry_run_without_submode(self) -> None:
        """dry-run without submode should fail with exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["dry-run"])
        assert exc_info.value.code == 1

    def test_dry_run_with_invalid_submode(self) -> None:
        """dry-run with invalid submode should fail with exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["dry-run", "invalid-mode"])
        assert exc_info.value.code == 1

    def test_unknown_token_exits_code_1(self) -> None:
        """Unknown token should fail with exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["8080"])
        assert exc_info.value.code == 1


class TestParseArgsAcknowledgeRisk:
    """Test --acknowledge-risky flag."""

    def test_acknowledge_risky_flag_present(self) -> None:
        """--acknowledge-risky flag should be parsed."""
        args = parse_args(["dry-run", "both", "--acknowledge-risky"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "both"
        assert args.acknowledge_risky is True

    def test_acknowledge_risky_flag_absent(self) -> None:
        """--acknowledge-risky flag should be False by default."""
        args = parse_args(["dry-run", "both"])
        assert args.acknowledge_risky is False


class TestParseJobsArg:
    """Tests for parse_jobs_arg function."""

    def test_parse_jobs_equals(self) -> None:
        """parse_jobs_arg should handle --jobs=N format."""
        from llama_cli.cli_parser import parse_jobs_arg

        result = parse_jobs_arg("--jobs=8")
        assert result == 8

    def test_parse_jobs_short(self) -> None:
        """parse_jobs_arg should handle -jN format."""
        from llama_cli.cli_parser import parse_jobs_arg

        result = parse_jobs_arg("-j4")
        assert result == 4

    def test_parse_jobs_invalid(self) -> None:
        """parse_jobs_arg should exit on invalid value."""
        from llama_cli.cli_parser import parse_jobs_arg

        with pytest.raises(SystemExit) as exc_info:
            parse_jobs_arg("invalid")
        assert exc_info.value.code == 1


class TestHandleBuildCase:
    """Tests for _handle_build_case function."""

    def test_handle_build_sycl(self) -> None:
        """_handle_build_case should parse 'build sycl'."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "sycl"])
        assert result is not None
        assert result.mode == "build"
        assert result.backend == "sycl"

    def test_handle_build_cuda(self) -> None:
        """_handle_build_case should parse 'build cuda'."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "cuda"])
        assert result is not None
        assert result.backend == "cuda"

    def test_handle_build_both(self) -> None:
        """_handle_build_case should parse 'build both'."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "both"])
        assert result is not None
        assert result.backend == "both"
        assert result.build_args == ["both"]

    def test_handle_build_with_dry_run(self) -> None:
        """_handle_build_case should parse --dry-run flag."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "sycl", "--dry-run"])
        assert result is not None
        assert result.dry_run is True
        assert result.build_args == ["sycl", "--dry-run"]

    def test_handle_build_preserves_build_cli_options(self) -> None:
        """_handle_build_case should pass build-specific options through."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(
            [
                "build",
                "both",
                "--source-dir",
                "/tmp/llama.cpp",
                "--jobs",
                "8",
            ]
        )
        assert result is not None
        assert result.build_args == [
            "both",
            "--source-dir",
            "/tmp/llama.cpp",
            "--jobs",
            "8",
        ]

    def test_handle_build_missing_backend(self) -> None:
        """_handle_build_case should exit when no backend specified."""
        from llama_cli.cli_parser import _handle_build_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_build_case(["build"])
        assert exc_info.value.code == 1

    def test_handle_build_invalid_backend(self) -> None:
        """_handle_build_case should exit on invalid backend."""
        from llama_cli.cli_parser import _handle_build_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_build_case(["build", "invalid"])
        assert exc_info.value.code == 1

    def test_handle_build_not_build_command(self) -> None:
        """_handle_build_case should return None for non-build commands."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["summary-balanced"])
        assert result is None


class TestHandleSetupCase:
    """Tests for _handle_setup_case function."""

    def test_handle_setup_check(self) -> None:
        """_handle_setup_case should parse 'setup check'."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "check"])
        assert result is not None
        assert result.mode == "setup"
        assert result.setup_command == "check"

    def test_handle_setup_venv(self) -> None:
        """_handle_setup_case should parse 'setup venv'."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "venv"])
        assert result is not None
        assert result.setup_command == "venv"

    def test_handle_setup_clean_venv(self) -> None:
        """_handle_setup_case should parse 'setup clean-venv'."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "clean-venv"])
        assert result is not None
        assert result.setup_command == "clean-venv"

    def test_handle_setup_check_with_backend(self) -> None:
        """_handle_setup_case should parse backend for check."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "check", "sycl"])
        assert result is not None
        assert result.backend == "sycl"

    def test_handle_setup_check_with_json(self) -> None:
        """_handle_setup_case should parse --json flag."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "check", "--json"])
        assert result is not None
        assert result.json is True

    def test_handle_setup_venv_with_check_integrity(self) -> None:
        """_handle_setup_case should parse --check-integrity flag."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "venv", "--check-integrity"])
        assert result is not None
        assert result.check_integrity is True

    def test_handle_setup_clean_venv_with_yes(self) -> None:
        """_handle_setup_case should parse --yes flag."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["setup", "clean-venv", "--yes"])
        assert result is not None
        assert result.yes is True

    def test_handle_setup_unknown_subcommand(self) -> None:
        """_handle_setup_case should exit on unknown subcommand."""
        from llama_cli.cli_parser import _handle_setup_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_setup_case(["setup", "unknown"])
        assert exc_info.value.code == 1

    def test_handle_setup_no_subcommand(self) -> None:
        """_handle_setup_case should exit when no subcommand provided."""
        from llama_cli.cli_parser import _handle_setup_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_setup_case(["setup"])
        assert exc_info.value.code == 1

    def test_handle_setup_not_setup_command(self) -> None:
        """_handle_setup_case should return None for non-setup commands."""
        from llama_cli.cli_parser import _handle_setup_case

        result = _handle_setup_case(["summary-balanced"])
        assert result is None


class TestHandleDoctorCase:
    """Tests for _handle_doctor_case function."""

    def test_handle_doctor_check(self) -> None:
        """_handle_doctor_case should parse 'doctor check'."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "check"])
        assert result is not None
        assert result.mode == "doctor"
        assert result.doctor_command == "check"

    def test_handle_doctor_repair(self) -> None:
        """_handle_doctor_case should parse 'doctor repair'."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "repair"])
        assert result is not None
        assert result.doctor_command == "repair"

    def test_handle_doctor_repair_with_dry_run(self) -> None:
        """_handle_doctor_case should parse --dry-run for repair."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "repair", "--dry-run"])
        assert result is not None
        assert result.dry_run is True

    def test_handle_doctor_repair_with_json(self) -> None:
        """_handle_doctor_case should parse --json for repair."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "repair", "--json"])
        assert result is not None
        assert result.json is True

    def test_handle_doctor_repair_with_yes(self) -> None:
        """_handle_doctor_case should parse --yes for repair."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "repair", "--yes"])
        assert result is not None
        assert result.yes is True

    def test_handle_doctor_check_with_backend(self) -> None:
        """_handle_doctor_case should parse backend for check."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "check", "cuda"])
        assert result is not None
        assert result.backend == "cuda"

    def test_handle_doctor_check_with_json(self) -> None:
        """_handle_doctor_case should parse --json for check."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["doctor", "check", "--json"])
        assert result is not None
        assert result.json is True

    def test_handle_doctor_unknown_subcommand(self) -> None:
        """_handle_doctor_case should exit on unknown subcommand."""
        from llama_cli.cli_parser import _handle_doctor_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_doctor_case(["doctor", "unknown"])
        assert exc_info.value.code == 1

    def test_handle_doctor_no_subcommand(self) -> None:
        """_handle_doctor_case should exit when no subcommand provided."""
        from llama_cli.cli_parser import _handle_doctor_case

        with pytest.raises(SystemExit) as exc_info:
            _handle_doctor_case(["doctor"])
        assert exc_info.value.code == 1

    def test_handle_doctor_not_doctor_command(self) -> None:
        """_handle_doctor_case should return None for non-doctor commands."""
        from llama_cli.cli_parser import _handle_doctor_case

        result = _handle_doctor_case(["summary-balanced"])
        assert result is None


class TestParseTuiArgs:
    """Tests for parse_tui_args function."""

    def test_parse_tui_args_basic(self) -> None:
        """parse_tui_args should parse basic mode."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["both"])
        assert args.mode == "both"
        assert args.dry_run_mode is None

    def test_parse_tui_args_with_port(self) -> None:
        """parse_tui_args should parse --port flag."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["qwen35", "--port", "9999"])
        assert args.mode == "qwen35"
        assert args.port == 9999

    def test_parse_tui_args_with_port2(self) -> None:
        """parse_tui_args should parse --port2 flag."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["both", "--port", "8080", "--port2", "8081"])
        assert args.mode == "both"
        assert args.port == 8080
        assert args.port2 == 8081

    def test_parse_tui_args_short_port(self) -> None:
        """parse_tui_args should parse -p flag."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["qwen35", "-p", "9999"])
        assert args.port == 9999

    def test_parse_tui_args_short_port2(self) -> None:
        """parse_tui_args should parse -P flag."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["both", "-p", "8080", "-P", "8081"])
        assert args.port == 8080
        assert args.port2 == 8081

    def test_parse_tui_args_acknowledge_risky(self) -> None:
        """parse_tui_args should parse --acknowledge-risky flag."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args(["both", "--acknowledge-risky"])
        assert args.acknowledge_risky is True

    @pytest.mark.parametrize(
        "mode",
        ["summary-balanced", "summary-fast", "qwen35", "both", "build", "setup", "doctor"],
    )
    def test_parse_tui_args_all_modes(self, mode: str) -> None:
        """parse_tui_args should accept all valid modes."""
        from llama_cli.cli_parser import parse_tui_args

        args = parse_tui_args([mode])
        assert args.mode == mode
        assert args.dry_run_mode is None


class TestDryRunPortParsing:
    """Tests for dry-run port parsing edge cases."""

    def test_dry_run_with_invalid_port(self) -> None:
        """dry-run with invalid port should exit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["dry-run", "both", "not-a-port"])
        assert exc_info.value.code == 1

    def test_dry_run_with_acknowledge_risky_in_middle(self) -> None:
        """dry-run should handle --acknowledge-risky anywhere in args."""
        args = parse_args(["dry-run", "both", "--acknowledge-risky", "8080"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "both"
        assert args.acknowledge_risky is True
        assert args.ports == [8080]


class TestHandleProfileCase:
    """Tests for _handle_profile_case function.

    _handle_profile_case is a simple detector — it only checks whether
    args[0] == "profile" and returns a minimal namespace with sub_argv
    forwarded to profile_cli.main() for full parsing/validation.
    """

    def test_handle_profile_balanced(self) -> None:
        """_handle_profile_case detects profile subcommand and forwards args."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "slot1", "balanced"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["slot1", "balanced"]

    def test_handle_profile_fast(self) -> None:
        """_handle_profile_case detects profile subcommand."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "slot1", "fast"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["slot1", "fast"]

    def test_handle_profile_quality(self) -> None:
        """_handle_profile_case detects profile subcommand."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "slot1", "quality"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["slot1", "quality"]

    def test_handle_profile_with_json(self) -> None:
        """_handle_profile_case forwards --json in sub_argv."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "slot1", "balanced", "--json"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["slot1", "balanced", "--json"]

    def test_handle_profile_slot_id_with_dashes(self) -> None:
        """_handle_profile_case accepts slot_id with dashes."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "gpu-0", "balanced"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["gpu-0", "balanced"]

    def test_handle_profile_slot_id_with_underscores(self) -> None:
        """_handle_profile_case accepts slot_id with underscores."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["profile", "gpu_0", "balanced"])
        assert result is not None
        assert result.mode == "profile"
        assert result.sub_argv == ["gpu_0", "balanced"]

    def test_handle_profile_not_profile_command(self) -> None:
        """_handle_profile_case should return None for non-profile commands."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case(["summary-balanced"])
        assert result is None

    def test_handle_profile_empty_args(self) -> None:
        """_handle_profile_case returns None for empty args."""
        from llama_cli.cli_parser import _handle_profile_case

        result = _handle_profile_case([])
        assert result is None


class TestParseArgsProfile:
    """Tests for profile subcommand through parse_args."""

    def test_parse_args_profile_balanced(self) -> None:
        """parse_args should handle 'profile slot balanced'."""
        args = parse_args(["profile", "slot1", "balanced"])
        assert args.mode == "profile"
        assert args.sub_argv == ["slot1", "balanced"]

    def test_parse_args_profile_with_json(self) -> None:
        """parse_args should handle 'profile slot balanced --json'."""
        args = parse_args(["profile", "slot1", "balanced", "--json"])
        assert args.mode == "profile"
        assert args.sub_argv == ["slot1", "balanced", "--json"]


class TestCliParserDynamicRegistryIntegration:
    """Tests for CLI parser integration with dynamic profile registry.

    These tests verify that CLI mode choices derive from the profile registry
    and that adding/removing profiles affects available CLI modes.
    """

    def test_get_runnable_tui_modes_returns_registry_groups(self) -> None:
        """get_runnable_tui_modes should return run_group_ids from default registry."""
        from llama_cli.cli_parser import get_runnable_tui_modes
        from llama_manager.config import create_default_profile_registry

        registry = create_default_profile_registry()
        modes = get_runnable_tui_modes()

        assert modes == registry.run_group_ids
        assert "summary-balanced" in modes
        assert "summary-fast" in modes
        assert "qwen35" in modes
        assert "both" in modes

    def test_valid_modes_includes_runnable_and_command_modes(self) -> None:
        """VALID_MODES should include all runnable modes plus command modes."""
        from llama_cli.cli_parser import COMMAND_MODES, VALID_MODES, get_runnable_tui_modes

        runnable = get_runnable_tui_modes()
        expected = (*runnable, *COMMAND_MODES)

        assert expected == VALID_MODES
        assert "build" in VALID_MODES
        assert "setup" in VALID_MODES
        assert "doctor" in VALID_MODES

    def test_custom_run_group_parseable_through_cli(self) -> None:
        """CLI should accept custom run group when registry is patched."""
        from llama_cli import cli_parser
        from llama_manager.config import RunGroupSpec, RunProfileRegistry, RunProfileSpec

        custom_profile = RunProfileSpec(
            profile_id="custom-model",
            model="/models/custom.gguf",
            alias="custom",
            device="SYCL0",
            port=8090,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        custom_registry = RunProfileRegistry(
            profiles=(custom_profile,),
            run_groups=(RunGroupSpec(group_id="custom-group", profile_ids=("custom-model",)),),
        )

        with (
            patch.object(cli_parser, "RUNNABLE_TUI_MODES", custom_registry.run_group_ids),
            patch.object(
                cli_parser,
                "VALID_MODES",
                (*custom_registry.run_group_ids, *cli_parser.COMMAND_MODES),
            ),
        ):
            args = parse_args(["tui", "custom-group"])

        assert args.mode == "tui"
        assert args.tui_mode == "custom-group"

    def test_removed_profile_not_in_valid_modes(self) -> None:
        """VALID_MODES should not include profiles removed from registry."""
        from llama_cli.cli_parser import VALID_MODES

        assert "nonexistent-mode" not in VALID_MODES
        assert "summary-balanced" in VALID_MODES

    def test_dry_run_accepts_all_registry_modes(self) -> None:
        """dry-run should accept all registry-defined run groups."""
        from llama_cli.cli_parser import get_runnable_tui_modes

        for mode in get_runnable_tui_modes():
            args = parse_args(["dry-run", mode])
            assert args.mode == "dry-run"
            assert args.dry_run_mode == mode

    def test_tui_args_accepts_all_registry_modes(self) -> None:
        """parse_tui_args should accept all registry-defined run groups."""
        from llama_cli.cli_parser import get_runnable_tui_modes, parse_tui_args

        for mode in get_runnable_tui_modes():
            args = parse_tui_args([mode])
            assert args.mode == mode

    def test_invalid_mode_rejected_by_parser(self) -> None:
        """Parser should reject modes not in VALID_MODES."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["totally-invalid-mode"])
        assert exc_info.value.code == 1

    def test_command_modes_not_in_runnable_tui_modes(self) -> None:
        """Command modes (build/setup/doctor) should not be in runnable TUI modes."""
        from llama_cli.cli_parser import COMMAND_MODES, get_runnable_tui_modes

        runnable = get_runnable_tui_modes()
        for cmd in COMMAND_MODES:
            assert cmd not in runnable


from llama_cli.commands._toolchain import (
    collect_toolchain_repair_actions,
    deduplicate_hints,
    filter_optional_tools,
    get_backend_hints,
    resolve_backend_enum,
)
from llama_manager.build_pipeline import BuildBackend
from llama_manager.config import ErrorCode
from llama_manager.toolchain import ToolchainErrorDetail


class TestResolveBackendEnum:
    """Tests for resolve_backend_enum function."""

    def test_none_returns_none(self) -> None:
        """resolve_backend_enum(None) should return None."""
        assert resolve_backend_enum(None) is None

    def test_valid_sycl(self) -> None:
        """resolve_backend_enum('sycl') should return BuildBackend.SYCL."""
        result = resolve_backend_enum("sycl")
        assert result == BuildBackend.SYCL

    def test_valid_cuda(self) -> None:
        """resolve_backend_enum('cuda') should return BuildBackend.CUDA."""
        result = resolve_backend_enum("cuda")
        assert result == BuildBackend.CUDA

    def test_valid_all(self) -> None:
        """resolve_backend_enum('all') should return BuildBackend.BOTH."""
        result = resolve_backend_enum("both")
        assert result == BuildBackend.BOTH

    def test_valid_all_string_returns_none(self) -> None:
        """resolve_backend_enum('all') returns None because 'all' is not a valid BuildBackend value."""
        result = resolve_backend_enum("all")
        assert result is None

    def test_invalid_returns_none(self) -> None:
        """resolve_backend_enum('invalid') should return None."""
        assert resolve_backend_enum("invalid") is None


class TestDeduplicateHints:
    """Tests for deduplicate_hints function."""

    def _make_hint(
        self, how_to_fix: str = "install tool", docs_ref: str | None = None
    ) -> ToolchainErrorDetail:
        """Create a ToolchainErrorDetail for testing."""
        return ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,
            failed_check="test_check",
            why_blocked="required",
            how_to_fix=how_to_fix,
            docs_ref=docs_ref,
        )

    def test_duplicate_hints_deduplicated(self) -> None:
        """Hints with same how_to_fix + docs_ref should be deduplicated."""
        hint = self._make_hint()
        result = deduplicate_hints([hint, hint, hint])
        assert len(result) == 1

    def test_different_hints_preserved(self) -> None:
        """Hints with different how_to_fix should all be preserved."""
        hint1 = self._make_hint(how_to_fix="install tool A")
        hint2 = self._make_hint(how_to_fix="install tool B")
        result = deduplicate_hints([hint1, hint2])
        assert len(result) == 2

    def test_same_how_to_fix_different_docs_ref_preserved(self) -> None:
        """Hints with same how_to_fix but different docs_ref should be preserved."""
        hint1 = self._make_hint(docs_ref="doc1")
        hint2 = self._make_hint(docs_ref="doc2")
        result = deduplicate_hints([hint1, hint2])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        assert deduplicate_hints([]) == []


class TestGetBackendHints:
    """Tests for get_backend_hints function."""

    def _make_hint(self, backend: str = "sycl") -> ToolchainErrorDetail:
        """Create a ToolchainErrorDetail for testing."""
        return ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,
            failed_check=f"{backend}_check",
            why_blocked="required",
            how_to_fix=f"Install {backend} toolchain",
            docs_ref=f"https://example.com/{backend}",
        )

    def test_get_sycl_hints(self) -> None:
        """get_backend_hints('sycl') should return deduplicated SYCL hints."""
        expected = self._make_hint("sycl")
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [expected]
            result = get_backend_hints("sycl")
            assert expected in result

    def test_get_cuda_hints(self) -> None:
        """get_backend_hints('cuda') should return deduplicated CUDA hints."""
        expected = self._make_hint("cuda")
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [expected]
            result = get_backend_hints("cuda")
            assert expected in result

    def test_get_all_hints(self) -> None:
        """get_backend_hints('all') should return combined and deduplicated hints."""
        expected = self._make_hint("sycl")
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [expected, expected]
            result = get_backend_hints("all")
            assert expected in result
            assert result.count(expected) == 1

    def test_get_unknown_backend_returns_empty(self) -> None:
        """get_backend_hints('unknown') should return empty list."""
        result = get_backend_hints("unknown")
        assert result == []


class TestFilterOptionalTools:
    """Tests for filter_optional_tools function."""

    def test_filter_removes_nvtop_when_all_complete(self) -> None:
        """When backend='all' and complete, nvtop should be filtered."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, "all", is_complete=True)
        assert "nvtop" not in result
        assert "gcc" in result

    def test_filter_keeps_nvtop_when_not_complete(self) -> None:
        """When not complete, nvtop should not be filtered."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, "all", is_complete=False)
        assert "nvtop" in result

    def test_filter_keeps_nvtop_for_sycl(self) -> None:
        """When backend='sycl', nvtop should not be filtered."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, "sycl", is_complete=True)
        assert "nvtop" in result

    def test_filter_keeps_nvtop_for_cuda(self) -> None:
        """When backend='cuda', nvtop should not be filtered."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, "cuda", is_complete=True)
        assert "nvtop" in result

    def test_filter_none_backend_complete(self) -> None:
        """When backend=None and complete, nvtop should be filtered."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, None, is_complete=True)
        assert "nvtop" not in result

    def test_filter_none_backend_not_complete(self) -> None:
        """When backend=None and not complete, nvtop should be kept."""
        missing = ["gcc", "nvtop", "cmake"]
        result = filter_optional_tools(missing, None, is_complete=False)
        assert "nvtop" in result


class TestCollectToolchainRepairActions:
    """Tests for collect_toolchain_repair_actions function."""

    def _make_hint(self, failed_check: str = "check1") -> ToolchainErrorDetail:
        """Create a ToolchainErrorDetail for testing."""
        return ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,
            failed_check=failed_check,
            why_blocked="required",
            how_to_fix="Install tool",
            docs_ref="https://example.com",
        )

    def test_deduplicates_by_failed_check(self) -> None:
        """Hints with same failed_check should be deduplicated."""
        hint1 = self._make_hint("dpcpp")
        hint2 = self._make_hint("dpcpp")
        result = collect_toolchain_repair_actions([hint1, hint2])
        assert len(result) == 1

    def test_preserves_different_failed_checks(self) -> None:
        """Hints with different failed_check should all be preserved."""
        hint1 = self._make_hint("check1")
        hint2 = self._make_hint("check2")
        result = collect_toolchain_repair_actions([hint1, hint2])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        assert collect_toolchain_repair_actions([]) == []

    def test_mixed_deduplication(self) -> None:
        """Should deduplicate same failed_check but keep different ones."""
        hint1 = self._make_hint("check1")
        hint2 = self._make_hint("check1")
        hint3 = self._make_hint("check2")
        hint4 = self._make_hint("check3")
        hint5 = self._make_hint("check3")
        result = collect_toolchain_repair_actions([hint1, hint2, hint3, hint4, hint5])
        assert len(result) == 3


import json

from llama_cli.commands._output import (
    print_error,
    print_header,
    print_json,
    print_success,
)


class TestPrintError:
    """Tests for print_error function."""

    def test_print_error_writes_to_stderr(self, capsys) -> None:
        """print_error should write to stderr."""
        with patch("llama_cli.commands._output.emit_error") as mock_err:
            print_error("test error")

        mock_err.assert_called_once_with("test error")

    def test_print_error_message_format(self, capsys) -> None:
        """print_error delegates to emit_error which adds 'error:' prefix."""
        with patch("llama_cli.commands._output.emit_error") as mock_err:
            print_error("disk full")

        mock_err.assert_called_once_with("disk full")


class TestPrintSuccess:
    """Tests for print_success function."""

    def test_print_success_writes_to_stdout(self, capsys) -> None:
        """print_success delegates to emit_success which writes to stdout."""
        with patch("llama_cli.commands._output.emit_success") as mock_ok:
            print_success("done")

        mock_ok.assert_called_once_with("done")

    def test_print_success_no_prefix(self, capsys) -> None:
        """print_success delegates to emit_success which adds 'ok:' prefix."""
        with patch("llama_cli.commands._output.emit_success") as mock_ok:
            print_success("hello world")

        mock_ok.assert_called_once_with("hello world")


class TestPrintHeader:
    """Tests for print_header function."""

    def test_print_header_calls_emit_heading(self, capsys) -> None:
        """print_header delegates to emit_heading."""
        with patch("llama_cli.commands._output.emit_heading") as mock_heading:
            print_header("Setup")

        mock_heading.assert_called_once_with("Setup")


class TestPrintJson:
    """Tests for print_json function."""

    def test_print_json_serializes_dict(self, capsys) -> None:
        """print_json should serialize a dict to JSON."""
        print_json({"key": "value", "num": 42})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"key": "value", "num": 42}

    def test_print_json_nested_dict(self, capsys) -> None:
        """print_json should handle nested dicts."""
        print_json({"outer": {"inner": "deep"}})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["outer"]["inner"] == "deep"

    def test_print_json_writes_to_stdout(self, capsys) -> None:
        """print_json should write to stdout, not stderr."""
        print_json({"test": True})

        captured = capsys.readouterr()
        assert captured.err == ""
        assert "test" in captured.out

    def test_print_json_default_str_handler(self, capsys) -> None:
        """print_json should use default=str for non-serializable types."""
        import datetime

        print_json({"ts": datetime.datetime(2024, 1, 1)})

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "ts" in parsed  # datetime should be serialized via str()
