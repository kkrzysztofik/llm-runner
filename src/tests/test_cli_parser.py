"""Tests for CLI argument parsing, especially dry-run mode.

Test Tasks:
- Validate parser behavior for dry-run mode with mode argument
- Ensure parser and help/examples are consistent and valid
"""

import pytest

from llama_cli.cli_parser import parse_args


class TestParseArgsBasic:
    """Basic argument parsing tests."""

    def test_parse_no_args(self) -> None:
        """parse_args with no args should return mode=None."""
        args = parse_args([])
        assert args.mode is None

    def test_parse_single_mode(self) -> None:
        """parse_args with single mode should work."""
        args = parse_args(["summary-balanced"])
        assert args.mode == "summary-balanced"
        assert args.dry_run_mode is None
        assert args.ports == []

    def test_parse_port_argument(self) -> None:
        """parse_args with port should capture it."""
        args = parse_args(["summary-balanced", "8080"])
        assert args.mode == "summary-balanced"
        assert args.ports == [8080]

    def test_parse_two_ports(self) -> None:
        """parse_args with two ports should capture both."""
        args = parse_args(["both", "8080", "8081"])
        assert args.mode == "both"
        assert args.ports == [8080, 8081]


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
    def test_all_non_dryrun_modes(self, mode: str) -> None:
        """All non-dry-run modes should work."""
        args = parse_args([mode])
        assert args.mode == mode
        assert args.dry_run_mode is None
        assert args.ports == []

    @pytest.mark.parametrize("mode", ["summary-balanced", "summary-fast", "qwen35", "both"])
    def test_dry_run_all_modes(self, mode: str) -> None:
        """dry-run with all valid sub-modes should work."""
        args = parse_args(["dry-run", mode])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == mode


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

    def test_port_without_mode(self) -> None:
        """Port without mode should fail with exit code 2 (argparse error)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["8080"])
        assert exc_info.value.code == 2


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

    def test_handle_build_with_dry_run(self) -> None:
        """_handle_build_case should parse --dry-run flag."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "sycl", "--dry-run"])
        assert result is not None
        assert result.dry_run is True

    def test_handle_build_with_jobs(self) -> None:
        """_handle_build_case should parse -jN flag."""
        from llama_cli.cli_parser import _handle_build_case

        result = _handle_build_case(["build", "sycl", "-j8"])
        assert result is not None
        assert result.jobs == 8

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
