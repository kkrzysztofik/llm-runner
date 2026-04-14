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

    @pytest.mark.parametrize(
        ("mode", "expected_dry_run_mode"),
        [
            ("summary-balanced", None),
            ("summary-fast", None),
            ("qwen35", None),
            ("both", None),
        ],
    )
    def test_all_non_dryrun_modes(self, mode: str, expected_dry_run_mode: str | None) -> None:
        """All non-dry-run modes should work."""
        args = parse_args([mode])
        assert args.mode == mode
        assert args.dry_run_mode == expected_dry_run_mode
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
