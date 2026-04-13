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

    def test_parse_dry_run_without_mode(self) -> None:
        """parse_args with dry-run but no mode should fail."""
        with pytest.raises(SystemExit):
            parse_args(["dry-run"])

    def test_parse_dry_run_with_mode(self) -> None:
        """parse_args with dry-run and mode should work."""
        args = parse_args(["dry-run", "both"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "both"
        assert args.ports == []

    def test_parse_dry_run_with_ports(self) -> None:
        """parse_args with dry-run, mode, and ports should work."""
        args = parse_args(["dry-run", "both", "8080", "8081"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "both"
        assert args.ports == [8080, 8081]

    def test_parse_dry_run_summary_balanced(self) -> None:
        """parse_args with dry-run summary-balanced should work."""
        args = parse_args(["dry-run", "summary-balanced", "8080"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "summary-balanced"
        assert args.ports == [8080]

    def test_parse_dry_run_summary_fast(self) -> None:
        """parse_args with dry-run summary-fast should work."""
        args = parse_args(["dry-run", "summary-fast"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "summary-fast"
        assert args.ports == []

    def test_parse_dry_run_qwen35(self) -> None:
        """parse_args with dry-run qwen35 should work."""
        args = parse_args(["dry-run", "qwen35", "8080"])
        assert args.mode == "dry-run"
        assert args.dry_run_mode == "qwen35"
        assert args.ports == [8080]


class TestParseArgsValidModes:
    """Test all valid mode combinations."""

    def test_all_non_dryrun_modes(self) -> None:
        """All non-dry-run modes should work."""
        for mode in ["summary-balanced", "summary-fast", "qwen35", "both"]:
            args = parse_args([mode])
            assert args.mode == mode
            assert args.dry_run_mode is None
            assert args.ports == []

    def test_dry_run_all_modes(self) -> None:
        """dry-run with all valid sub-modes should work."""
        for mode in ["summary-balanced", "summary-fast", "qwen35", "both"]:
            args = parse_args(["dry-run", mode])
            assert args.mode == "dry-run"
            assert args.dry_run_mode == mode


class TestParseArgsInvalidModes:
    """Test invalid mode combinations."""

    def test_dry_run_without_submode(self) -> None:
        """dry-run without submode should fail."""
        with pytest.raises(SystemExit):
            parse_args(["dry-run"])

    def test_dry_run_with_invalid_submode(self) -> None:
        """dry-run with invalid submode should fail."""
        with pytest.raises(SystemExit):
            parse_args(["dry-run", "invalid-mode"])

    def test_port_without_mode(self) -> None:
        """Port without mode should fail."""
        with pytest.raises(SystemExit):
            parse_args(["8080"])


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
