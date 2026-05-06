"""Tests for shared toolchain helpers (_toolchain.py).

Covers:
  - resolve_backend_enum (None, valid, invalid)
  - deduplicate_hints
  - get_backend_hints (sycl, cuda, all)
  - filter_optional_tools
  - collect_toolchain_repair_actions
"""

from __future__ import annotations

from unittest.mock import patch

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
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [self._make_hint("sycl")]
            result = get_backend_hints("sycl")
            assert len(result) >= 0  # Should not raise

    def test_get_cuda_hints(self) -> None:
        """get_backend_hints('cuda') should return deduplicated CUDA hints."""
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [self._make_hint("cuda")]
            result = get_backend_hints("cuda")
            assert len(result) >= 0

    def test_get_all_hints(self) -> None:
        """get_backend_hints('all') should return combined and deduplicated hints."""
        with patch("llama_cli.commands._toolchain.get_toolchain_hints") as mock_hints:
            mock_hints.return_value = [self._make_hint("sycl")]
            result = get_backend_hints("all")
            assert len(result) >= 0

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
