"""Tests for SpeculativeDecodingConfig — DFlash and existing spec types."""

import pytest

from llama_manager.config import SpeculativeDecodingConfig


class TestSpecTypeValidation:
    """spec_type accepts valid values and rejects unknown types."""

    def test_spec_type_accepts_dflash(self) -> None:
        """DFlash spec_type is accepted."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/draft.gguf",
        )
        assert cfg.spec_type == "dflash"

    def test_spec_type_accepts_ngram_mod(self) -> None:
        """ngram-mod spec_type is accepted."""
        cfg = SpeculativeDecodingConfig(spec_type="ngram-mod")
        assert cfg.spec_type == "ngram-mod"

    def test_spec_type_accepts_draft_mtp(self) -> None:
        """draft-mtp spec_type is accepted."""
        cfg = SpeculativeDecodingConfig(spec_type="draft-mtp")
        assert cfg.spec_type == "draft-mtp"

    def test_spec_type_accepts_empty(self) -> None:
        """Empty spec_type is accepted (no speculative decoding)."""
        cfg = SpeculativeDecodingConfig(spec_type="")
        assert cfg.spec_type == ""

    def test_spec_type_rejects_unknown(self) -> None:
        """Unknown spec_type raises ValueError."""
        with pytest.raises(ValueError, match="spec_type must be"):
            SpeculativeDecodingConfig(spec_type="unknown-type")


class TestDFlashDraftSourceValidation:
    """DFlash requires exactly one draft source."""

    def test_dflash_requires_exactly_one_draft_source_neither(self) -> None:
        """DFlash with neither draft source raises ValueError."""
        with pytest.raises(ValueError, match="spec_draft_model or spec_draft_hf required"):
            SpeculativeDecodingConfig(spec_type="dflash")

    def test_dflash_requires_exactly_one_draft_source_both(self) -> None:
        """DFlash with both draft sources raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            SpeculativeDecodingConfig(
                spec_type="dflash",
                spec_draft_model="/path/to/model",
                spec_draft_hf="repo:quant",
            )

    def test_dflash_accepts_local_draft_model(self) -> None:
        """DFlash with exactly one local draft model passes."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
        )
        assert cfg.spec_draft_model == "/path/to/model"
        assert cfg.spec_draft_hf == ""

    def test_dflash_accepts_hf_draft(self) -> None:
        """DFlash with exactly one HF draft passes."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_hf="repo:quant",
        )
        assert cfg.spec_draft_hf == "repo:quant"
        assert cfg.spec_draft_model == ""


class TestDFlashCrossCtxValidation:
    """spec_dflash_cross_ctx must be non-negative."""

    def test_dflash_rejects_negative_cross_ctx(self) -> None:
        """Negative spec_dflash_cross_ctx raises ValueError."""
        with pytest.raises(ValueError, match="spec_dflash_cross_ctx must be non-negative"):
            SpeculativeDecodingConfig(
                spec_type="dflash",
                spec_draft_model="/path/to/model",
                spec_dflash_cross_ctx=-1,
            )

    def test_dflash_accepts_zero_cross_ctx(self) -> None:
        """Zero spec_dflash_cross_ctx is accepted."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
            spec_dflash_cross_ctx=0,
        )
        assert cfg.spec_dflash_cross_ctx == 0

    def test_dflash_accepts_positive_cross_ctx(self) -> None:
        """Positive spec_dflash_cross_ctx is accepted."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
            spec_dflash_cross_ctx=512,
        )
        assert cfg.spec_dflash_cross_ctx == 512


class TestDFlashDraftNgl:
    """spec_draft_ngl accepts int or str values."""

    def test_dflash_draft_ngl_as_int(self) -> None:
        """spec_draft_ngl accepts an integer."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
            spec_draft_ngl=32,
        )
        assert cfg.spec_draft_ngl == 32

    def test_dflash_draft_ngl_as_string_all(self) -> None:
        """spec_draft_ngl accepts 'all' string."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
            spec_draft_ngl="all",
        )
        assert cfg.spec_draft_ngl == "all"

    def test_dflash_draft_ngl_default_empty(self) -> None:
        """spec_draft_ngl defaults to empty string."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
        )
        assert cfg.spec_draft_ngl == ""


class TestNgramModPreserved:
    """Existing ngram-mod behavior is preserved."""

    def test_ngram_mod_basic(self) -> None:
        """ngram-mod spec_type with parameters is accepted."""
        cfg = SpeculativeDecodingConfig(
            spec_type="ngram-mod",
            spec_ngram_size_n=2,
            draft_min=1,
            draft_max=5,
        )
        assert cfg.spec_type == "ngram-mod"
        assert cfg.spec_ngram_size_n == 2
        assert cfg.draft_min == 1
        assert cfg.draft_max == 5

    def test_ngram_mod_draft_min_exceeds_draft_max(self) -> None:
        """draft_min > draft_max raises ValueError."""
        with pytest.raises(ValueError, match="draft_min must be <= draft_max"):
            SpeculativeDecodingConfig(
                spec_type="ngram-mod",
                draft_min=10,
                draft_max=5,
            )

    def test_ngram_mod_negative_ngram_size(self) -> None:
        """Negative spec_ngram_size_n raises ValueError."""
        with pytest.raises(ValueError, match="spec_ngram_size_n must be non-negative"):
            SpeculativeDecodingConfig(
                spec_type="ngram-mod",
                spec_ngram_size_n=-1,
            )


class TestDraftMtpPreserved:
    """Existing draft-mtp behavior is preserved."""

    def test_draft_mtp_basic(self) -> None:
        """draft-mtp spec_type with parameters is accepted."""
        cfg = SpeculativeDecodingConfig(
            spec_type="draft-mtp",
            spec_draft_n_max=4,
        )
        assert cfg.spec_type == "draft-mtp"
        assert cfg.spec_draft_n_max == 4

    def test_draft_mtp_negative_n_max(self) -> None:
        """Negative spec_draft_n_max raises ValueError."""
        with pytest.raises(ValueError, match="spec_draft_n_max must be non-negative"):
            SpeculativeDecodingConfig(
                spec_type="draft-mtp",
                spec_draft_n_max=-1,
            )

    def test_draft_mtp_p_min_out_of_range(self) -> None:
        """spec_draft_p_min outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="spec_draft_p_min"):
            SpeculativeDecodingConfig(
                spec_type="draft-mtp",
                spec_draft_p_min=1.5,
            )


class TestReasoningFields:
    """Reasoning mode/format fields are preserved."""

    def test_reasoning_mode_and_format(self) -> None:
        """Reasoning fields are stored correctly."""
        cfg = SpeculativeDecodingConfig(
            reasoning_mode="on",
            reasoning_format="deepseek",
        )
        assert cfg.reasoning_mode == "on"
        assert cfg.reasoning_format == "deepseek"

    def test_reasoning_defaults(self) -> None:
        """Reasoning defaults to auto/none."""
        cfg = SpeculativeDecodingConfig()
        assert cfg.reasoning_mode == "auto"
        assert cfg.reasoning_format == "none"


class TestSpeculativeDecodingConfigDictBehavior:
    """SpeculativeDecodingConfig acts as a dict after __post_init__."""

    def test_dict_contains_all_fields(self) -> None:
        """Config dict representation contains all field keys."""
        cfg = SpeculativeDecodingConfig(
            spec_type="dflash",
            spec_draft_model="/path/to/model",
            spec_dflash_cross_ctx=512,
        )
        assert "spec_type" in cfg
        assert "spec_draft_model" in cfg
        assert "spec_dflash_cross_ctx" in cfg
        assert cfg["spec_type"] == "dflash"
        assert cfg["spec_draft_model"] == "/path/to/model"
        assert cfg["spec_dflash_cross_ctx"] == 512

    def test_dict_reflects_dataclass_attrs(self) -> None:
        """Dict values match dataclass attribute values."""
        cfg = SpeculativeDecodingConfig(
            spec_type="ngram-mod",
            spec_ngram_size_n=10,
            draft_min=2,
            draft_max=8,
        )
        assert cfg["spec_type"] == cfg.spec_type
        assert cfg["spec_ngram_size_n"] == cfg.spec_ngram_size_n
        assert cfg["draft_min"] == cfg.draft_min
        assert cfg["draft_max"] == cfg.draft_max
