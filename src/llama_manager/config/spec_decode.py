"""Speculative decoding configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_VALID_SPEC_TYPES: frozenset[str] = frozenset({"ngram-mod", "draft-mtp", "draft-dflash"})


def spec_type_members(spec_type: str) -> list[str]:
    """Split a comma-separated ``--spec-type`` value into trimmed members.

    Empty components (e.g. ``draft-mtp,,ngram-mod`` or a trailing comma) raise
    ``ValueError`` rather than being silently dropped.
    """
    if not spec_type:
        return []
    members: list[str] = []
    for part in spec_type.split(","):
        stripped = part.strip()
        if not stripped:
            raise ValueError("spec_type must not contain empty comma-separated components")
        members.append(stripped)
    return members


@dataclass
class SpeculativeDecodingConfig:
    """llama-server speculative decoding and reasoning options."""

    spec_type: str = ""
    spec_ngram_size_n: int = 0
    draft_min: int = 0
    draft_max: int = 0
    spec_draft_n_max: int = 0
    spec_draft_p_min: float = 0.0
    spec_draft_cache_type_k: str = ""
    spec_draft_cache_type_v: str = ""
    spec_draft_device: str = ""
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    reasoning_budget: str = ""
    spec_draft_model: str = ""
    spec_draft_hf: str = ""
    spec_draft_ngl: int | str = ""
    spec_dflash_cross_ctx: int = 0

    def __post_init__(self) -> None:
        _validate_speculative_decoding(self)
        # Normalize once here so every consumer (argv builder, profile store,
        # profile IO, TUI prefill) sees the same canonical comma-joined form.
        # Sorting and deduping is safe: llama.cpp reduces --spec-type to a bitmask
        # (common_get_enabled_speculative_configs) and then applies its own hardcoded
        # speculator priority, so member order and repeats carry no meaning.
        self.spec_type = ",".join(dict.fromkeys(sorted(spec_type_members(self.spec_type))))


def _validate_speculative_decoding(config: SpeculativeDecodingConfig) -> None:
    if config.spec_ngram_size_n < 0:
        raise ValueError("spec_ngram_size_n must be non-negative")
    if config.draft_min < 0:
        raise ValueError("draft_min must be non-negative")
    if config.draft_max < 0:
        raise ValueError("draft_max must be non-negative")
    if config.draft_min > config.draft_max:
        raise ValueError("draft_min must be <= draft_max")
    if config.spec_draft_n_max < 0:
        raise ValueError("spec_draft_n_max must be non-negative")
    if config.spec_draft_p_min < 0.0:
        raise ValueError("spec_draft_p_min must be non-negative")
    if config.spec_draft_p_min > 1.0:
        raise ValueError("spec_draft_p_min must be <= 1.0")
    _validate_spec_type(config)
    if config.spec_dflash_cross_ctx < 0:
        raise ValueError("spec_dflash_cross_ctx must be non-negative")


def _validate_spec_type(config: SpeculativeDecodingConfig) -> None:
    """Validate every member of the comma-separated ``spec_type`` value."""
    if not config.spec_type:
        return
    members = spec_type_members(config.spec_type)
    known = ", ".join(sorted(_VALID_SPEC_TYPES))
    if not members:
        raise ValueError(f"spec_type must be '' or a comma-separated list of: {known}")
    for member in members:
        if member not in _VALID_SPEC_TYPES:
            raise ValueError(f"spec_type member '{member}' is unknown; valid members: {known}")
    if "draft-dflash" in members:
        _validate_dflash_config(config)


def _validate_dflash_config(config: SpeculativeDecodingConfig) -> None:
    if not config.spec_draft_model and not config.spec_draft_hf:
        raise ValueError(
            "spec_draft_model or spec_draft_hf required when spec_type is 'draft-dflash'"
        )
    if config.spec_draft_model and config.spec_draft_hf:
        raise ValueError("spec_draft_model and spec_draft_hf are mutually exclusive")


SPECULATIVE_DECODING_FIELD_NAMES = frozenset(SpeculativeDecodingConfig.__dataclass_fields__)


def resolve_speculative_decoding_config(
    spec_decode: SpeculativeDecodingConfig | None,
    values: Mapping[str, Any],
) -> SpeculativeDecodingConfig:
    """Build spec-decoding config from an optional base and constructor values."""
    resolved = spec_decode or SpeculativeDecodingConfig()
    active_overrides: dict[str, Any] = {
        key: value
        for key in SPECULATIVE_DECODING_FIELD_NAMES
        if (value := values.get(key)) is not None
    }
    if not active_overrides:
        return resolved
    resolved_values: dict[str, Any] = dict(resolved.__dict__)
    resolved_values.update(active_overrides)
    return SpeculativeDecodingConfig(**resolved_values)
