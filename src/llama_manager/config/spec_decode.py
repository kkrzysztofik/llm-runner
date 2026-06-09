"""Speculative decoding configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass
class SpeculativeDecodingConfig(dict[str, object]):
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
        self.clear()
        self.update(
            {
                "spec_type": self.spec_type,
                "spec_ngram_size_n": self.spec_ngram_size_n,
                "draft_min": self.draft_min,
                "draft_max": self.draft_max,
                "spec_draft_n_max": self.spec_draft_n_max,
                "spec_draft_p_min": self.spec_draft_p_min,
                "spec_draft_cache_type_k": self.spec_draft_cache_type_k,
                "spec_draft_cache_type_v": self.spec_draft_cache_type_v,
                "spec_draft_device": self.spec_draft_device,
                "reasoning_mode": self.reasoning_mode,
                "reasoning_format": self.reasoning_format,
                "reasoning_budget": self.reasoning_budget,
                "spec_draft_model": self.spec_draft_model,
                "spec_draft_hf": self.spec_draft_hf,
                "spec_draft_ngl": self.spec_draft_ngl,
                "spec_dflash_cross_ctx": self.spec_dflash_cross_ctx,
            }
        )


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
    if config.spec_type not in ("", "ngram-mod", "draft-mtp", "dflash"):
        raise ValueError("spec_type must be '', 'ngram-mod', 'draft-mtp', or 'dflash'")
    if config.spec_type == "dflash":
        _validate_dflash_config(config)
    if config.spec_dflash_cross_ctx < 0:
        raise ValueError("spec_dflash_cross_ctx must be non-negative")


def _validate_dflash_config(config: SpeculativeDecodingConfig) -> None:
    if not config.spec_draft_model and not config.spec_draft_hf:
        raise ValueError("spec_draft_model or spec_draft_hf required when spec_type is 'dflash'")
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


class SpeculativeDecodingFieldsMixin:
    """Expose nested spec-decoding fields as direct config attributes."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> object:
        if name in SPECULATIVE_DECODING_FIELD_NAMES:
            return getattr(self._spec_decode(), name)
        return object.__getattribute__(self, name)

    def _spec_decode(self) -> SpeculativeDecodingConfig:
        return cast(SpeculativeDecodingConfig, object.__getattribute__(self, "spec_decode"))

    @property
    def reasoning_mode(self) -> str:
        return self._spec_decode().reasoning_mode

    @property
    def reasoning_format(self) -> str:
        return self._spec_decode().reasoning_format

    @property
    def reasoning_budget(self) -> str:
        return self._spec_decode().reasoning_budget

    @property
    def spec_type(self) -> str:
        return self._spec_decode().spec_type

    @property
    def spec_ngram_size_n(self) -> int:
        return self._spec_decode().spec_ngram_size_n

    @property
    def draft_min(self) -> int:
        return self._spec_decode().draft_min

    @property
    def draft_max(self) -> int:
        return self._spec_decode().draft_max

    @property
    def spec_draft_n_max(self) -> int:
        return self._spec_decode().spec_draft_n_max

    @property
    def spec_draft_p_min(self) -> float:
        return self._spec_decode().spec_draft_p_min

    @property
    def spec_draft_cache_type_k(self) -> str:
        return self._spec_decode().spec_draft_cache_type_k

    @property
    def spec_draft_cache_type_v(self) -> str:
        return self._spec_decode().spec_draft_cache_type_v

    @property
    def spec_draft_device(self) -> str:
        return self._spec_decode().spec_draft_device

    @property
    def spec_draft_model(self) -> str:
        return self._spec_decode().spec_draft_model

    @property
    def spec_draft_hf(self) -> str:
        return self._spec_decode().spec_draft_hf

    @property
    def spec_draft_ngl(self) -> int | str:
        return self._spec_decode().spec_draft_ngl

    @property
    def spec_dflash_cross_ctx(self) -> int:
        return self._spec_decode().spec_dflash_cross_ctx
