"""Speculative decoding configuration."""

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        if self.spec_ngram_size_n < 0:
            raise ValueError("spec_ngram_size_n must be non-negative")
        if self.draft_min < 0:
            raise ValueError("draft_min must be non-negative")
        if self.draft_max < 0:
            raise ValueError("draft_max must be non-negative")
        if self.draft_min > self.draft_max:
            raise ValueError("draft_min must be <= draft_max")
        if self.spec_draft_n_max < 0:
            raise ValueError("spec_draft_n_max must be non-negative")
        if self.spec_draft_p_min < 0.0:
            raise ValueError("spec_draft_p_min must be non-negative")
        if self.spec_draft_p_min > 1.0:
            raise ValueError("spec_draft_p_min must be <= 1.0")
        if self.spec_type not in ("", "ngram-mod", "draft-mtp"):
            raise ValueError("spec_type must be '', 'ngram-mod', or 'draft-mtp'")
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
            }
        )
