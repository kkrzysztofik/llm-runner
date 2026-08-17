"""Shared flat launch/runtime option fields for profiles, servers, and defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, cast


@dataclass
class LaunchRuntimeFields:
    """Canonical launch/runtime option fields (kw-only for safe dataclass inheritance)."""

    kv_unified: bool = field(default=False, kw_only=True)
    mmproj_offload: bool = field(default=True, kw_only=True)
    load_mode: str = field(default="auto", kw_only=True)
    split_mode: str = field(default="layer", kw_only=True)
    no_host_buffer: bool = field(default=False, kw_only=True)
    ui: bool = field(default=False, kw_only=True)
    reasoning_preserve: str = field(default="auto", kw_only=True)
    reasoning_budget_message: str = field(default="", kw_only=True)
    reasoning_effort: str = field(default="medium", kw_only=True)
    fit: str = field(default="auto", kw_only=True)
    ctx_checkpoints: int | None = field(default=None, kw_only=True)
    temperature: float | None = field(default=None, kw_only=True)
    top_k: int | None = field(default=None, kw_only=True)
    top_p: float | None = field(default=None, kw_only=True)
    min_p: float | None = field(default=None, kw_only=True)
    presence_penalty: float | None = field(default=None, kw_only=True)
    repeat_penalty: float | None = field(default=None, kw_only=True)


# Backward-compatible alias used by earlier refactor commits / imports.
LaunchRuntimeFieldsMixin = LaunchRuntimeFields

LAUNCH_RUNTIME_KEYS: frozenset[str] = frozenset(f.name for f in fields(LaunchRuntimeFields))

_LAUNCH_RUNTIME_DEFAULTS = LaunchRuntimeFields()


class LaunchRuntimeAttributeMixin:
    """Flat attribute access to a nested ``launch_runtime`` dataclass field.

    Used by frozen/slotted models that cannot inherit the non-frozen
    ``LaunchRuntimeFields`` dataclass base.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> object:
        if name in LAUNCH_RUNTIME_KEYS:
            return getattr(object.__getattribute__(self, "launch_runtime"), name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        if name in LAUNCH_RUNTIME_KEYS:
            current = object.__getattribute__(self, "launch_runtime")
            object.__setattr__(self, "launch_runtime", replace(current, **{name: value}))
            return
        object.__setattr__(self, name, value)


def _resolve_launch_runtime_values(
    overrides: Mapping[str, object],
    *,
    fill_missing: bool,
) -> dict[str, object]:
    """Build a complete launch-runtime value dict from overrides."""
    values: dict[str, object] = {}
    for key in LAUNCH_RUNTIME_KEYS:
        default = getattr(_LAUNCH_RUNTIME_DEFAULTS, key)
        if key in overrides:
            value = overrides[key]
            if value is None:
                if not fill_missing:
                    continue
                value = default
            values[key] = value
        elif fill_missing:
            values[key] = default
    return values


def apply_launch_runtime(
    target: object,
    overrides: Mapping[str, object],
    *,
    frozen: bool = False,
    nested: bool = False,
) -> None:
    """Apply launch/runtime overrides onto ``target``.

    When ``nested`` is true, values are written to ``target.launch_runtime``.
    When ``frozen`` is true, missing/None overrides use defaults and assignment
    goes through ``object.__setattr__``.
    """
    values = _resolve_launch_runtime_values(overrides, fill_missing=frozen or nested)
    if nested:
        config = LaunchRuntimeFields(**cast(Any, values))
        object.__setattr__(target, "launch_runtime", config)
        return
    for key, value in values.items():
        if frozen:
            object.__setattr__(target, key, value)
        else:
            setattr(target, key, value)


def split_launch_runtime_kwargs(
    kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split ``**kwargs`` into launch-runtime overrides and remaining kwargs."""
    runtime: dict[str, object] = {}
    rest: dict[str, object] = {}
    for key, value in kwargs.items():
        if key in LAUNCH_RUNTIME_KEYS:
            runtime[key] = value
        else:
            rest[key] = value
    return runtime, rest
