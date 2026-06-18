"""Slot runtime stats: parse /slots, persist to JSON, collect via HTTP.

This module is a pure library — no I/O at module level.
All functions take typed parameters and return values or mutate state explicitly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llama_manager.common.file_ops import atomic_write_json
from llama_manager.orchestration.lockfile import resolve_runtime_dir


def _number(value: Any) -> float | None:
    """Return *value* as float, or ``None`` if not a valid numeric type."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _int_value(value: Any) -> int:
    """Return *value* as int, defaulting to ``0`` for invalid types."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _nested(data: dict, *keys: str) -> Any:
    """Walk nested dicts via *keys*, returning ``None`` on missing keys."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _first_number(data: dict, candidates: list[str]) -> float | None:
    """Return the first numeric value found among *candidates* in *data*."""
    for key in candidates:
        if key in data:
            val = _number(data[key])
            if val is not None:
                return val
        nested_val = _nested(data, key)
        if isinstance(nested_val, dict):
            for sub_key in ("value", "n", "count", "rate"):
                val = _number(nested_val.get(sub_key))
                if val is not None:
                    return val
    return None


def _format_rate(value: float | None) -> str:
    """Format a rate value for display, returning ``--`` for None/negative."""
    if value is None or value < 0:
        return "--"
    return f"{value:.1f}"


def parse_slots_payload(
    alias: str,
    port: int,
    payload: Any,
    *,
    now: float | None = None,
) -> SlotStatsSnapshot:
    """Parse a ``GET /slots`` JSON response into a ``SlotStatsSnapshot``.

    Accepts ``list[dict[str, object]]``; ignores anything else.
    Sums across all returned llama.cpp internal slots for one server instance.
    """
    if not isinstance(payload, list):
        return SlotStatsSnapshot(
            alias=alias,
            port=port,
            updated_at=now or time.time(),
        )

    total_tokens_out = 0
    total_tokens_in = 0
    tps_value: float | None = None
    prompt_tps_value: float | None = None

    for slot in payload:
        if not isinstance(slot, dict):
            continue

        # tokens_out: sum next_token.n_decoded
        next_token = slot.get("next_token")
        if isinstance(next_token, dict):
            total_tokens_out += _int_value(next_token.get("n_decoded"))

        # tokens_in: best available prompt/processed token fields
        # top-level keys
        top_in = _int_value(slot.get("n_prompt_tokens_processed"))
        if top_in == 0:
            top_in = _int_value(slot.get("n_prompt_tokens"))
        # nested keys
        prompt_data = slot.get("prompt")
        if isinstance(prompt_data, dict):
            nested_in = _int_value(prompt_data.get("n_tokens_processed"))
            if nested_in == 0:
                nested_in = _int_value(prompt_data.get("n_tokens"))
            if nested_in > top_in:
                top_in = nested_in
        total_tokens_in += top_in

        # tps: first available numeric generation-rate field
        if tps_value is None:
            tps_value = _first_number(
                slot,
                [
                    "next_token.tps",
                    "tokens_per_second",
                    "generation_tps",
                    "tps",
                ],
            )
            # Also check nested next_token.tps
            if tps_value is None and isinstance(next_token, dict):
                tps_value = _number(next_token.get("tps"))

        # prompt_tps: first available numeric prompt-rate field
        if prompt_tps_value is None:
            prompt_tps_value = _first_number(
                slot,
                [
                    "prompt_tps",
                    "prompt_tokens_per_second",
                ],
            )
            nested_prompt_tps = _nested(slot, "prompt", "tokens_per_second")
            if nested_prompt_tps is not None:
                val = _number(nested_prompt_tps)
                if val is not None:
                    prompt_tps_value = val

    return SlotStatsSnapshot(
        alias=alias,
        port=port,
        updated_at=now or time.time(),
        tps=tps_value,
        prompt_tps=prompt_tps_value,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
    )


def slot_stats_file_path(runtime_dir: Path | None = None) -> Path:
    """Return the path to the slot-stats JSON file."""
    base = runtime_dir or resolve_runtime_dir()
    return base / "slot-stats.json"


def load_slot_stats(runtime_dir: Path | None = None) -> dict[str, SlotStatsSnapshot]:
    """Load persisted slot stats from JSON.

    Returns empty dict on missing/invalid file.
    Skips individual invalid slot entries.
    """
    path = slot_stats_file_path(runtime_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError, FileNotFoundError:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError, ValueError:
        return {}

    if not isinstance(data, dict):
        return {}

    slots_data = data.get("slots")
    if not isinstance(slots_data, dict):
        return {}

    result: dict[str, SlotStatsSnapshot] = {}
    for alias, entry in slots_data.items():
        if not isinstance(entry, dict):
            continue
        try:
            snapshot = SlotStatsSnapshot(
                alias=entry.get("alias", alias),
                port=entry.get("port", 0),
                updated_at=entry.get("updated_at", 0.0),
                tps=entry.get("tps"),
                prompt_tps=entry.get("prompt_tps"),
                tokens_in=_int_value(entry.get("tokens_in")),
                tokens_out=_int_value(entry.get("tokens_out")),
                source=entry.get("source", "slots"),
            )
            result[alias] = snapshot
        except TypeError, ValueError:
            continue

    return result


def save_slot_stats(
    stats_by_alias: dict[str, SlotStatsSnapshot],
    runtime_dir: Path | None = None,
) -> None:
    """Save slot stats to JSON using atomic_write_json."""
    path = slot_stats_file_path(runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    slots_data: dict[str, dict[str, Any]] = {}
    for alias, stats in stats_by_alias.items():
        entry: dict[str, Any] = {
            "alias": stats.alias,
            "port": stats.port,
            "updated_at": stats.updated_at,
            "source": stats.source,
        }
        if stats.tps is not None:
            entry["tps"] = stats.tps
        if stats.prompt_tps is not None:
            entry["prompt_tps"] = stats.prompt_tps
        entry["tokens_in"] = max(0, stats.tokens_in)
        entry["tokens_out"] = max(0, stats.tokens_out)
        slots_data[alias] = entry

    data = {
        "version": 1,
        "slots": slots_data,
    }
    atomic_write_json(path, data)


def collect_slot_stats(
    alias: str,
    host: str,
    port: int,
    *,
    timeout_s: float = 0.2,
    now: float | None = None,
) -> SlotStatsSnapshot | None:
    """Fetch and parse ``GET /slots`` from a running llama-server instance.

    Returns ``None`` for any HTTP/connection/parse failure.
    """
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/slots"  # noqa: S310
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except OSError, TimeoutError, urllib.error.URLError:
        return None
    except Exception:
        return None

    try:
        payload = json.loads(body)
    except json.JSONDecodeError, ValueError:
        return None

    if not isinstance(payload, list):
        return None

    return parse_slots_payload(alias, port, payload, now=now)


class SlotStatsSnapshot:
    """Immutable snapshot of per-slot runtime statistics.

    Attributes:
        alias: Server/profile alias.
        port: Server port.
        updated_at: Unix timestamp of when stats were collected.
        tps: Generation throughput (tokens/s), or None if unavailable.
        prompt_tps: Prompt processing throughput (tokens/s), or None.
        tokens_in: Cumulative prompt/input tokens.
        tokens_out: Cumulative generated/output tokens.
        source: Source of the stats (default "slots").
    """

    __slots__ = (
        "alias",
        "port",
        "updated_at",
        "tps",
        "prompt_tps",
        "tokens_in",
        "tokens_out",
        "source",
    )

    def __init__(
        self,
        alias: str,
        port: int,
        updated_at: float,
        tps: float | None = None,
        prompt_tps: float | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        source: str = "slots",
    ) -> None:
        self.alias = alias
        self.port = port
        self.updated_at = updated_at
        self.tps = tps
        self.prompt_tps = prompt_tps
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.source = source

    def to_display(self) -> dict[str, str]:
        """Convert to display-safe string values."""
        return {
            "tps": _format_rate(self.tps),
            "pp": _format_rate(self.prompt_tps),
            "tokens_in": str(max(0, self.tokens_in)),
            "tokens_out": str(max(0, self.tokens_out)),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlotStatsSnapshot):
            return False
        return (
            self.alias == other.alias
            and self.port == other.port
            and self.updated_at == other.updated_at
            and self.tps == other.tps
            and self.prompt_tps == other.prompt_tps
            and self.tokens_in == other.tokens_in
            and self.tokens_out == other.tokens_out
            and self.source == other.source
        )

    def __repr__(self) -> str:
        return (
            f"SlotStatsSnapshot(alias={self.alias!r}, port={self.port}, "
            f"tps={self.tps}, prompt_tps={self.prompt_tps}, "
            f"tokens_in={self.tokens_in}, tokens_out={self.tokens_out})"
        )
