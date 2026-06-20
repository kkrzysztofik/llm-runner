"""Slot runtime stats: parse /metrics or /slots, persist to JSON, collect via HTTP.

This module is a pure library — no I/O at module level.
All functions take typed parameters and return values or mutate state explicitly.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from llama_manager.common.file_ops import atomic_write_json
from llama_manager.orchestration.lockfile import resolve_runtime_dir

_METRIC_PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
_METRIC_PROMPT_TPS = "llamacpp:prompt_tokens_seconds"
_METRIC_PREDICTED_TOKENS = "llamacpp:tokens_predicted_total"
_METRIC_PREDICTED_TPS = "llamacpp:predicted_tokens_seconds"


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


def _float_text(value: str) -> float | None:
    """Return *value* parsed as float, or ``None`` if invalid."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _metric_name(token: str) -> str:
    """Return Prometheus sample name without labels."""
    return token.split("{", 1)[0]


def _parse_prometheus_samples(payload: str) -> dict[str, list[float]]:
    """Parse simple Prometheus text samples by metric name."""
    samples: dict[str, list[float]] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        value = _float_text(parts[1])
        if value is None:
            continue
        samples.setdefault(_metric_name(parts[0]), []).append(value)
    return samples


def _sum_metric(samples: dict[str, list[float]], name: str) -> float | None:
    values = samples.get(name)
    if not values:
        return None
    return sum(values)


def _first_metric(samples: dict[str, list[float]], name: str) -> float | None:
    values = samples.get(name)
    if not values:
        return None
    return values[0]


def parse_metrics_payload(
    alias: str,
    port: int,
    payload: str,
    *,
    now: float | None = None,
) -> SlotStatsSnapshot | None:
    """Parse llama.cpp Prometheus ``GET /metrics`` text into runtime counters."""
    samples = _parse_prometheus_samples(payload)
    prompt_tokens = _sum_metric(samples, _METRIC_PROMPT_TOKENS)
    predicted_tokens = _sum_metric(samples, _METRIC_PREDICTED_TOKENS)
    prompt_tps = _first_metric(samples, _METRIC_PROMPT_TPS)
    predicted_tps = _first_metric(samples, _METRIC_PREDICTED_TPS)

    if (
        prompt_tokens is None
        and predicted_tokens is None
        and prompt_tps is None
        and predicted_tps is None
    ):
        return None

    return SlotStatsSnapshot(
        alias=alias,
        port=port,
        updated_at=now or time.time(),
        tps=predicted_tps,
        prompt_tps=prompt_tps,
        tokens_in=_int_value(prompt_tokens),
        tokens_out=_int_value(predicted_tokens),
        source="metrics",
    )


def _sum_tokens_out(slot: dict) -> int:
    """Sum decoded tokens from a single slot's next_token field."""
    next_token = slot.get("next_token")
    if isinstance(next_token, dict):
        return _int_value(next_token.get("n_decoded"))
    return 0


def _best_tokens_in(slot: dict) -> int:
    """Find the best tokens_in value for a single slot across all key paths."""
    top_in = _int_value(slot.get("n_prompt_tokens_processed"))
    if top_in == 0:
        top_in = _int_value(slot.get("n_prompt_tokens"))
    prompt_data = slot.get("prompt")
    if isinstance(prompt_data, dict):
        nested_in = _int_value(prompt_data.get("n_tokens_processed"))
        if nested_in == 0:
            nested_in = _int_value(prompt_data.get("n_tokens"))
        if nested_in > top_in:
            top_in = nested_in
    return top_in


def _slot_tps(slot: dict, next_token: Any) -> float | None:
    """First available generation-rate from a slot."""
    tps_value = _first_number(
        slot,
        [
            "next_token.tps",
            "tokens_per_second",
            "generation_tps",
            "tps",
        ],
    )
    if tps_value is None and isinstance(next_token, dict):
        tps_value = _number(next_token.get("tps"))
    return tps_value


def _slot_prompt_tps(slot: dict) -> float | None:
    """First available prompt-rate from a slot."""
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
    return prompt_tps_value


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

        next_token = slot.get("next_token")
        total_tokens_out += _sum_tokens_out(slot)
        total_tokens_in += _best_tokens_in(slot)

        if tps_value is None:
            tps_value = _slot_tps(slot, next_token)

        if prompt_tps_value is None:
            prompt_tps_value = _slot_prompt_tps(slot)

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


def profile_stats_file_path(runtime_dir: Path | None = None) -> Path:
    """Return the path to the persisted profile aggregate stats JSON file."""
    base = runtime_dir or resolve_runtime_dir()
    return base / "profile-stats.json"


def load_slot_stats(runtime_dir: Path | None = None) -> dict[str, SlotStatsSnapshot]:
    """Load persisted slot stats from JSON.

    Returns empty dict on missing/invalid file.
    Skips individual invalid slot entries.
    """
    path = slot_stats_file_path(runtime_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
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


def load_profile_stats(runtime_dir: Path | None = None) -> dict[str, ProfileStatsAggregate]:
    """Load profile aggregate stats from JSON.

    Returns empty dict on missing/invalid file. Session counters are loaded into
    each aggregate so future updates can compute positive deltas.
    """
    path = profile_stats_file_path(runtime_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError, ValueError:
        return {}

    if not isinstance(data, dict):
        return {}

    profiles_data = data.get("profiles")
    sessions_data = data.get("sessions")
    if not isinstance(profiles_data, dict):
        return {}
    if not isinstance(sessions_data, dict):
        sessions_data = {}

    result: dict[str, ProfileStatsAggregate] = {}
    for profile_id, entry in profiles_data.items():
        if not isinstance(profile_id, str) or not isinstance(entry, dict):
            continue
        try:
            aggregate = ProfileStatsAggregate(
                profile_id=entry.get("profile_id", profile_id),
                updated_at=float(entry.get("updated_at", 0.0)),
                tokens_in=_int_value(entry.get("tokens_in")),
                tokens_out=_int_value(entry.get("tokens_out")),
                sessions_count=_int_value(entry.get("sessions_count")),
            )
        except TypeError, ValueError:
            continue

        aggregate.sessions = _load_profile_sessions(sessions_data.get(profile_id))
        result[profile_id] = aggregate

    return result


def save_profile_stats(
    stats_by_profile: dict[str, ProfileStatsAggregate],
    runtime_dir: Path | None = None,
) -> None:
    """Save profile aggregate stats to JSON using atomic_write_json."""
    path = profile_stats_file_path(runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    profiles_data: dict[str, dict[str, Any]] = {}
    sessions_data: dict[str, dict[str, dict[str, Any]]] = {}
    for profile_id, stats in stats_by_profile.items():
        profiles_data[profile_id] = {
            "profile_id": stats.profile_id,
            "updated_at": stats.updated_at,
            "tokens_in": max(0, stats.tokens_in),
            "tokens_out": max(0, stats.tokens_out),
            "sessions_count": max(0, stats.sessions_count),
        }
        sessions_data[profile_id] = {
            session_id: session.to_json() for session_id, session in stats.sessions.items()
        }

    atomic_write_json(
        path,
        {
            "version": 1,
            "profiles": profiles_data,
            "sessions": sessions_data,
        },
    )


def update_profile_stats(
    stats_by_profile: dict[str, ProfileStatsAggregate],
    profile_id: str,
    session_id: str,
    snapshot: SlotStatsSnapshot,
) -> dict[str, ProfileStatsAggregate]:
    """Update aggregate stats from one successful per-slot snapshot.

    Only positive counter deltas are added. New sessions and counter resets
    update the session baseline without adding historical or negative values.
    """
    if not profile_id or not session_id:
        return stats_by_profile

    updated = dict(stats_by_profile)
    aggregate = updated.get(profile_id)
    if aggregate is None:
        aggregate = ProfileStatsAggregate(profile_id=profile_id)
        updated[profile_id] = aggregate

    aggregate.apply_snapshot(session_id, snapshot)
    return updated


def _load_profile_sessions(data: Any) -> dict[str, ProfileStatsSession]:
    if not isinstance(data, dict):
        return {}
    result: dict[str, ProfileStatsSession] = {}
    for session_id, entry in data.items():
        if not isinstance(session_id, str) or not isinstance(entry, dict):
            continue
        try:
            result[session_id] = ProfileStatsSession(
                session_id=entry.get("session_id", session_id),
                updated_at=float(entry.get("updated_at", 0.0)),
                last_tokens_in=_int_value(entry.get("last_tokens_in")),
                last_tokens_out=_int_value(entry.get("last_tokens_out")),
            )
        except TypeError, ValueError:
            continue
    return result


def collect_slot_stats(
    alias: str,
    host: str,
    port: int,
    *,
    timeout_s: float = 0.2,
    now: float | None = None,
) -> SlotStatsSnapshot | None:
    """Fetch and parse runtime stats from a running llama-server instance.

    Returns ``None`` for any HTTP/connection/parse failure.
    """
    metrics_payload = _http_get_text(host, port, "/metrics", timeout_s)
    if metrics_payload is not None:
        stats = parse_metrics_payload(alias, port, metrics_payload, now=now)
        if stats is not None:
            return stats

    slots_payload = _http_get_text(host, port, "/slots", timeout_s)
    if slots_payload is None:
        return None

    try:
        payload = json.loads(slots_payload)
    except json.JSONDecodeError, ValueError:
        return None

    if not isinstance(payload, list):
        return None

    return parse_slots_payload(alias, port, payload, now=now)


def _http_get_text(host: str, port: int, path: str, timeout_s: float) -> str | None:
    """Fetch one localhost llama-server endpoint and return decoded text."""
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}{path}"  # noqa: S310
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except OSError, TimeoutError, urllib.error.URLError:
        return None


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


class ProfileStatsSession:
    """Last observed per-server counters for one profile session."""

    __slots__ = ("session_id", "updated_at", "last_tokens_in", "last_tokens_out")

    def __init__(
        self,
        session_id: str,
        updated_at: float = 0.0,
        last_tokens_in: int = 0,
        last_tokens_out: int = 0,
    ) -> None:
        self.session_id = session_id
        self.updated_at = updated_at
        self.last_tokens_in = max(0, last_tokens_in)
        self.last_tokens_out = max(0, last_tokens_out)

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable session baseline data."""
        return {
            "session_id": self.session_id,
            "updated_at": self.updated_at,
            "last_tokens_in": max(0, self.last_tokens_in),
            "last_tokens_out": max(0, self.last_tokens_out),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProfileStatsSession):
            return False
        return (
            self.session_id == other.session_id
            and self.updated_at == other.updated_at
            and self.last_tokens_in == other.last_tokens_in
            and self.last_tokens_out == other.last_tokens_out
        )


class ProfileStatsAggregate:
    """Persisted aggregate token counters for one run profile."""

    __slots__ = (
        "profile_id",
        "updated_at",
        "tokens_in",
        "tokens_out",
        "sessions_count",
        "sessions",
    )

    def __init__(
        self,
        profile_id: str,
        updated_at: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        sessions_count: int = 0,
        sessions: dict[str, ProfileStatsSession] | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.updated_at = updated_at
        self.tokens_in = max(0, tokens_in)
        self.tokens_out = max(0, tokens_out)
        self.sessions_count = max(0, sessions_count)
        self.sessions = dict(sessions or {})

    def apply_snapshot(self, session_id: str, snapshot: SlotStatsSnapshot) -> None:
        """Accumulate positive counter deltas from *snapshot*."""
        current_in = max(0, snapshot.tokens_in)
        current_out = max(0, snapshot.tokens_out)
        session = self.sessions.get(session_id)
        if session is None:
            self.sessions[session_id] = ProfileStatsSession(
                session_id=session_id,
                updated_at=snapshot.updated_at,
                last_tokens_in=current_in,
                last_tokens_out=current_out,
            )
            self.sessions_count += 1
            self.updated_at = snapshot.updated_at
            return

        delta_in = current_in - session.last_tokens_in
        delta_out = current_out - session.last_tokens_out
        if delta_in > 0:
            self.tokens_in += delta_in
        if delta_out > 0:
            self.tokens_out += delta_out

        session.last_tokens_in = current_in
        session.last_tokens_out = current_out
        session.updated_at = snapshot.updated_at
        self.updated_at = snapshot.updated_at

    def to_display(self) -> dict[str, str]:
        """Convert aggregate counters to display-safe strings."""
        return {
            "tokens_in": str(max(0, self.tokens_in)),
            "tokens_out": str(max(0, self.tokens_out)),
            "sessions": str(max(0, self.sessions_count)),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProfileStatsAggregate):
            return False
        return (
            self.profile_id == other.profile_id
            and self.updated_at == other.updated_at
            and self.tokens_in == other.tokens_in
            and self.tokens_out == other.tokens_out
            and self.sessions_count == other.sessions_count
            and self.sessions == other.sessions
        )
