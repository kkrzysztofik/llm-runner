"""System statistics collection via psutil.

Provides CPU, memory, swap, uptime, and task/thread counts for
dashboard display.  All functions are pure library helpers — no I/O
except psutil calls.
"""

import time

import psutil


def collect_cpu_percentages(percpu: bool = True) -> list[float]:
    """Return current per-core CPU usage percentages.

    Args:
        percpu: If True, return one value per CPU core.

    Returns:
        List of CPU usage percentages (0.0–100.0).
    """
    samples = psutil.cpu_percent(interval=None, percpu=percpu)
    if isinstance(samples, list):
        return [float(s) for s in samples]
    return [float(samples)]


def collect_memory_usage() -> dict[str, dict[str, object]]:
    """Return memory and swap usage snapshots.

    Returns:
        Dict with ``"mem"`` and ``"swp"`` keys, each mapping to a dict
        with ``"label"``, ``"percent"``, and ``"value_text"`` keys.
    """
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "mem": {
            "label": "Mem",
            "percent": float(mem.percent),
            "value_text": _format_bytes(int(mem.used)) + "/" + _format_bytes(int(mem.total)),
        },
        "swp": {
            "label": "Swp",
            "percent": float(swap.percent),
            "value_text": _format_bytes(int(swap.used)) + "/" + _format_bytes(int(swap.total)),
        },
    }


def collect_system_info() -> dict[str, object]:
    """Return process, load, and uptime state for the dashboard.

    Returns:
        Dict with keys ``tasks``, ``threads``, ``running``, ``load_values``,
        and ``uptime``.
    """
    uptime_s = int(time.time() - psutil.boot_time())
    tasks, threads, running = _get_task_stats()

    try:
        load_values: tuple[float, float, float] | None = psutil.getloadavg()
    except AttributeError, OSError:
        load_values = None

    return {
        "tasks": tasks,
        "threads": threads,
        "running": running,
        "load_values": load_values,
        "uptime": _format_uptime(uptime_s),
    }


def _get_task_stats() -> tuple[int, int, int]:
    """Return (task_count, thread_count, running_count) with caching.

    Uses a 1.5-second TTL cache to avoid expensive psutil iteration.

    Returns:
        Tuple of (total_tasks, total_threads, running_tasks).
    """
    now = time.time()
    if hasattr(_get_task_stats, "_cache"):
        cache_ts: float = _get_task_stats._cache_ts  # type: ignore[attr-defined]
        cache: tuple[int, int, int] | None = _get_task_stats._cache  # type: ignore[attr-defined]
        if cache is not None and now - cache_ts < 1.5:
            return cache

    task_count = 0
    thread_count = 0
    running_count = 0
    try:
        for proc in psutil.process_iter(attrs=["status", "num_threads"]):
            try:
                info = proc.info
            except Exception:  # noqa: S112
                continue
            task_count += 1
            thread_count += int(info.get("num_threads") or 0)
            if info.get("status") == psutil.STATUS_RUNNING:
                running_count += 1
    except Exception:
        if hasattr(_get_task_stats, "_cache") and _get_task_stats._cache is not None:  # type: ignore[attr-defined]
            return _get_task_stats._cache  # type: ignore[attr-defined]
        _get_task_stats._cache = (0, 0, 0)  # type: ignore[attr-defined]
        _get_task_stats._cache_ts = now  # type: ignore[attr-defined]
        return (0, 0, 0)

    _get_task_stats._cache = (task_count, thread_count, running_count)  # type: ignore[attr-defined]
    _get_task_stats._cache_ts = now  # type: ignore[attr-defined]
    return (task_count, thread_count, running_count)


def _format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable gibibytes.

    Args:
        num_bytes: Number of bytes.

    Returns:
        String like ``"16.00G"`` or ``"128.5G"``.
    """
    gib = num_bytes / (1024**3)
    if gib >= 10:
        return f"{gib:,.1f}G"
    return f"{gib:,.2f}G"


def _format_uptime(seconds: int) -> str:
    """Format seconds as HH:MM:SS.

    Args:
        seconds: Total uptime in seconds.

    Returns:
        String like ``"02:30:45"``.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
