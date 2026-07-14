"""Lightweight network-traffic metrics logging for online search/browse tools.

Disabled (no-op) unless NET_METRICS_LOG_FILE is set, so it has zero effect on
existing behavior when not configured. Appends one JSON line per network
request, tagged with the trajectory that issued it, for later analysis when
designing rate limiting / shared caching.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

_lock = threading.Lock()
_log_path = os.getenv("NET_METRICS_LOG_FILE", "").strip()


def enabled() -> bool:
    return bool(_log_path)


def log_request(
    *,
    tool: str,
    trajectory_id: Optional[str],
    kind: str,
    target: str,
    method: str = "GET",
    bytes_out: int = 0,
    bytes_in: int = 0,
    elapsed_s: float = 0.0,
    status=None,
    cache_hit: Optional[bool] = None,
    extra: Optional[dict] = None,
) -> None:
    if not _log_path:
        return
    entry = {
        "ts": time.time(),
        "tool": tool,
        "trajectory_id": trajectory_id,
        "kind": kind,
        "target": target,
        "method": method,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "elapsed_s": elapsed_s,
        "status": status,
        "cache_hit": cache_hit,
        "extra": extra or {},
    }
    try:
        line = json.dumps(entry, ensure_ascii=False)
    except Exception:
        return
    with _lock:
        try:
            with open(_log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


@contextmanager
def timed_request(
    *,
    tool: str,
    trajectory_id: Optional[str],
    kind: str,
    target: str,
    method: str = "GET",
    bytes_out: int = 0,
    cache_hit: Optional[bool] = None,
):
    """Context manager that logs elapsed time and lets the caller fill in
    bytes_in/status once the request completes.

    Usage:
        with timed_request(tool="web_browser", trajectory_id=tid, kind="url",
                            target=url) as m:
            resp = requests.get(url)
            m["bytes_in"] = len(resp.content)
            m["status"] = resp.status_code
    """
    m = {"bytes_in": 0, "status": None}
    start = time.time()
    try:
        yield m
    finally:
        log_request(
            tool=tool,
            trajectory_id=trajectory_id,
            kind=kind,
            target=target,
            method=method,
            bytes_out=bytes_out,
            bytes_in=m.get("bytes_in", 0),
            elapsed_s=time.time() - start,
            status=m.get("status"),
            cache_hit=cache_hit,
        )
