"""In-memory request limits for a single API process.

This is the local guardrail: it slows accidental loops and casual abuse
without requiring accounts. It is not a substitute for a login wall, and the
counters reset when the process restarts. One uvicorn worker is required
anyway (Chroma's local database), so a process-wide map is enough.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from rag.errors import RateLimitError

_HITS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()


def check_rate(key: str, limit: int, window_seconds: float = 60.0) -> None:
    """Record one hit for ``key`` or raise 429 when the window is full."""

    now = time.monotonic()
    with _LOCK:
        bucket = _HITS[key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            raise RateLimitError()
        bucket.append(now)
