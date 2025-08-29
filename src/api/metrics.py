# src/api/metrics.py
from __future__ import annotations
import os
import time
import datetime as dt
from typing import Optional, Any
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool  # <- use this for sync Redis calls

router = APIRouter()

# ---- Redis client selection: async if available, else sync ----
ASYNC_MODE = False
try:
    import redis.asyncio as redis  # type: ignore
    ASYNC_MODE = True
except Exception:
    import redis  # type: ignore

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

if ASYNC_MODE:
    r = redis.from_url(REDIS_URL, decode_responses=True)
else:
    from urllib.parse import urlparse

    parsed = urlparse(REDIS_URL)
    host = parsed.hostname or "redis"
    port = int(parsed.port or 6379)
    db = 0
    if parsed.path and parsed.path.strip("/").isdigit():
        db = int(parsed.path.strip("/"))
    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

async def redis_get(key: str) -> Optional[str]:
    if ASYNC_MODE:
        return await r.get(key)  # type: ignore[no-any-return]
    # offload sync get() to a worker thread; keeps event loop non-blocking
    return await run_in_threadpool(r.get, key)  # type: ignore[arg-type]

def _parse_ts(value: str) -> Optional[float]:
    # Try epoch seconds
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # Try ISO-8601
    try:
        try:
            dt_obj = dt.datetime.fromisoformat(value)
        except ValueError:
            dt_obj = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt_obj.timestamp()
    except Exception:
        return None

@router.get("/metrics")
async def metrics() -> dict[str, Any]:
    now = time.time()

    last_ts_raw = await redis_get("speeds:last_ts")
    last_ts: Optional[float] = _parse_ts(last_ts_raw) if last_ts_raw is not None else None
    age = None if last_ts is None else max(0.0, now - last_ts)

    return {
        "now": now,
        "speeds_last_ts": last_ts,
        "prediction_age_seconds": age,
        "keys_pred_count": await _count_pred_keys(),
    }

async def _count_pred_keys() -> int:
    pattern = "edge:pred:*"
    if ASYNC_MODE:
        # Async SCAN
        cursor = 0
        count = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=pattern, count=1000)  # type: ignore[attr-defined]
            count += len(keys)
            if cursor == 0:
                break
        return count

    # Sync SCAN offloaded to threadpool
    def _scan_all() -> int:
        c = 0
        cur = 0
        while True:
            cur, keys = r.scan(cursor=cur, match=pattern, count=1000)  # type: ignore[attr-defined]
            c += len(keys)
            if cur == 0:
                break
        return c

    return await run_in_threadpool(_scan_all)
