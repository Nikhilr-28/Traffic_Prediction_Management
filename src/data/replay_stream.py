# src/data/replay_stream.py
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
from redis.client import Redis
import redis

from src.serving.infer_torchscript import get_ts

RAW = Path("data/processed/speeds_raw")
META = Path("data/processed/meta.json")

HORIZONS: List[int] = [15, 30, 45, 60]  # minutes
T_IN = 12   # must match training input length
T_OUT = 12  # model output length (steps)


def iter_parquet_files():
    for daydir in sorted(RAW.glob("date=*/")):
        for f in sorted(daydir.glob("*.parquet")):
            yield f


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    if pd.isna(x):
        return None
    v = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(v) else float(v)


def ffill_inplace(arr: np.ndarray) -> None:
    """Forward-fill NaNs along time for shape [T, N]."""
    T, N = arr.shape
    for n in range(N):
        col = arr[:, n]
        last = None
        for t in range(T):
            if np.isnan(col[t]):
                if last is not None:
                    col[t] = last
            else:
                last = col[t]
        # if all NaN, set to 0
        if last is None:
            col[:] = 0.0


def main(rate: float = 12.0) -> None:
    rc: Redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Load metadata (columns, mu, sigma, freq)
    meta = pd.read_json(META, typ="series")
    cols: List[str] = list(meta["columns"])
    freq = int(meta.get("freq_minutes", 5))
    mu_map = meta.get("mu", {}) or {}
    sg_map = meta.get("sigma", {}) or {}
    mu = np.array([float(mu_map.get(c, 0.0)) for c in cols], dtype=np.float32)    # [N]
    sg = np.array([float(sg_map.get(c, 1.0)) for c in cols], dtype=np.float32)    # [N]
    sg[sg == 0.0] = 1.0

    tick_secs = 60.0 * (freq / rate)

    # Model (CPU for serving)
    ts_model = get_ts("data/checkpoints/gwnet/gwnet_ts.pt", device="cpu")

    # Rolling window of last T_IN raw-speed vectors
    history: deque[np.ndarray] = deque(maxlen=T_IN)

    for f in iter_parquet_files():
        df = pd.read_parquet(f, columns=["timestamp"] + cols)
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Process each tick
        for row in df.itertuples(index=False, name=None):
            ts = row[0]
            values = row[1:]  # tuple aligned to cols
            pipe = rc.pipeline()

            # write current speeds + update EMA as fallback
            curr_vec = np.empty((len(cols),), dtype=np.float32)
            for i, (c, raw_v) in enumerate(zip(cols, values)):
                v = safe_float(raw_v)
                curr_vec[i] = np.nan if v is None else v

                if v is None:
                    continue

                # 1) current speed
                pipe.set(f"edge:speed:{c}", v)

                # 2) EMA fallback
                ema_key = f"edge:ema:{c}"
                old_raw: Any = rc.get(ema_key)  # str|None
                old_val = safe_float(old_raw)
                ema = v if old_val is None else (0.2 * v + 0.8 * old_val)
                pipe.set(ema_key, ema)

            pipe.execute()

            # update history
            history.append(curr_vec.copy())

            # model prediction once we have T_IN frames
            if len(history) == T_IN:
                recent = np.stack(list(history), axis=0)  # [T_IN, N] raw speeds
                # impute missing
                ffill_inplace(recent)

                # normalize per sensor
                norm = (recent - mu) / sg  # [T_IN, N]

                # model forecast in normalized space
                y_norm = ts_model.predict(norm.astype(np.float32))  # [T_OUT, N]

                # de-normalize
                y = y_norm * sg + mu  # [T_OUT, N]

                # write horizons
                steps_per_15 = max(1, 15 // freq)  # steps per 15 minutes
                with rc.pipeline() as p2:
                    for i, h in enumerate(HORIZONS, start=1):
                        idx = i * steps_per_15  # 15→step, 30→2*step, ...
                        if idx < 1 or idx > y.shape[0]:
                            continue
                        pred_slice = y[idx - 1]  # [N]
                        for c, vhat in zip(cols, pred_slice):
                            p2.set(f"edge:pred:{c}:{h}", float(vhat))
                    p2.execute()

            # update freshness marker
            rc.set("speeds:last_ts", ts.isoformat())

            # optional stream record
            rc.xadd("speeds:raw", {"ts": ts.isoformat()}, maxlen=10000)

            time.sleep(tick_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=12.0, help=">1.0 plays faster than real time")
    args = parser.parse_args()
    main(rate=args.rate)
    print("Done.")