# src/data/replay_stream.py
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from redis.client import Redis  # explicit type for Pylance
import redis

RAW = Path("data/processed/speeds_raw")
META = Path("data/processed/meta.json")
HORIZONS: List[int] = [15, 30, 45, 60]  # minutes

def iter_parquet_files():
    for daydir in sorted(RAW.glob("date=*/")):
        for f in sorted(daydir.glob("*.parquet")):
            yield f

def alpha_for(h_min: int, base_freq_min: int) -> float:
    k = max(1, h_min // base_freq_min)
    return max(0.05, min(0.5, 1.0 / (k + 1)))

def safe_float(x) -> Optional[float]:
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

def main(rate: float = 12.0) -> None:
    rc: Redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    meta = pd.read_json(META, typ="series")
    cols: List[str] = list(meta["columns"])
    freq = int(meta.get("freq_minutes", 5))
    tick_secs = 60.0 * (freq / rate)

    for f in iter_parquet_files():
        df = pd.read_parquet(f, columns=["timestamp"] + cols)
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        for row in df.itertuples(index=False, name=None):
            ts = row[0]
            values = row[1:]
            pipe = rc.pipeline()

            for c, raw_v in zip(cols, values):
                v = safe_float(raw_v)
                if v is None:
                    continue

                # 1) current speed
                pipe.set(f"edge:speed:{c}", v)

                # 2) EMA state (stored in Redis)
                ema_key = f"edge:ema:{c}"
                old_str = rc.get(ema_key)  # decode_responses=True -> str|None
                old_val = safe_float(old_str)
                ema = v if old_val is None else (0.2 * v + 0.8 * old_val)
                pipe.set(ema_key, ema)

                # 3) horizon predictions
                for h in HORIZONS:
                    a = alpha_for(h, freq)
                    pred = a * v + (1.0 - a) * ema
                    pipe.set(f"edge:pred:{c}:{h}", pred)

            pipe.execute()
            rc.xadd("speeds:raw", {"ts": ts.isoformat()}, maxlen=10000)
            time.sleep(tick_secs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=12.0, help=">1.0 plays faster than real time")
    args = parser.parse_args()
    main(rate=args.rate)
