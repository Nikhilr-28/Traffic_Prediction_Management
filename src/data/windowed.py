#!/usr/bin/env python3
"""
Simple window builder for sanity checks.

- Loads processed normalized speeds (partitioned Parquet).
- Restricts to the last full calendar day (to mimic offline eval).
- Builds sliding windows with multi-step Y and writes an NPZ.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


PROC = Path("data/processed")
OUT  = PROC / "windows"
OUT.mkdir(parents=True, exist_ok=True)


def _load_meta_freq(default_freq: int = 5) -> int:
    meta_path = PROC / "meta.json"
    if meta_path.exists():
        try:
            meta = pd.read_json(meta_path, typ="series")
            return int(meta.get("freq_minutes", default_freq))
        except Exception:
            return default_freq
    return default_freq


def _pick_last_full_day_index(wide: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise TypeError("wide.index must be a DatetimeIndex")
    if len(wide.index) < 2:
        raise RuntimeError("Insufficient timestamps to infer frequency.")
    step = (wide.index[1] - wide.index[0])
    if step <= pd.Timedelta(0):
        raise RuntimeError("Non-increasing timestamps.")
    steps_per_day = int(pd.Timedelta(days=1) / step)

    days = wide.index.floor("D")
    counts = days.value_counts().sort_index()
    full_days = counts[counts == steps_per_day]
    if full_days.empty:
        raise RuntimeError("No full calendar day present.")
    target_day = full_days.index[-1]
    mask = (days == target_day)
    return pd.DatetimeIndex(wide.index[mask])


def build_windows(t_in: int = 12, t_out: int = 12, freq_min: int | None = None):
    if freq_min is None:
        freq_min = _load_meta_freq(default_freq=5)

    root = PROC / "speeds_norm"
    parts = sorted(root.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet files under {root.resolve()}")
    dfs = [pd.read_parquet(p).sort_values("timestamp") for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp")

    cols: List[str] = [c for c in df.columns if c != "timestamp"]
    wide = df.set_index("timestamp")[cols].sort_index()

    held_idx = _pick_last_full_day_index(wide)
    wide = wide.loc[wide.index.isin(held_idx)]

    arr = wide.to_numpy(dtype=np.float32)  # [T, N]
    T, N = arr.shape
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    for t in range(T - t_in - t_out + 1):
        X_list.append(arr[t:t+t_in])              # [t_in, N]
        Y_list.append(arr[t+t_in:t+t_in+t_out])   # [t_out, N]
    if not X_list:
        raise RuntimeError("No windows built; check t_in/t_out vs ticks in the last full day.")
    X = np.stack(X_list)  # [B, t_in, N]
    Y = np.stack(Y_list)  # [B, t_out, N]

    out_path = OUT / f"heldout_day_t{t_in}_o{t_out*freq_min}.npz"
    np.savez_compressed(out_path, X=X.astype(np.float32), Y=Y.astype(np.float32),
                        cols=np.array(cols), freq_minutes=np.int32(freq_min),
                        horizon_minutes=np.int32(t_out*freq_min), t_in=np.int32(t_in))
    print("windows:", X.shape, Y.shape, "->", out_path)


if __name__ == "__main__":
    build_windows()
