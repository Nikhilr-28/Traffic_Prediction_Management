#!/usr/bin/env python3
"""
Build held-out multi-step NPZ windows for offline evaluation.

- Accepts CSV/Parquet or a directory of partitioned Parquet.
- Supports both TALL (timestamp, edge_id, speed) and WIDE (timestamp, s0..sN [, date]) schemas.
- Resamples to a fixed frequency, fills missing values, and clips speeds.
- Picks the last full calendar day as held-out.
- Builds sliding windows:
    X: (B, T_in, N)
    Y: (B, T_out, N)  where T_out = horizon_minutes / freq_minutes.
- Writes NPZ with keys: X, Y, cols, freq_minutes, horizon_minutes, t_in.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import DatetimeIndex


# ---------- IO & schema ----------

def read_table(input_path: Path) -> pd.DataFrame:
    """
    Load speed table. Accepts:
      • a single CSV/Parquet, or
      • a directory containing CSV/Parquet (including partitioned Parquet).
    Supports both TALL (timestamp, edge_id, speed) and WIDE (timestamp, s0…sK [, date]).
    """
    if input_path.is_dir():
        files = sorted(list(input_path.rglob("*.parquet")) +
                       list(input_path.rglob("*.pq")) +
                       list(input_path.rglob("*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV/Parquet files found under {input_path}")
        dfs = []
        for f in files:
            if f.suffix.lower() in (".parquet", ".pq"):
                dfs.append(pd.read_parquet(f))
            elif f.suffix.lower() == ".csv":
                dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
    else:
        if input_path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # Normalize timestamp
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected a 'timestamp' column; got {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

    # Drop common partition columns if present
    for part_col in ("date", "hour"):
        if part_col in df.columns:
            df = df.drop(columns=[part_col])

    # Detect schema
    cols_lower = {c.lower() for c in df.columns}
    if {"timestamp", "edge_id", "speed"}.issubset(cols_lower):
        # TALL schema → keep three columns with canonical names
        ts_col = next(c for c in df.columns if c.lower() == "timestamp")
        id_col = next(c for c in df.columns if c.lower() == "edge_id")
        sp_col = next(c for c in df.columns if c.lower() == "speed")
        return df[[ts_col, id_col, sp_col]].rename(columns={ts_col: "timestamp",
                                                            id_col: "edge_id",
                                                            sp_col: "speed"}).copy()

    # WIDE schema: use sensor columns directly
    non_ts_cols = [c for c in df.columns if c != "timestamp"]
    if not non_ts_cols:
        raise ValueError("No sensor columns found besides 'timestamp'.")
    wide = df.sort_values("timestamp").reset_index(drop=True)
    for c in non_ts_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide


def to_wide(df: pd.DataFrame, freq_minutes: int, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Produce a wide matrix indexed by a fixed frequency.
    If df is TALL (timestamp, edge_id, speed), pivot to wide.
    If WIDE, just set index and keep sensor columns.
    """
    cols = [c.lower() for c in df.columns]
    if {"timestamp", "edge_id", "speed"}.issubset(set(cols)):
        # TALL → pivot
        wide = df.pivot_table(index="timestamp", columns="edge_id", values="speed", aggfunc="mean")
    else:
        sensor_cols = [c for c in df.columns if c != "timestamp"]
        wide = df.set_index("timestamp")[sensor_cols]

    wide = wide.sort_index()

    # --- fix duplicates BEFORE reindexing ---
    if wide.index.has_duplicates:
        dup_ct = int(wide.index.duplicated(keep=False).sum())
        print(f"[make_windows] Found {dup_ct} duplicate timestamp rows → aggregating by mean.")
        wide = wide.groupby(level=0).mean()

    # Reindex to a regular grid
    freq = f"{freq_minutes}min"
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(full_index)

    # Fill missing values along the regular grid
    if fill_method == "ffill":
        wide = wide.ffill()
    elif fill_method == "interpolate":
        wide = wide.interpolate(limit_direction="both")
    elif fill_method == "none":
        pass
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    return wide.clip(lower=0, upper=150)



# ---------- Held-out day & windows ----------

def pick_last_full_day_index(wide: pd.DataFrame) -> DatetimeIndex:
    """
    Return the DatetimeIndex covering the last full calendar day present in the data.

    Uses index.floor('D') counts to avoid Pylance confusion around groupby types.
    """
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise TypeError("wide.index must be a DatetimeIndex")
    if len(wide.index) < 2:
        raise RuntimeError("Need at least two timestamps to infer sampling interval.")

    # infer sampling interval and steps per day
    step = (wide.index[1] - wide.index[0])
    if step <= pd.Timedelta(0):
        raise RuntimeError("Non-increasing timestamps.")
    steps_per_day = int(pd.Timedelta(days=1) / step)

    days = wide.index.floor("D")
    counts = days.value_counts().sort_index()  # Index is daily timestamps
    full_days = counts[counts == steps_per_day]
    if full_days.empty:
        raise RuntimeError("No full calendar day found to use as held-out.")

    target_day = full_days.index[-1]
    mask = (days == target_day)
    return DatetimeIndex(wide.index[mask])


def make_windows_multistep(wide: pd.DataFrame, t_in: int, t_out_steps: int, heldout_idx: DatetimeIndex):
    """
    Build sliding windows on the held-out day with multi-step targets.
    X: (B, t_in, N)
    Y: (B, t_out_steps, N)
    """
    values = wide.values  # (T, N)
    times = wide.index

    held_mask = times.isin(heldout_idx)
    X_list, Y_list = [], []

    # Candidate starts are label-start positions within the held-out day
    candidate_starts = np.where(held_mask)[0]
    for t in candidate_starts:
        tgt_end = t + t_out_steps - 1
        if tgt_end >= len(times):
            break
        # ensure the entire Y window is within the held-out day
        if not held_mask[t:tgt_end + 1].all():
            continue

        src_end = t - 1
        src_start = src_end - (t_in - 1)
        if src_start < 0:
            continue

        X = values[src_start:src_end + 1, :]         # (t_in, N)
        Y = values[t:tgt_end + 1, :]                 # (t_out_steps, N)

        if np.isnan(X).any() or np.isnan(Y).any():
            continue

        X_list.append(X)
        Y_list.append(Y)

    if not X_list:
        raise RuntimeError("No windows constructed. Check t_in/t_out_steps, frequency, or missing data.")

    X_arr = np.stack(X_list, axis=0)  # (B, t_in, N)
    Y_arr = np.stack(Y_list, axis=0)  # (B, t_out_steps, N)
    return X_arr, Y_arr


# ---------- Save ----------

def save_npz(out_path: Path, X: np.ndarray, y: np.ndarray, cols: List[str],
             freq_minutes: int, horizon_minutes: int, t_in: int) -> None:
    """
    Save NPZ in the evaluator's expected format:
      X: (B, T_in, N), float32
      Y: (B, T_out, N), float32   # T_out may be > 1
      cols: list of sensor/edge column names
      freq_minutes: sampling frequency
      horizon_minutes: total horizon covered by Y
      t_in: the input window length
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:   # allow callers to pass (B, N) single-step
        y = y[:, None, :]

    np.savez_compressed(
        out_path,
        X=X,
        Y=y,
        cols=np.array([str(c) for c in cols]),
        freq_minutes=np.int32(freq_minutes),
        horizon_minutes=np.int32(horizon_minutes),
        t_in=np.int32(t_in),
    )


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Build held-out NPZ windows with multi-step labels for offline eval.")
    p.add_argument("--input", type=str, required=True,
                   help="CSV/Parquet file or directory (TALL or WIDE schema).")
    p.add_argument("--freq-minutes", type=int, default=5,
                   help="Sampling frequency in minutes (default: 5).")
    p.add_argument("--t-in", type=int, default=12,
                   help="Input window length (default: 12).")
    p.add_argument("--horizon-minutes", type=int, default=60,
                   help="Total horizon in minutes covered by Y (default: 60).")
    p.add_argument("--outdir", type=str, default="data/processed/windows",
                   help="Output directory (default: data/processed/windows)")
    p.add_argument("--fill", type=str, default="ffill", choices=["ffill", "interpolate", "none"],
                   help="Missing-value handling after reindexing (default: ffill).")
    args = p.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    freq_minutes = int(args.freq_minutes)
    t_in = int(args.t_in)
    horizon_minutes = int(args.horizon_minutes)

    print(f"[make_windows] Loading: {input_path}")
    df = read_table(input_path)

    print(f"[make_windows] To wide @ {freq_minutes}min (fill='{args.fill}')")
    wide = to_wide(df, freq_minutes=freq_minutes, fill_method=args.fill)

    print("[make_windows] Picking last FULL calendar day…")
    heldout_idx = pick_last_full_day_index(wide)
    print(f"[make_windows] Held-out day: {heldout_idx[0].date()}  ticks={len(heldout_idx)}")

    if horizon_minutes % freq_minutes != 0:
        raise ValueError(f"horizon_minutes {horizon_minutes} must be a multiple of freq_minutes {freq_minutes}")
    t_out_steps = horizon_minutes // freq_minutes

    print(f"[make_windows] Building windows: t_in={t_in}, t_out={t_out_steps} steps "
          f"({t_out_steps * freq_minutes} min)")
    X, Y = make_windows_multistep(wide, t_in=t_in, t_out_steps=t_out_steps, heldout_idx=heldout_idx)

    cols = [str(c) for c in wide.columns]
    out_name = f"heldout_day_t{t_in}_o{horizon_minutes}.npz"
    out_path = outdir / out_name
    print(f"[make_windows] Saving {out_path}  (X: {X.shape}, Y: {Y.shape})")
    save_npz(out_path, X, Y, cols, freq_minutes=freq_minutes, horizon_minutes=horizon_minutes, t_in=t_in)
    print("[make_windows] Done.")


if __name__ == "__main__":
    main()
