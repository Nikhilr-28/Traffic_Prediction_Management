# src/data/process_pems.py
from __future__ import annotations
from pathlib import Path
import pickle, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw/pems-bay")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# put near the top of both src/data/download_pems.py and src/data/process_pems.py
import pickle

def load_pickle_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)  # works if py3-pickled
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")  # py2 compatibility


def load_adj():
    p = RAW_DIR / "adj_mx_bay.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    sensor_ids, id2idx, adj_mx = load_pickle_compat(p)
    if not isinstance(sensor_ids, (list, tuple)) or not isinstance(id2idx, dict):
        raise ValueError("adj_mx_bay.pkl must be (sensor_ids: list, id2idx: dict, adj_mx: ndarray)")
    return sensor_ids, id2idx, adj_mx


def find_first_2d_dataset(h5):
    import h5py
    def _walk(g, prefix=""):
        for k, v in g.items():
            name = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset) and v.ndim >= 2:
                return name, v
            if isinstance(v, h5py.Group):
                res = _walk(v, name)
                if res is not None:
                    return res
        return None
    return _walk(h5)

def load_speeds():
    import h5py
    h5_path = RAW_DIR / "speeds.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")
    with h5py.File(str(h5_path), "r") as f:
        found = find_first_2d_dataset(f)
        if not found:
            raise ValueError("No 2D dataset found in HDF5.")
        name, dset = found
        arr = dset[()]  # numpy array
        print(f"[INFO] Using dataset '{name}' with shape={arr.shape}")
    return arr  # shape T×N or N×T

def to_dataframe(speed_matrix: np.ndarray, freq_minutes: int = 5):
    if speed_matrix.ndim != 2:
        raise ValueError("Expected a 2D speed matrix")
    # ensure T x N
    if speed_matrix.shape[0] < speed_matrix.shape[1]:
        speed_matrix = speed_matrix.T
    T, N = speed_matrix.shape
    ts = pd.date_range("2017-01-01", periods=T, freq=f"{freq_minutes}min")
    df = pd.DataFrame(speed_matrix, index=ts, columns=[f"s{i}" for i in range(N)])
    df.index.name = "timestamp"
    return df

def compute_congestion_weights(raw_df: pd.DataFrame):
    speeds = raw_df.values
    ff = np.nanpercentile(speeds, 90, axis=0)
    ff = np.maximum(ff, 1.0)
    ratio = speeds / ff
    weights = np.where(ratio < 0.4, 2.0, 1.0)
    return pd.DataFrame(weights, index=raw_df.index, columns=raw_df.columns)

def normalize(df: pd.DataFrame):
    mu = df.mean()
    sigma = df.std().replace(0, 1.0)
    norm = (df - mu) / sigma
    return norm, mu, sigma

def write_partitioned(df: pd.DataFrame, name: str):
    df = df.reset_index()
    df["date"] = df["timestamp"].dt.date.astype(str)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, root_path=str(OUT_DIR / name), partition_cols=["date"])

def main():
    sensor_ids, id2idx, adj = load_adj()
    raw = load_speeds()
    raw_df = to_dataframe(raw, freq_minutes=5)

    wdf = compute_congestion_weights(raw_df)    # BEFORE normalization
    norm_df, mu, sigma = normalize(raw_df)

    write_partitioned(raw_df,  "speeds_raw")
    write_partitioned(norm_df, "speeds_norm")
    write_partitioned(wdf,     "weights")

    meta = {
        "n_sensors": raw_df.shape[1],
        "columns": list(raw_df.columns),
        "freq_minutes": 5,
        "mu": {k: float(v) for k, v in mu.items()},
        "sigma": {k: float(v) for k, v in sigma.items()},
    }
    (OUT_DIR / "meta.json").write_text(pd.Series(meta, dtype="object").to_json(indent=2), encoding="utf-8")
    print(f"[OK] Wrote Parquet datasets under {OUT_DIR.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
