# src/data/download_pems.py
from __future__ import annotations
from pathlib import Path
import pickle
import sys

RAW_DIR = Path("data/raw/pems-bay")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# put near the top of both src/data/download_pems.py and src/data/process_pems.py
import pickle

def load_pickle_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)  # works if py3-pickled
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")  # py2 compatibility


def verify_adj_pickle(pkl_path):
    obj = load_pickle_compat(pkl_path)
    if not isinstance(obj, (tuple, list)) or len(obj) != 3:
        raise ValueError("adj_mx_bay.pkl must be (sensor_ids, id2idx, adj_mx)")
    sensor_ids, id2idx, adj_mx = obj
    n_sensors = len(sensor_ids) if hasattr(sensor_ids, "__len__") else None
    adj_shape = getattr(adj_mx, "shape", None)
    print(f"[OK] adj pickle: sensors={n_sensors}, adj_shape={adj_shape}")
    return sensor_ids, id2idx, adj_mx

def find_first_2d_dataset(h5):
    """Return (name, dataset) for first 2D h5py.Dataset found."""
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

def verify_h5(h5_path: Path):
    try:
        import h5py  # type: ignore
    except ImportError:
        print("Install h5py to verify HDF5: python -m pip install h5py")
        return None
    with h5py.File(str(h5_path), "r") as f:
        found = find_first_2d_dataset(f)
        if not found:
            raise ValueError("No 2D dataset found in HDF5.")
        name, dset = found
        print(f"[OK] HDF5 dataset '{name}' shape={dset.shape}")
        return name, dset.shape

def main():
    adj = RAW_DIR / "adj_mx_bay.pkl"
    h5  = RAW_DIR / "speeds.h5"   # rename your file to this

    if adj.exists():
        verify_adj_pickle(adj)
    else:
        print(f"Place adj_mx_bay.pkl in {RAW_DIR.resolve()}")

    if h5.exists():
        verify_h5(h5)
    else:
        print(f"Place speeds.h5 in {RAW_DIR.resolve()} (rename your HDF5 to 'speeds.h5')")

    print("Done. If both are verified, run: python -m src.data.process_pems")
    return 0

if __name__ == "__main__":
    sys.exit(main())
