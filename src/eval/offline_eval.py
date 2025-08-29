from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import torch

# --- config ---
# If your held-out file is single-step (T_out=1), you can temporarily
# point this at your multi-step training windows to verify end-to-end:
# WINDOWS_HELDOUT = Path("data/processed/windows/windows_t12_o12.npz")
WINDOWS_HELDOUT = Path("data/processed/windows/heldout_day_t12_o60.npz")
CKPT = Path("data/checkpoints/gwnet/gwnet_ts.pt")
STATS = Path("data/checkpoints/gwnet/stats.json")
HORIZONS = [15, 30, 45, 60]   # minutes
EMA_ALPHA = 0.3               # baseline smoothing factor

def load_data(npz_path: Path):
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"].astype(np.float32)   # [B, T_in, N]
    Y = npz["Y"].astype(np.float32)   # [B, T_out, N]
    return X, Y

def ema_baseline(x_hist: np.ndarray, t_out: int) -> np.ndarray:
    # x_hist: [T_in, N], oldest -> newest
    ema = x_hist[0]
    for t in range(1, x_hist.shape[0]):
        ema = EMA_ALPHA * x_hist[t] + (1 - EMA_ALPHA) * ema
    pred = np.repeat(ema[None, :], t_out, axis=0)
    return np.clip(pred, 0.0, 150.0)

def take_horizon(y_pred: np.ndarray, horizon: int, minutes_per_step: int) -> np.ndarray:
    steps = max(1, min(y_pred.shape[0], horizon // minutes_per_step))
    return y_pred[:steps]

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(np.mean(diff**2)))
    return mae, rmse

def main():
    assert WINDOWS_HELDOUT.exists(), f"Missing {WINDOWS_HELDOUT}"
    assert CKPT.exists(), f"Missing {CKPT}"
    assert STATS.exists(), f"Missing {STATS}"

    X, Y = load_data(WINDOWS_HELDOUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer minutes per step
    assert X.ndim == 3 and Y.ndim == 3, f"Bad shapes: X{X.shape} Y{Y.shape}"
    t_out = int(Y.shape[1])

    if t_out == 1:
        # Fallback: treat the single label as a 60-min step (so all horizons use the same slice)
        minutes_per_step = 60
        print("WARNING: T_out == 1 (single-step labels). Horizons will be identical. "
              "Regenerate held-out windows with multi-step labels (e.g., T_out=12 for 5-min steps).")
    else:
        assert 60 % t_out == 0, f"60 minutes not divisible by T_out={t_out}"
        minutes_per_step = 60 // t_out

    # load model & stats
    model = torch.jit.load(str(CKPT), map_location=device).eval()
    stats = json.loads(Path(STATS).read_text())
    mean = float(stats.get("mean", 0.0))
    std  = float(stats.get("std",  1.0))
    if std < 1e-6: std = 1.0

    results = {h: {"ema": {"mae": 0.0, "rmse": 0.0, "n": 0},
                   "model": {"mae": 0.0, "rmse": 0.0, "n": 0}} for h in HORIZONS}

    with torch.inference_mode():
        for i in range(len(X)):
            x_np = X[i]                # [T_in, N]
            y_np = Y[i]                # [T_out, N]

            # normalize input to model domain
            x_n = (x_np - mean) / std
            x_t = torch.from_numpy(x_n[None, ...]).to(device)  # [1, T_in, N]

            # model predicts normalized → de-normalize to mph
            y_hat_n = model(x_t).detach().cpu().numpy()[0]     # [T_out, N]
            y_hat   = y_hat_n * std + mean                     # mph
            y_hat   = np.clip(y_hat, 0.0, 150.0)

            # ema baseline (mph)
            y_ema = ema_baseline(x_np, t_out=y_np.shape[0])

            for h in HORIZONS:
                yt   = take_horizon(y_np,   h, minutes_per_step)
                yh_m = take_horizon(y_hat,  h, minutes_per_step)
                yh_e = take_horizon(y_ema,  h, minutes_per_step)

                mae_m, rmse_m = metrics(yt, yh_m)
                mae_e, rmse_e = metrics(yt, yh_e)

                results[h]["model"]["mae"] += mae_m
                results[h]["model"]["rmse"] += rmse_m
                results[h]["ema"]["mae"]   += mae_e
                results[h]["ema"]["rmse"]  += rmse_e
                results[h]["model"]["n"]   += 1
                results[h]["ema"]["n"]     += 1

    print("\n=== Offline Eval (held-out day, TorchScript GWNet) ===")
    for h in HORIZONS:
        m = results[h]["model"]; e = results[h]["ema"]; n = max(1, m["n"])
        print(f"H{h:>3} | MODEL  MAE {m['mae']/n:6.3f}  RMSE {m['rmse']/n:6.3f} "
              f"| EMA  MAE {e['mae']/n:6.3f}  RMSE {e['rmse']/n:6.3f}")

if __name__ == "__main__":
    main()
