# src/train/train_gwnet.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, ContextManager, cast
from contextlib import nullcontext
import warnings
import json
import random
import math
import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

"""
Train a residualized traffic forecaster with GRAPH MIXING and export TorchScript.

Includes:
- Bottleneck temporal conv (per-node temporal features)
- GraphMix: adjacency-masked learnable mixing across nodes (applied to residuals)
- Residualization vs EMA(last) inside wrapper → ABSOLUTE normalized outputs
- SmoothL1 loss, gentle step-weighting for longer horizons
- AMP, gradient clipping
- Full-dataset DataLoader
- OneCycleLR (per-batch) for faster convergence
- Parameter EMA (Polyak) and EMA export to TorchScript
"""

# -------------------------- Config --------------------------

CKPT_DIR = Path("data/checkpoints/gwnet")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Training windows NPZ (X:[B,T_in,N], Y:[B,T_out,N])
WINDOWS = Path("data/processed/windows/windows_t12_o60.npz")

# Optional adjacency (auto-discovered if ADJ_PATH is None)
ADJ_PATH: Optional[Path] = None  # e.g., Path("data/processed/adjacency.npz")

# Training hyperparameters
EPOCHS = 90                   # faster wall-clock with OneCycleLR
BASE_LR = 3e-4
MAX_LR = 6e-4                 # OneCycle peak LR
MIN_LR = 3e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
NUM_WORKERS = 0               # 0 on Windows avoids spawn overhead
PIN_MEMORY = True
CLIP_NORM = 1.0
INPUT_NOISE_STD = 0.00        # set 0.01 for tiny jitter in normalized space

# Loss shaping
USE_STEP_WEIGHTING = True
STEP_WEIGHT_MAX = 1.3         # last horizon gets this weight (start=1.0)

# Residualization mode
USE_EMA_REFERENCE = True      # True: residual vs EMA(last); False: residual vs last tick
EMA_ALPHA = 0.2               # smoothing factor for EMA reference

# Parameter-EMA (Polyak averaging)
USE_PARAM_EMA = True
PARAM_EMA_DECAY = 0.999

# Model sizes
BOTTLENECK_K = 64
DROPOUT = 0.05

# Reproducibility
SEED = 42

# Silence legacy cuda.amp deprecation nudges at runtime
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")


# -------------------------- Graph utilities --------------------------

def _discover_adj_path() -> Optional[Path]:
    """Try a few common locations/formats for adjacency."""
    candidates = [
        Path("data/processed/adjacency.npz"),
        Path("data/processed/weights/adjacency.npz"),
        Path("data/raw/pems-bay/adjacency.npz"),
        Path("data/raw/pems-bay/adj.npy"),
        Path("data/processed/graph.npy"),
        Path("data/processed/graph.csv"),              # expects src,dst[,w]
        Path("data/raw/pems-bay/graph.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_adjacency(n_nodes: int, hint: Optional[Path]) -> np.ndarray:
    """
    Load adjacency as a dense [N,N] float32 array with self-loops.
    Accepted:
      - .npz with key 'A' or 'adj'
      - .npy dense array
      - .csv with columns src,dst[,w] (0-indexed or string ids convertible to int)
    If missing/invalid, returns identity.
    """
    path = hint or _discover_adj_path()
    if path is None or not path.exists():
        print("[graph] No adjacency file found → using identity (no spatial mixing).")
        return np.eye(n_nodes, dtype=np.float32)

    try:
        if path.suffix.lower() == ".npz":
            z = np.load(path)
            key = "A" if "A" in z else ("adj" if "adj" in z else None)
            if key is None:
                raise ValueError("NPZ missing 'A' or 'adj' key.")
            A = np.array(z[key], dtype=np.float32)
        elif path.suffix.lower() == ".npy":
            A = np.array(np.load(path), dtype=np.float32)
        elif path.suffix.lower() == ".csv":
            A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            with path.open("r", newline="") as f:
                rdr = csv.DictReader(f)
                if rdr.fieldnames is None or "src" not in rdr.fieldnames or "dst" not in rdr.fieldnames:
                    raise ValueError("CSV must have 'src','dst'[, 'w'] columns")
                for row in rdr:
                    i = int(row["src"]); j = int(row["dst"])
                    w = float(row["w"]) if "w" in row and row["w"] not in (None, "") else 1.0
                    if 0 <= i < n_nodes and 0 <= j < n_nodes:
                        A[i, j] = max(A[i, j], w)
        else:
            raise ValueError(f"Unsupported adjacency format: {path.suffix}")
    except Exception as e:
        print(f"[graph] Failed to load adjacency from {path}: {e} → using identity.")
        A = np.eye(n_nodes, dtype=np.float32)

    # ensure square of correct size
    if A.shape != (n_nodes, n_nodes):
        print(f"[graph] Adjacency shape {A.shape} != ({n_nodes},{n_nodes}) → padding/cropping.")
        B = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        m = min(n_nodes, A.shape[0])
        n = min(n_nodes, A.shape[1])
        B[:m, :n] = A[:m, :n]
        A = B

    # add self-loops
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)


class GraphMix(nn.Module):
    """
    Adjacency-masked learnable mixing: for each time step t,
      y[t] = x[t] @ (softplus(W) ⊙ M + I)
    where M is a binary mask from adjacency (including self-loops).

    x: [B, T, N]  → y: [B, T, N]
    """
    def __init__(self, adj_dense: np.ndarray):
        super().__init__()
        A = torch.tensor(adj_dense, dtype=torch.float32)
        mask = (A > 0).to(torch.float32)
        self.register_buffer("mask", mask)              # [N,N]
        self.W = nn.Parameter(torch.zeros_like(mask))   # [N,N], learnable
        self.register_buffer("I", torch.eye(mask.size(0), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N], mix across node dim N
        Wp = torch.nn.functional.softplus(self.W)       # ensure non-negative
        M = Wp * self.mask + self.I                     # masked weights + self
        # (B,T,N) x (N,N) -> (B,T,N)
        return torch.einsum("btn,nm->btm", x, M)


# -------------------------- Models --------------------------

class SimpleGWNet(nn.Module):
    """
    Bottleneck temporal block with GraphMix on residuals:
      per-node temporal features (N→K→N via 1×1 + dilated 3×1 + 1×1),
      then GraphMix across nodes, returning RESIDUALS (normalized).
    """
    def __init__(self, n_nodes: int, t_in: int, t_out: int,
                 k_bottleneck: int = 64, dropout: float = 0.05,
                 graph_mix: Optional[GraphMix] = None):
        super().__init__()
        self.proj_in  = nn.Conv1d(in_channels=n_nodes, out_channels=k_bottleneck, kernel_size=1)
        self.temporal = nn.Conv1d(in_channels=k_bottleneck, out_channels=k_bottleneck,
                                  kernel_size=3, padding=2, dilation=2)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head     = nn.Conv1d(in_channels=k_bottleneck, out_channels=n_nodes, kernel_size=1)
        self.t_out    = t_out
        self.graph_mix = graph_mix  # may be None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, N] → residuals: [B, T_out, N]
        x = x.transpose(1, 2)                      # [B,N,T_in]
        h = torch.relu(self.proj_in(x))
        h = torch.relu(self.temporal(h))
        h = self.dropout(h)
        y = self.head(h)[..., -self.t_out:]        # [B,N,T_out]
        y = y.transpose(1, 2)                      # [B,T_out,N] residuals
        if self.graph_mix is not None:
            y = self.graph_mix(y)                  # mix neighbors on residuals
        return y


class Residualize(nn.Module):
    """Wrap residual predictor → ABSOLUTE normalized prediction using LAST tick."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base(x)                          # [B, T_out, N]
        last = x[:, -1:, :].expand_as(res)         # [B, T_out, N]
        return last + res


class ResidualizeEMA(nn.Module):
    """Wrap residual predictor → ABSOLUTE normalized prediction using EMA(last)."""
    def __init__(self, base: nn.Module, alpha: float = 0.2):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ema = x[:, :1, :]                           # [B,1,N]
        for t in range(1, x.size(1)):
            ema = self.alpha * x[:, t:t+1, :] + (1 - self.alpha) * ema
        last_ref = ema[:, -1:, :]                   # [B,1,N]
        res = self.base(x)                          # [B, T_out, N]
        return last_ref.expand_as(res) + res        # [B, T_out, N]


# -------------------------- AMP & EMA utils --------------------------

def setup_amp(device_type: str) -> Tuple[Optional[Any], ContextManager[None]]:
    """Returns (scaler, ctx). Prefers torch.amp; falls back to torch.cuda.amp."""
    use_cuda = (device_type == "cuda")
    if not use_cuda:
        return None, cast(ContextManager[None], nullcontext())
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler") and hasattr(amp_mod, "autocast"):
        scaler = getattr(amp_mod, "GradScaler")("cuda")
        ctx    = cast(ContextManager[None], getattr(amp_mod, "autocast")("cuda"))
        return scaler, ctx
    from torch.cuda import amp as cuda_amp  # type: ignore
    return cuda_amp.GradScaler(), cast(ContextManager[None], cuda_amp.autocast())


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
    for t_buf, s_buf in zip(target.buffers(), source.buffers()):
        t_buf.data.mul_(decay).add_(s_buf.data, alpha=1 - decay)


# -------------------------- Training --------------------------

def main():
    # Repro
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    assert WINDOWS.exists(), f"Missing windows file: {WINDOWS}"
    npz = np.load(WINDOWS, allow_pickle=True)
    X_np: np.ndarray = npz["X"].astype(np.float32)  # [B,T_in,N]
    Y_np: np.ndarray = npz["Y"].astype(np.float32)  # [B,T_out,N]

    # Light global z-score stats
    flat = np.concatenate([X_np.reshape(-1), Y_np.reshape(-1)])
    mean = float(np.mean(flat)); std = float(np.std(flat)) or 1.0
    (CKPT_DIR / "stats.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"[train] stats: mean={mean:.4f} std={std:.4f} → saved {CKPT_DIR/'stats.json'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    n_nodes = X_np.shape[2]; t_in = X_np.shape[1]; t_out = Y_np.shape[1]

    # Dataset / DataLoader
    ds = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(Y_np))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device.type == "cuda"),
                        drop_last=False)

    # GraphMix
    A = load_adjacency(n_nodes, ADJ_PATH)                      # [N,N]
    graph_mix = GraphMix(A).to(device)

    # Base model + GraphMix, then residualization to absolute
    base = SimpleGWNet(n_nodes, t_in, t_out, k_bottleneck=BOTTLENECK_K,
                       dropout=DROPOUT, graph_mix=graph_mix)
    model = (ResidualizeEMA(base, alpha=EMA_ALPHA) if USE_EMA_REFERENCE else Residualize(base)).to(device)

    # Parameter-EMA
    if USE_PARAM_EMA:
        base_ema = SimpleGWNet(n_nodes, t_in, t_out, k_bottleneck=BOTTLENECK_K,
                               dropout=DROPOUT, graph_mix=graph_mix)
        model_ema = (ResidualizeEMA(base_ema, alpha=EMA_ALPHA) if USE_EMA_REFERENCE else Residualize(base_ema)).to(device)
        model_ema.load_state_dict(model.state_dict())
        for p in model_ema.parameters(): p.requires_grad_(False)
    else:
        model_ema = None

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    # Step weighting (gently emphasize longer horizons)
    if USE_STEP_WEIGHTING and t_out > 1:
        sw = torch.linspace(1.0, STEP_WEIGHT_MAX, steps=t_out, device=device)
        step_weights = (sw / sw.mean()).view(1, -1, 1)  # [1,T_out,1]
    else:
        step_weights = torch.ones((1, t_out, 1), device=device)

    # AMP
    scaler, ctx = setup_amp(device.type)

    # Normalization tensors
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
    std_t  = torch.tensor(std,  device=device, dtype=torch.float32)

    # OneCycleLR (per-batch)
    steps_per_epoch = max(1, len(loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, div_factor=(BASE_LR / MAX_LR if MAX_LR > 0 else 25.0),
        final_div_factor=(BASE_LR / MIN_LR if MIN_LR > 0 else 10.0)
    )

    # Training
    for epoch in range(EPOCHS):
        running_mae = 0.0; batches = 0
        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            # Light z-normalization
            xb_n = (xb - mean_t) / std_t
            yb_n = (yb - mean_t) / std_t

            # Optional tiny input noise
            if INPUT_NOISE_STD > 0.0:
                xb_n = xb_n + torch.randn_like(xb_n) * INPUT_NOISE_STD

            model.train(); opt.zero_grad(set_to_none=True)

            if scaler is None:
                with ctx:
                    yhat_n = model(xb_n)  # absolute normalized
                    loss = loss_fn(yhat_n * step_weights, yb_n * step_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
            else:
                with ctx:
                    yhat_n = model(xb_n)
                    loss = loss_fn(yhat_n * step_weights, yb_n * step_weights)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update()

            # Update EMA
            if model_ema is not None:
                update_ema(model_ema, model, PARAM_EMA_DECAY)

            # Report MAE in de-zscored space (same norm as inputs)
            with torch.no_grad():
                yhat = yhat_n * std_t + mean_t
                mae = torch.mean(torch.abs(yhat - yb)).item()
            running_mae += mae; batches += 1

            scheduler.step()

        avg_mae = running_mae / max(1, batches)
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:03d} | lr {lr_now:.6f} | train_mae_norm {avg_mae:.4f}")

    # Export TorchScript (EMA model if available)
    export_model = model_ema if model_ema is not None else model
    export_model.eval()
    example = torch.from_numpy(X_np[:1]).to(device)
    example_n = (example - mean_t) / std_t
    with torch.inference_mode():
        traced = torch.jit.trace(export_model, example_n)
    out_path = CKPT_DIR / ("gwnet_ts_ema.pt" if model_ema is not None else "gwnet_ts.pt")
    torch.jit.save(traced, str(out_path))
    print(f"saved TorchScript: {out_path}")


if __name__ == "__main__":
    main()
