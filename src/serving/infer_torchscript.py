# src/serving/infer_torchscript.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

class TSForecastor:
    def __init__(self, ts_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(ts_path, map_location=self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def predict(self, recent: np.ndarray) -> np.ndarray:
        """
        recent: [T_in, N] normalized speeds
        returns: [T_out, N] normalized forecast
        """
        x = torch.from_numpy(recent[None, ...]).to(self.device)  # [1, T_in, N]
        y = self.model(x)                                        # [1, T_out, N]
        return y.squeeze(0).cpu().numpy()

# simple singleton loader
_ts_instance: TSForecastor | None = None
def get_ts(ts_path: str = "data/checkpoints/gwnet/gwnet_ts.pt", device: str = "cpu") -> TSForecastor:
    global _ts_instance
    if _ts_instance is None:
        _ts_instance = TSForecastor(ts_path, device=device)
    return _ts_instance
