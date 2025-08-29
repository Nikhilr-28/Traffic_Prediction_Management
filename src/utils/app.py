import streamlit as st
import numpy as np
import torch
import pandas as pd
import json, pathlib

HELDOUT = pathlib.Path("data/processed/windows/heldout_day_t12_o60.npz")
MODEL   = pathlib.Path("data/checkpoints/gwnet/gwnet_ts_ema.pt")  # fallback to gwnet_ts.pt if missing
STATS   = pathlib.Path("data/checkpoints/gwnet/stats.json")
META    = pathlib.Path("data/processed/meta.json")                # optional: per-sensor mean/std
NODES   = pathlib.Path("data/processed/nodes.csv")                # optional: node,lat,lon

st.set_page_config(layout="wide", page_title="Traffic Forecast Explorer")
st.title("Traffic Forecast — Held-out Day Error Explorer")

@st.cache_data
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32), d["cols"]

@st.cache_resource
def load_model(path):
    p = path if path.exists() else pathlib.Path(str(path).replace("_ema",""))
    m = torch.jit.load(str(p), map_location="cpu")
    m.eval()
    return m

@st.cache_data
def load_stats():
    if STATS.exists():
        s = json.loads(STATS.read_text())
        return float(s.get("mean",0.0)), float(s.get("std",1.0))
    return 0.0, 1.0

@st.cache_data
def load_meta(cols):
    # Try to get per-sensor mean/std arrays aligned with 'cols'
    if META.exists():
        try:
            m = json.loads(META.read_text())
            mu = np.array([m["means"].get(str(c), 0.0) for c in cols], dtype=np.float32) if "means" in m else None
            sd = np.array([m["stds"].get(str(c), 1.0) for c in cols], dtype=np.float32) if "stds" in m else None
            if mu is not None and sd is not None and mu.shape[0]==len(cols) and sd.shape[0]==len(cols):
                return mu, sd
        except Exception:
            pass
    return None, None

X, Y, cols = load_npz(HELDOUT)
model = load_model(MODEL)
mean, std = load_stats()
mu_s, sd_s = load_meta(cols)    # per-sensor (optional)

sample = st.slider("Max batches to score (speed vs accuracy)", 128, min(2048,len(X)), 512, step=128)
with torch.inference_mode():
    x = torch.from_numpy(X[:sample])
    y = torch.from_numpy(Y[:sample])
    yhat = model(x)                   # absolute predictions but in trainer's normalized space

# De-zscore with trainer stats
yhat_dz = yhat * std + mean
y_dz    = y    * std + mean

# Optional mph conversion if per-sensor stats are available
if mu_s is not None and sd_s is not None:
    yhat_mph = (yhat_dz.numpy() * sd_s.reshape(1,1,-1)) + mu_s.reshape(1,1,-1)
    y_mph    = (y_dz.numpy()    * sd_s.reshape(1,1,-1)) + mu_s.reshape(1,1,-1)
    abs_err_norm = (yhat_dz - y_dz).abs().numpy()
    abs_err_mph  = np.abs(yhat_mph - y_mph)
    unit_note = "mph"
else:
    abs_err_norm = (yhat_dz - y_dz).abs().numpy()
    abs_err_mph  = None
    unit_note = "normalized (trainer de-zscored)"

# Errors per node per step
err_by_step_node = abs_err_norm.mean(axis=0)   # [T_out,N]  normalized
df_err = pd.DataFrame(err_by_step_node, columns=cols)

st.subheader(f"Horizon × Node MAE ({unit_note})")
st.caption("If per-sensor stats are present in meta.json, an mph table will appear below.")
st.dataframe(df_err.round(4), use_container_width=True)

if abs_err_mph is not None:
    st.subheader("Horizon × Node MAE (mph)")
    st.dataframe(pd.DataFrame(abs_err_mph.mean(axis=0), columns=cols).round(3), use_container_width=True)

st.markdown("### Top-K high-error nodes")
c1, c2 = st.columns(2)
with c1:
    step = st.slider("Horizon step (1=5m ...)", 1, df_err.shape[0], df_err.shape[0])
with c2:
    k = st.slider("Top-K", 5, min(50, df_err.shape[1]), 20)

series = pd.Series(err_by_step_node[step-1], index=cols).sort_values(ascending=False)
st.write(series.head(k).round(4))

# Optional map if lat/lon is available
if NODES.exists():
    try:
        nodes = pd.read_csv(NODES)
        df = pd.DataFrame({"node": cols, "mae": err_by_step_node[step-1]})
        merged = nodes.merge(df, on="node", how="inner")
        st.map(merged.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude","longitude","mae"]])
    except Exception as e:
        st.warning(f"Could not render map: {e}")
else:
    st.info("Add data/processed/nodes.csv with columns node,lat,lon to view a map.")
