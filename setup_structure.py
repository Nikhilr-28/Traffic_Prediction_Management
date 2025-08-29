#!/usr/bin/env python3
"""
One-shot project bootstrap for a lean MVP.
- Creates folders
- Writes minimal FastAPI app with health + metrics
- Writes .env.example, .gitignore, requirements stubs
Safe to re-run.
"""
from pathlib import Path

ROOT = Path.cwd()

GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
.build/
.venv/
.env
env/
venv/
*.egg-info/

# Data & artifacts
data/raw/
data/processed/
data/cache/
data/reports/
mlruns/
checkpoints/
*.duckdb
*.parquet
*.pkl
*.pt
*.onnx

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"""

ENV_EXAMPLE = """\
# FastAPI / service
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
MODEL_GENERATION=1

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
"""

REQUIREMENTS_GPU = """\
--index-url https://download.pytorch.org/whl/cu121
--extra-index-url https://pypi.org/simple
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
"""

REQUIREMENTS_IN = """\
# compiled with: pip-compile requirements.in -o requirements.txt
# (Install requirements-gpu.txt FIRST to keep CUDA wheels.)

# Core
numpy>=1.24
pandas>=2.0
polars>=0.20
pyarrow>=14
duckdb>=0.10
tqdm>=4.66

# API
fastapi>=0.109
uvicorn[standard]>=0.25
pydantic>=2.5
pydantic-settings>=2.1
python-dotenv>=1.0

# Geo / graph
osmnx>=1.8
geopandas>=0.14
shapely>=2.0
networkx>=3.0
folium>=0.15
haversine>=2.8

# Caching / streaming
redis>=5.0
hiredis>=2.2

# Viz & tracking
mlflow>=2.9
matplotlib>=3.7
plotly>=5.18
streamlit>=1.29
streamlit-folium>=0.17

# Dev / test
black>=23.0
ruff>=0.1
pytest>=7.4
pytest-asyncio>=0.21
hypothesis>=6.90
pip-tools>=7.3
"""

FASTAPI_MAIN = """\
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import os, time, asyncio, redis
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Traffic Prediction API", version="0.1.0")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
MODEL_GENERATION = int(os.getenv("MODEL_GENERATION", 1))

redis_client = None

class PredictionRequest(BaseModel):
    edge_ids: List[str]
    horizon: int = Field(15, ge=15, le=60)

class RouteRequest(BaseModel):
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    departure_time: Optional[datetime] = None

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    try:
        redis_client.ping()
    except Exception as e:
        raise RuntimeError(f"Redis not reachable: {e}")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        redis_client.close()

@app.get("/")
async def root():
    return {"name": "Traffic Prediction API", "version": "0.1.0", "model_generation": MODEL_GENERATION}

@app.get("/liveness")
async def liveness():
    ok = True
    try:
        redis_client.ping()
    except Exception:
        ok = False
    return JSONResponse(
        status_code=(status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE),
        content={"status": "healthy" if ok else "unhealthy", "timestamp": datetime.utcnow().isoformat()}
    )

@app.get("/readiness")
async def readiness():
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    # Redis check (set+get)
    try:
        t0 = time.time()
        k = f"edge:pred:test_edge:15:gen{MODEL_GENERATION}"
        redis_client.setex(k, 60, "55.5")
        v = redis_client.get(k)
        checks["redis_cache"] = v is not None
        details["redis_latency_ms"] = round((time.time()-t0)*1000, 2)
    except Exception as e:
        checks["redis_cache"] = False
        details["redis_error"] = str(e)

    # Fake inference 10 ms (non-blocking)
    t0 = time.time()
    await asyncio.sleep(0.01)
    details["inference_latency_ms"] = round((time.time()-t0)*1000, 2)
    checks["model_inference"] = True

    total = details["redis_latency_ms"] + details["inference_latency_ms"] if "redis_latency_ms" in details else 9999
    checks["latency_slo"] = total < 200
    details["total_latency_ms"] = total

    ready = all(checks.values())
    return JSONResponse(
        status_code=(status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE),
        content={"status": "ready" if ready else "not ready", "checks": checks, "details": details,
                 "timestamp": datetime.utcnow().isoformat()}
    )

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = []
    lines.append("# HELP api_up Always 1 if the API process is running")
    lines.append("# TYPE api_up gauge")
    lines.append("api_up 1")
    lines.append("# HELP model_generation Current model generation number")
    lines.append("# TYPE model_generation gauge")
    lines.append(f"model_generation {MODEL_GENERATION}")
    return "\\n".join(lines)

@app.post("/predict")
async def predict(req: PredictionRequest):
    return {"predictions": {e: 50.0 for e in req.edge_ids}, "horizon": req.horizon,
            "model_generation": MODEL_GENERATION, "prediction_timestamp": datetime.utcnow().isoformat()}

@app.post("/route")
async def route(req: RouteRequest):
    return {"route": {"polyline": "placeholder", "distance_km": 0.0,
                      "static_eta_minutes": 0.0, "dynamic_eta_minutes": 0.0, "confidence": 0.0},
            "origin": req.origin, "destination": req.destination,
            "departure_time": req.departure_time or datetime.utcnow(),
            "model_generation": MODEL_GENERATION}
"""

def write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(text, encoding="utf-8")

def main():
    # dirs
    for d in [
        "src/api",
        "src/data",
        "src/mapping",
        "src/utils",
        "configs",
        "data/raw/pems-bay",
        "data/raw/osm",
        "data/processed",
        "data/cache",
        "data/reports",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # files
    write(ROOT/".gitignore", GITIGNORE)
    write(ROOT/".env.example", ENV_EXAMPLE)
    write(ROOT/"requirements-gpu.txt", REQUIREMENTS_GPU)
    write(ROOT/"requirements.in", REQUIREMENTS_IN)
    write(ROOT/"src/api/main.py", FASTAPI_MAIN)
    write(ROOT/"configs/system.yaml",
          "project_name: traffic-prediction\nredis_host: localhost\nredis_port: 6379\n")

    print("✓ Project scaffolded. Next:\n"
          "1) python -m pip install -U pip pip-tools\n"
          "2) python -m pip install -r requirements-gpu.txt\n"
          "3) pip-compile requirements.in -o requirements.txt && python -m pip install -r requirements.txt\n"
          "4) uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    main()
