# src/api/main.py
from __future__ import annotations
import os, time, asyncio, redis
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="Traffic Prediction API", version="0.1.0")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
MODEL_GENERATION = int(os.getenv("MODEL_GENERATION", 1))

redis_client: Optional[redis.Redis] = None

def get_redis() -> redis.Redis:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return redis_client

# ---------- Models ----------
class PredictionRequest(BaseModel):
    edge_ids: List[str]
    horizon: int = Field(15, ge=15, le=60)

class RouteRequest(BaseModel):
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    departure_time: Optional[datetime] = None

# ---------- Shared metrics helpers ----------
def _build_metrics_dict() -> Dict[str, Any]:
    # Use epoch seconds for "now" to keep things simple and unambiguous
    now_epoch = time.time()
    last_ts_epoch: Optional[float] = None
    age: Optional[float] = None

    # Parse speeds:last_ts as EPOCH first, then ISO-8601 fallback
    try:
        raw = get_redis().get("speeds:last_ts")
        if raw:
            try:
                # Ensure raw is string before converting to float
                last_ts_epoch = float(str(raw))  # Convert to str first to satisfy type checker
            except (TypeError, ValueError):
                # If it's not a float, try parsing as ISO timestamp
                dt_obj = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                last_ts_epoch = dt_obj.timestamp()
    except Exception:
        pass

    if last_ts_epoch is not None:
        age = max(0.0, now_epoch - last_ts_epoch)

    # Count prediction keys robustly (sync-safe)
    keys_pred_count = 0
    try:
        r = get_redis()
        # scan_iter returns an iterator of strings when decode_responses=True
        for _ in r.scan_iter(match="edge:pred:*", count=1000):
            keys_pred_count += 1
    except Exception:
        pass

    return {
        "now": now_epoch,
        "speeds_last_ts": last_ts_epoch,
        "prediction_age_seconds": age,
        "keys_pred_count": keys_pred_count,
        "model_generation": MODEL_GENERATION,
    }

def _metrics_as_prometheus(m: Dict[str, Any]) -> str:
    age = m.get("prediction_age_seconds")
    age_str = "NaN" if age is None else f"{float(age)}"
    return "\n".join([
        "# HELP api_up Always 1 if the API process is running",
        "# TYPE api_up gauge",
        "api_up 1",
        "# HELP model_generation Current model generation number",
        "# TYPE model_generation gauge",
        f"model_generation {m.get('model_generation', 0)}",
        "# HELP prediction_age_seconds Seconds since last replay tick",
        "# TYPE prediction_age_seconds gauge",
        f"prediction_age_seconds {age_str}",
        "# HELP keys_pred_count Count of cached prediction keys",
        "# TYPE keys_pred_count gauge",
        f"keys_pred_count {m.get('keys_pred_count', 0)}",
    ])

# ---------- Endpoints ----------
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
        get_redis().ping()
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
    # Redis set/get
    try:
        t0 = time.time()
        k = f"edge:pred:test_edge:15:gen{MODEL_GENERATION}"
        r = get_redis()
        r.setex(k, 60, "55.5")
        v = r.get(k)
        checks["redis_cache"] = v is not None
        details["redis_latency_ms"] = round((time.time() - t0) * 1000, 2)
    except Exception as e:
        checks["redis_cache"] = False
        details["redis_error"] = str(e)

    # Fake inference sleep (placeholder)
    t0 = time.time()
    await asyncio.sleep(0.01)
    details["inference_latency_ms"] = round((time.time() - t0) * 1000, 2)
    checks["model_inference"] = True

    total = details.get("redis_latency_ms", 9999) + details["inference_latency_ms"]
    checks["latency_slo"] = total < 200
    details["total_latency_ms"] = total

    ready = all(checks.values())
    return JSONResponse(
        status_code=(status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE),
        content={"status": "ready" if ready else "not ready", "checks": checks, "details": details,
                 "timestamp": datetime.utcnow().isoformat()}
    )

@app.get("/healthz")
async def healthz(threshold: float = 5.0):
    m = _build_metrics_dict()
    age = m.get("prediction_age_seconds")
    bad = (age is None) or (float(age) > threshold)
    if bad:
        raise HTTPException(status_code=503, detail={"prediction_age_seconds": age})
    return {"ok": True, "prediction_age_seconds": age}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return _metrics_as_prometheus(_build_metrics_dict())

@app.post("/predict")
async def predict(req: PredictionRequest):
    r = get_redis()
    vals: Dict[str, Optional[float]] = {}
    pipe = r.pipeline()
    for e in req.edge_ids:
        pipe.get(f"edge:pred:{e}:{req.horizon}")
    pred_res = pipe.execute()

    missing = [i for i, v in enumerate(pred_res) if v is None]
    if missing:
        pipe = r.pipeline()
        for i in missing:
            pipe.get(f"edge:speed:{req.edge_ids[i]}")
        got = pipe.execute()
        for j, i in enumerate(missing):
            vals[req.edge_ids[i]] = float(got[j]) if got[j] is not None else None

    for e, v in zip(req.edge_ids, pred_res):
        if v is not None:
            vals[e] = float(v)

    return {
        "predictions": vals,
        "horizon": req.horizon,
        "model_generation": MODEL_GENERATION,
        "prediction_timestamp": datetime.utcnow().isoformat()
    }

@app.post("/route")
async def route(req: RouteRequest):
    # Stub; wire Valhalla later
    return {"route": {"polyline": "placeholder", "distance_km": 0.0,
                      "static_eta_minutes": 0.0, "dynamic_eta_minutes": 0.0, "confidence": 0.0},
            "origin": req.origin, "destination": req.destination,
            "departure_time": (req.departure_time or datetime.utcnow()),
            "model_generation": MODEL_GENERATION}
