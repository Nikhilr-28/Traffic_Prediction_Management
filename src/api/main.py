from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import os, time, asyncio, redis
from datetime import datetime
from dotenv import load_dotenv
# --- at top imports ---
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional

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
        get_redis().ping()
    except Exception:
        ok = False
    return JSONResponse(
        status_code=(status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE),
        content={"status": "healthy" if ok else "unhealthy", "timestamp": datetime.utcnow().isoformat()}
    )

@app.get("/readiness")
async def readiness():
    checks, details = {}, {}
    # Redis set/get check
    try:
        t0 = time.time()
        k = f"edge:pred:test_edge:15:gen{MODEL_GENERATION}"
        r = get_redis()
        r.setex(k, 60, "55.5")
        v = r.get(k)
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
    return "\n".join(lines)

@app.post("/predict")
async def predict(req: PredictionRequest):
    r = get_redis()
    vals = {}
    pipe = r.pipeline()
    # try horizon prediction first
    for e in req.edge_ids:
        pipe.get(f"edge:pred:{e}:{req.horizon}")
    pred_res = pipe.execute()

    # any missing? fetch current speed as fallback
    missing = [i for i, v in enumerate(pred_res) if v is None]
    fallback = {}
    if missing:
        pipe = r.pipeline()
        for i in missing:
            pipe.get(f"edge:speed:{req.edge_ids[i]}")
        got = pipe.execute()
        fallback = {req.edge_ids[i]: (float(got[j]) if got[j] is not None else None)
                    for j, i in enumerate(missing)}

    for e, v in zip(req.edge_ids, pred_res):
        vals[e] = float(v) if v is not None else fallback.get(e, None)

    return {
        "predictions": vals,
        "horizon": req.horizon,
        "model_generation": MODEL_GENERATION,
        "prediction_timestamp": datetime.utcnow().isoformat()
    }



@app.post("/route")
async def route(req: RouteRequest):
    return {"route": {"polyline": "placeholder", "distance_km": 0.0,
                      "static_eta_minutes": 0.0, "dynamic_eta_minutes": 0.0, "confidence": 0.0},
            "origin": req.origin, "destination": req.destination,
            "departure_time": req.departure_time or datetime.utcnow(),
            "model_generation": MODEL_GENERATION}
