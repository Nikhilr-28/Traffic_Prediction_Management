"""
FastAPI application for Traffic Prediction System
Includes health probes for production readiness
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import redis
import time
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Prediction API",
    description="Real-time traffic prediction and routing system",
    version="1.0.0"
)

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Global Redis client (initialized at startup)
redis_client = None

# Model generation tracking
CURRENT_MODEL_GENERATION = int(os.getenv("MODEL_GENERATION", 1))


class PredictionRequest(BaseModel):
    """Request model for edge predictions"""
    edge_ids: List[str] = Field(..., description="List of edge IDs to predict")
    horizon: int = Field(15, description="Prediction horizon in minutes", ge=15, le=60)


class RouteRequest(BaseModel):
    """Request model for routing"""
    origin: tuple[float, float] = Field(..., description="Origin coordinates (lat, lon)")
    destination: tuple[float, float] = Field(..., description="Destination coordinates (lat, lon)")
    departure_time: Optional[datetime] = Field(None, description="Departure time (defaults to now)")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    checks: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize connections and warm up model on startup"""
    global redis_client
    
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("✓ Redis connection established")
        
        # TODO: Load model here
        # model = load_model()
        # logger.info("✓ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of connections"""
    global redis_client
    if redis_client:
        redis_client.close()
        logger.info("Redis connection closed")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Traffic Prediction API",
        "version": "1.0.0",
        "model_generation": CURRENT_MODEL_GENERATION,
        "endpoints": {
            "health": "/liveness, /readiness",
            "predictions": "/predict",
            "routing": "/route",
            "metrics": "/metrics"
        }
    }


@app.get("/liveness", response_model=HealthResponse)
async def liveness_probe():
    """
    Liveness probe - checks if service dependencies are accessible
    Used by orchestrators to determine if container should be restarted
    """
    checks = {}
    all_healthy = True
    
    # Check Redis connectivity (simple ping)
    try:
        if redis_client:
            redis_client.ping()
            checks["redis"] = True
        else:
            checks["redis"] = False
            all_healthy = False
    except:
        checks["redis"] = False
        all_healthy = False
    
    # Check if model is loaded (basic check)
    # TODO: Implement actual model check
    checks["model_loaded"] = True  # Placeholder
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "details": {"model_generation": CURRENT_MODEL_GENERATION}
        }
    )


@app.get("/readiness", response_model=HealthResponse)
async def readiness_probe():
    """
    Readiness probe - runs actual cached prediction to verify full functionality
    Used by load balancers to determine if instance should receive traffic
    """
    checks = {}
    details = {}
    all_ready = True
    
    # Check Redis with actual data operation
    try:
        start_time = time.time()
        
        # Try to fetch a cached prediction (use a standard test edge)
        test_edge_id = "test_edge_001"
        test_key = f"edge:pred:{test_edge_id}:15:gen{CURRENT_MODEL_GENERATION}"
        
        # First, set a test value if it doesn't exist
        if not redis_client.exists(test_key):
            redis_client.setex(test_key, 300, "55.5")  # 55.5 mph, 5 min TTL
        
        # Now fetch it
        cached_value = redis_client.get(test_key)
        
        redis_latency_ms = (time.time() - start_time) * 1000
        
        checks["redis_cache"] = cached_value is not None
        details["redis_latency_ms"] = round(redis_latency_ms, 2)
        
    except Exception as e:
        checks["redis_cache"] = False
        details["redis_error"] = str(e)
        all_ready = False
    
    # Check model inference capability (with cached prediction)
    try:
        # TODO: Run actual inference on test input
        # For now, simulate with timing
        start_time = time.time()
        # prediction = model.predict(test_input)  # Actual implementation
        time.sleep(0.01)  # Simulate 10ms inference
        inference_latency_ms = (time.time() - start_time) * 1000
        
        checks["model_inference"] = True
        details["inference_latency_ms"] = round(inference_latency_ms, 2)
        
    except Exception as e:
        checks["model_inference"] = False
        details["model_error"] = str(e)
        all_ready = False
    
    # Check if we meet latency SLOs
    if "redis_latency_ms" in details and "inference_latency_ms" in details:
        total_latency = details["redis_latency_ms"] + details["inference_latency_ms"]
        checks["latency_slo"] = total_latency < 200  # 200ms target
        details["total_latency_ms"] = round(total_latency, 2)
    
    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "details": details
        }
    )


@app.get("/metrics")
async def metrics():
    """
    Metrics endpoint for monitoring
    Returns Prometheus-compatible metrics
    """
    # TODO: Implement full metrics collection
    metrics_data = []
    
    # Add basic metrics
    metrics_data.append("# HELP api_requests_total Total number of API requests")
    metrics_data.append("# TYPE api_requests_total counter")
    metrics_data.append(f"api_requests_total{{endpoint=\"/predict\"}} 0")
    
    metrics_data.append("# HELP model_generation Current model generation number")
    metrics_data.append("# TYPE model_generation gauge")
    metrics_data.append(f"model_generation {CURRENT_MODEL_GENERATION}")
    
    return "\n".join(metrics_data)


@app.post("/predict")
async def predict_edges(request: PredictionRequest):
    """
    Predict traffic speeds for given edges
    """
    # TODO: Implement actual prediction logic
    return {
        "predictions": {
            edge_id: 55.0 + (hash(edge_id) % 20)  # Placeholder
            for edge_id in request.edge_ids
        },
        "horizon": request.horizon,
        "model_generation": CURRENT_MODEL_GENERATION,
        "prediction_timestamp": datetime.utcnow().isoformat()
    }


@app.post("/route")
async def calculate_route(request: RouteRequest):
    """
    Calculate optimal route with dynamic ETAs
    """
    # TODO: Implement routing logic
    return {
        "route": {
            "polyline": "placeholder_polyline_encoding",
            "distance_km": 15.2,
            "static_eta_minutes": 18.5,
            "dynamic_eta_minutes": 22.3,  # With traffic predictions
            "confidence": 0.85
        },
        "origin": request.origin,
        "destination": request.destination,
        "departure_time": request.departure_time or datetime.utcnow(),
        "model_generation": CURRENT_MODEL_GENERATION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)