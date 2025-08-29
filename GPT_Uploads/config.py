"""
Type-safe configuration schema using Pydantic
Ensures all configs are validated at startup
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from pathlib import Path


class DataConfig(BaseModel):
    """Data configuration schema"""
    name: str
    dataset: str
    
    # Paths
    raw_data: Path
    processed_data: Path
    adjacency_matrix: Path
    sensor_data: Path
    osm_extract: Path
    
    # Dataset characteristics
    num_sensors: int = Field(gt=0)
    time_steps_per_day: int = Field(gt=0)
    sample_rate_minutes: int = Field(gt=0)
    
    # Temporal settings
    lookback_window: int = Field(gt=0)
    prediction_horizons: List[int]
    
    # Data partitioning
    train_ratio: float = Field(gt=0, le=1)
    val_ratio: float = Field(gt=0, le=1)
    test_ratio: float = Field(gt=0, le=1)
    temporal_split: bool = True
    
    # Preprocessing
    normalize_method: Literal["z_score", "min_max"]
    fill_missing: Literal["forward_fill", "interpolate", "zero"]
    
    # Mapping configuration
    direction_aware: bool = True
    split_edges_at_sensors: bool = True
    interpolation_method: Literal["idw", "nearest", "linear"]
    max_interpolation_distance_km: float = Field(gt=0)
    
    @validator("test_ratio")
    def validate_ratios(cls, v, values):
        """Ensure train/val/test ratios sum to 1.0"""
        if "train_ratio" in values and "val_ratio" in values:
            total = values["train_ratio"] + values["val_ratio"] + v
            assert abs(total - 1.0) < 0.001, f"Ratios must sum to 1.0, got {total}"
        return v


class ModelConfig(BaseModel):
    """Model configuration schema"""
    name: str
    architecture: str
    
    # Architecture parameters
    hidden_dim: int = Field(gt=0)
    skip_dim: int = Field(gt=0)
    end_dim: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    
    # Training
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    epochs: int = Field(gt=0)
    gradient_clip_val: float = Field(gt=0)
    
    # Loss configuration
    loss_type: Literal["mae", "mse", "huber"]
    huber_delta: float = Field(gt=0)
    congestion_weighted: bool
    congestion_weight_factor: float = Field(gt=1)
    
    # GPU settings
    gpu_enabled: bool
    amp_enabled: bool
    cudnn_benchmark: bool
    torch_compile: bool
    
    # Export settings
    export_format: Literal["torchscript", "onnx"]
    quantization_enabled: bool
    quantization_type: Literal["dynamic_int8", "static_int8", "none"]


class PerformanceConfig(BaseModel):
    """Performance SLO configuration"""
    latency_p50_ms: int = Field(gt=0)
    latency_p95_ms: int = Field(gt=0)
    min_cache_hit_rate: float = Field(ge=0, le=1)
    max_prediction_age_seconds: int = Field(gt=0)


class SystemConfig(BaseModel):
    """Complete system configuration"""
    project_name: str
    experiment_name: str
    seed: int
    
    # Component configs
    data: DataConfig
    model: ModelConfig
    performance: PerformanceConfig
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_generation: int = 1
    
    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True


def load_config(config_path: str = "configs/config.yaml") -> SystemConfig:
    """
    Load and validate configuration from Hydra
    """
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")
        
    # Convert to dictionary and validate with Pydantic
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return SystemConfig(**config_dict)


def validate_environment():
    """
    Validate that the environment is properly configured
    """
    import torch
    import redis
    from pathlib import Path
    
    checks = []
    
    # Check CUDA availability
    if torch.cuda.is_available():
        checks.append(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        checks.append("✗ CUDA not available - will use CPU")
    
    # Check Redis connection
    try:
        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        checks.append("✓ Redis connection successful")
    except:
        checks.append("✗ Redis not available - please start Redis")
    
    # Check data directories
    data_path = Path("data")
    if data_path.exists():
        checks.append("✓ Data directory exists")
    else:
        checks.append("✗ Data directory missing")
    
    return checks


if __name__ == "__main__":
    # Test configuration loading
    print("Validating environment...")
    for check in validate_environment():
        print(check)
    
    print("\nLoading configuration...")
    try:
        config = load_config()
        print(f"✓ Configuration loaded: {config.project_name}")
        print(f"  Model: {config.model.name}")
        print(f"  Data: {config.data.name}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")