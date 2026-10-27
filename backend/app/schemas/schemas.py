from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any

# Prediction Schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, description="News text or headline to classify")

class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    model_version: str

# Training Schemas
class TrainingRequest(BaseModel):
    dataset_name: str = "Mikami063/ag_news"
    model_type: str = "cbow_mlp" # cbow_mlp or baseline
    optimizer: str = "sgd" # sgd or adam
    epochs: int = Field(6, ge=1, le=50)
    batch_size: int = Field(64, ge=8, le=512)
    lr: float = Field(0.3, gt=0, le=1.0)
    embed_dim: int = Field(100, ge=16, le=512)
    hidden_dim: int = Field(128, ge=16, le=1024)
    dropout: float = Field(0.2, ge=0.0, le=0.9)
    pooling: str = Field("mean", description="Pooling method: concat, mean, or max")
    cbow_window: int = Field(2, ge=1, le=5)
    num_layers: int = Field(1, ge=1, le=3)
    quick_mode: bool = False
    subset_size: Optional[int] = 15000

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    model_type: str
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    elapsed_seconds: float
    message: str
    error: Optional[str] = None
    history: List[Dict[str, Any]] = []

# Experiment Schemas
class ExperimentRequest(BaseModel):
    name: str = "Experiment Trial"
    model_type: str = "cbow_mlp"
    optimizer: str = "sgd"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    embed_dim: int = 100
    hidden_dim: int = 128
    dropout: float = 0.3
    pooling: str = "concat"
    cbow_window: int = 2
    num_layers: int = 1
    quick_mode: bool = True
    subset_size: Optional[int] = 10000
