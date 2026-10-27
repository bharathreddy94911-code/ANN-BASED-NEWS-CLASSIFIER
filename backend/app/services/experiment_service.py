import uuid
from typing import Dict, Any, List, Optional
from app.services.training_service import training_service
from app.utils.persistence import load_experiments, get_experiment_by_id

class ExperimentService:
    @staticmethod
    def run_experiment(
        name: str,
        model_type: str = "cbow_mlp",
        optimizer: str = "sgd",
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.1,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pooling: str = "concat",
        cbow_window: int = 2,
        num_layers: int = 1,
        quick_mode: bool = True,
        subset_size: Optional[int] = 10000
    ) -> str:
        job_id = training_service.start_training_job(
            model_type=model_type,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
            cbow_window=cbow_window,
            num_layers=num_layers,
            quick_mode=quick_mode,
            subset_size=subset_size
        )
        return job_id

    @staticmethod
    def get_all_experiments() -> List[Dict[str, Any]]:
        return load_experiments()

    @staticmethod
    def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
        return get_experiment_by_id(experiment_id)

experiment_service = ExperimentService()
