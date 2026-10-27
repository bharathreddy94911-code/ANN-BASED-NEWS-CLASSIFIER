from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Response
from typing import Optional, List
from app.ml.data_loader import dataset_manager
from app.services.model_service import model_service
from app.services.training_service import training_service
from app.services.experiment_service import experiment_service
from app.utils.persistence import load_evaluation, export_experiments_csv
from app.schemas.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
    ExperimentRequest
)

api_router = APIRouter(prefix="/api")

# --- Health Endpoint ---
@api_router.get("/health")
def health_check():
    dataset_info = dataset_manager.get_info()
    model_avail = model_service.is_model_available()
    return {
        "status": "healthy",
        "dataset_status": dataset_info["status"],
        "model_available": model_avail,
        "is_training": training_service.is_busy(),
        "model_info": model_service.get_info()
    }

# --- Dataset Endpoints ---
@api_router.get("/dataset/info")
def get_dataset_info():
    try:
        return dataset_manager.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dataset/examples")
def get_dataset_examples(
    split: str = Query("train", enum=["train", "test"]),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    try:
        return dataset_manager.get_examples(split=split, page=page, page_size=page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dataset/distribution")
def get_dataset_distribution():
    try:
        return dataset_manager.get_distribution()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Prediction Endpoint ---
@api_router.post("/predict", response_model=PredictionResponse)
def predict_news(request: PredictionRequest):
    try:
        result = model_service.predict(request.text)
        return result
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Training Endpoints ---
@api_router.post("/train", response_model=TrainingResponse)
def trigger_training(req: TrainingRequest):
    try:
        job_id = training_service.start_training_job(
            model_type=req.model_type,
            optimizer=req.optimizer,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            embed_dim=req.embed_dim,
            hidden_dim=req.hidden_dim,
            dropout=req.dropout,
            pooling=req.pooling,
            cbow_window=req.cbow_window,
            num_layers=req.num_layers,
            quick_mode=req.quick_mode,
            subset_size=req.subset_size
        )
        return {"job_id": job_id, "status": "started", "message": "Training job launched in background."}
    except RuntimeError as re:
        raise HTTPException(status_code=409, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/train/status/{job_id}", response_model=TrainingStatusResponse)
def get_training_status(job_id: str):
    job = training_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job ID not found.")
    return job

# --- Experiment Endpoints ---
@api_router.post("/experiments")
def run_experiment(req: ExperimentRequest):
    try:
        job_id = experiment_service.run_experiment(
            name=req.name,
            model_type=req.model_type,
            optimizer=req.optimizer,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            embed_dim=req.embed_dim,
            hidden_dim=req.hidden_dim,
            dropout=req.dropout,
            pooling=req.pooling,
            cbow_window=req.cbow_window,
            num_layers=req.num_layers,
            quick_mode=req.quick_mode,
            subset_size=req.subset_size
        )
        return {"job_id": job_id, "status": "started", "message": f"Experiment '{req.name}' is running."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/experiments")
def list_experiments():
    return experiment_service.get_all_experiments()

@api_router.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: str):
    exp = experiment_service.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return exp

# --- Evaluation Endpoints ---
@api_router.get("/evaluate")
def get_latest_evaluation():
    eval_data = load_evaluation()
    if not eval_data:
        raise HTTPException(status_code=404, detail="No evaluation results found. Please train and evaluate a model first.")
    return eval_data

@api_router.get("/evaluate/confusion-matrix")
def get_confusion_matrix():
    eval_data = load_evaluation()
    if not eval_data or "metrics" not in eval_data:
        raise HTTPException(status_code=404, detail="Evaluation results not available.")
    return {
        "confusion_matrix": eval_data["metrics"].get("confusion_matrix"),
        "labels": eval_data["metrics"].get("labels")
    }

@api_router.get("/evaluate/report")
def get_classification_report():
    eval_data = load_evaluation()
    if not eval_data or "metrics" not in eval_data:
        raise HTTPException(status_code=404, detail="Evaluation results not available.")
    return eval_data["metrics"].get("per_class")

@api_router.get("/evaluate/errors")
def get_misclassified_examples(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    eval_data = load_evaluation()
    if not eval_data or "metrics" not in eval_data:
        raise HTTPException(status_code=404, detail="Evaluation results not available.")
    
    errors = eval_data["metrics"].get("misclassified", [])
    total = len(errors)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": errors[start_idx:end_idx]
    }

@api_router.get("/results/export")
def export_results():
    csv_data = export_experiments_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiments_results.csv"}
    )
