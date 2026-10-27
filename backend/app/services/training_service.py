import threading
import uuid
import time
import datetime
import logging
from typing import Dict, Any, Optional
import torch

from app.config import MODELS_DIR, VOCAB_DIR, DATASET_NAME
from app.ml.data_loader import dataset_manager
from app.ml.preprocessing import tokenize
from app.ml.vocabulary import Vocabulary
from app.ml.train import TrainingPipeline
from app.ml.evaluate import EvaluationPipeline
from app.services.model_service import model_service
from app.utils.persistence import save_evaluation, save_experiment

logger = logging.getLogger(__name__)

class TrainingService:
    _instance: Optional['TrainingService'] = None
    _lock = threading.Lock()
    _jobs: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingService, cls).__new__(cls)
        return cls._instance

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def is_busy(self) -> bool:
        return self._lock.locked()

    def start_training_job(
        self,
        model_type: str = "cbow_mlp", # "cbow_mlp" or "baseline"
        optimizer: str = "sgd",
        epochs: int = 6,
        batch_size: int = 64,
        lr: float = 0.3,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean",
        cbow_window: int = 2,
        num_layers: int = 1,
        quick_mode: bool = False,
        subset_size: Optional[int] = None
    ) -> str:
        if self._lock.locked():
            raise RuntimeError("A training job is already running. Please wait for it to finish.")

        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "model_type": model_type,
            "current_epoch": 0,
            "total_epochs": epochs,
            "train_loss": None,
            "val_loss": None,
            "val_accuracy": None,
            "elapsed_seconds": 0,
            "error": None,
            "history": [],
            "message": "Initializing training job...",
            "created_at": datetime.datetime.now().isoformat()
        }

        # Launch background thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job_id, model_type, optimizer, epochs, batch_size, lr, embed_dim, hidden_dim, dropout, pooling, cbow_window, num_layers, quick_mode, subset_size)
        )
        thread.daemon = True
        thread.start()

        return job_id

    def _run_training(
        self,
        job_id: str,
        model_type: str,
        optimizer: str,
        epochs: int,
        batch_size: int,
        lr: float,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        pooling: str,
        cbow_window: int,
        num_layers: int,
        quick_mode: bool,
        subset_size: Optional[int]
    ):
        with self._lock:
            start_time = time.time()
            job = self._jobs[job_id]
            job["status"] = "running"
            
            try:
                # 1. Load dataset
                job["message"] = "Loading AG News dataset..."
                ds = dataset_manager.load()
                
                # Shuffle train set reproducibly before taking split to ensure class balance
                full_train = ds["train"].shuffle(seed=42)
                n_total = len(full_train)
                
                if quick_mode:
                    effective_n = subset_size or 12000
                    n_train = int(effective_n * 0.85)
                    n_val = effective_n - n_train
                    train_data = full_train.select(range(n_train))
                    val_data = full_train.select(range(n_train, effective_n))
                else:
                    n_train = int(n_total * 0.85)
                    train_data = full_train.select(range(n_train))
                    val_data = full_train.select(range(n_train, n_total))
                    
                train_texts = [row["text"] for row in train_data]
                train_labels = [int(row["label"]) for row in train_data]
                val_texts = [row["text"] for row in val_data]
                val_labels = [int(row["label"]) for row in val_data]
                
                def progress_cb(ep_data):
                    job["current_epoch"] = ep_data["epoch"]
                    job["train_loss"] = ep_data["train_loss"]
                    job["val_loss"] = ep_data["val_loss"]
                    job["val_accuracy"] = ep_data["val_accuracy"]
                    job["elapsed_seconds"] = round(time.time() - start_time, 1)
                    job["history"].append(ep_data)
                    job["message"] = f"Epoch {ep_data['epoch']}/{epochs} completed."
                
                if model_type == "baseline":
                    job["message"] = "Training TF-IDF + MLP Baseline..."
                    baseline_model, history = TrainingPipeline.train_baseline(
                        train_texts=train_texts,
                        train_labels=train_labels,
                        val_texts=val_texts,
                        val_labels=val_labels,
                        max_features=5000,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        optimizer_name=optimizer,
                        progress_callback=progress_cb
                    )
                    
                    # Evaluate on test set
                    job["message"] = "Evaluating baseline on test split..."
                    test_ds = ds["test"]
                    test_texts = [row["text"] for row in test_ds]
                    test_labels = [int(row["label"]) for row in test_ds]
                    test_results = EvaluationPipeline.evaluate_baseline(baseline_model, test_texts, test_labels)
                    
                    save_evaluation({
                        "job_id": job_id,
                        "model_type": "TF-IDF + MLP Baseline",
                        "quick_mode": quick_mode,
                        "evaluated_at": datetime.datetime.now().isoformat(),
                        "metrics": test_results
                    })

                else:
                    # CBOW + MLP Proposed Architecture
                    # Build vocabulary from training text ONLY
                    job["message"] = "Building vocabulary from training text..."
                    tokenized_train = [tokenize(t) for t in train_texts]
                    vocab = Vocabulary(min_freq=2, max_size=25000)
                    vocab.build_from_tokens(tokenized_train)
                    vocab.save(str(VOCAB_DIR / "vocab.json"))
                    
                    # Step 1: Pre-train CBOW Embeddings
                    def cbow_status_cb(msg):
                        job["message"] = msg
                        job["elapsed_seconds"] = round(time.time() - start_time, 1)
                        
                    cbow_embed_weights = TrainingPipeline.train_cbow_embeddings(
                        train_texts=train_texts,
                        vocab=vocab,
                        embed_dim=embed_dim,
                        window_size=cbow_window,
                        epochs=2 if quick_mode else 3,
                        batch_size=256,
                        max_pairs=50000 if quick_mode else 150000,
                        progress_callback=cbow_status_cb
                    )
                    
                    # Step 2: Train CBOW + MLP Classifier
                    job["message"] = "Training CBOW + MLP Classifier..."
                    model, history = TrainingPipeline.train_classifier(
                        train_texts=train_texts,
                        train_labels=train_labels,
                        val_texts=val_texts,
                        val_labels=val_labels,
                        vocab=vocab,
                        pretrained_embeddings=cbow_embed_weights,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        pooling=pooling,
                        lr=lr,
                        batch_size=batch_size,
                        epochs=epochs,
                        optimizer_name=optimizer,
                        progress_callback=progress_cb
                    )
                    
                    # Save best checkpoint
                    model_path = MODELS_DIR / "best_cbow_mlp.pt"
                    torch.save(model.state_dict(), model_path)
                    
                    import json
                    with open(MODELS_DIR / "best_cbow_mlp_meta.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "version": f"cbow-mlp-{job_id}",
                            "embed_dim": embed_dim,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "dropout": dropout,
                            "pooling": pooling,
                            "optimizer": optimizer,
                            "quick_mode": quick_mode,
                            "trained_at": datetime.datetime.now().isoformat()
                        }, f)
                        
                    # Reload active predictor in service
                    model_service.load_active_model()
                    
                    # Official Test Split Evaluation
                    job["message"] = "Running evaluation on official test split..."
                    test_ds = ds["test"]
                    test_texts = [row["text"] for row in test_ds]
                    test_labels = [int(row["label"]) for row in test_ds]
                    
                    test_results = EvaluationPipeline.evaluate_cbow_mlp(
                        model=model,
                        vocab=vocab,
                        test_texts=test_texts,
                        test_labels=test_labels
                    )
                    
                    save_evaluation({
                        "job_id": job_id,
                        "model_type": "CBOW Embeddings + MLP",
                        "quick_mode": quick_mode,
                        "evaluated_at": datetime.datetime.now().isoformat(),
                        "metrics": test_results
                    })

                # Register experiment record
                duration = round(time.time() - start_time, 2)
                exp_record = {
                    "id": job_id,
                    "name": f"{model_type.upper()} ({optimizer.upper()}, lr={lr}, {pooling})",
                    "status": "completed",
                    "config": {
                        "model_type": model_type,
                        "optimizer": optimizer,
                        "lr": lr,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "embed_dim": embed_dim,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "pooling": pooling,
                        "cbow_window": cbow_window,
                        "quick_mode": quick_mode
                    },
                    "results": {
                        "training_time": duration,
                        "val_metrics": {
                            "accuracy": job["val_accuracy"],
                            "loss": job["val_loss"],
                            "macro_f1": job["history"][-1].get("val_macro_f1") if job["history"] else None
                        },
                        "test_metrics": {
                            "accuracy": test_results.get("accuracy"),
                            "macro_f1": test_results.get("macro_f1"),
                            "latency_ms": test_results.get("inference_latency_ms")
                        }
                    },
                    "created_at": datetime.datetime.now().isoformat()
                }
                save_experiment(exp_record)

                job["status"] = "completed"
                job["message"] = f"Training successfully completed in {duration}s."
                job["elapsed_seconds"] = duration

            except Exception as e:
                logger.exception("Training error")
                job["status"] = "failed"
                job["error"] = str(e)
                job["message"] = f"Error: {e}"
                job["elapsed_seconds"] = round(time.time() - start_time, 1)

training_service = TrainingService()
