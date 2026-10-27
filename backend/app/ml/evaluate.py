import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader, TensorDataset

from app.config import DEVICE, LABEL_NAMES
from app.ml.preprocessing import tokenize
from app.ml.vocabulary import Vocabulary
from app.ml.classifier import CBOWMLPClassifier
from app.ml.baseline import BaselineModel
from app.ml.metrics import compute_all_metrics

class EvaluationPipeline:
    @staticmethod
    def evaluate_cbow_mlp(
        model: CBOWMLPClassifier,
        vocab: Vocabulary,
        test_texts: List[str],
        test_labels: List[int],
        batch_size: int = 128,
        max_seq_len: int = 64
    ) -> Dict[str, Any]:
        X_test = [vocab.encode(tokenize(t), max_len=max_seq_len) for t in test_texts]
        t_X_test = torch.tensor(X_test, dtype=torch.long)
        t_y_test = torch.tensor(test_labels, dtype=torch.long)
        
        loader = DataLoader(TensorDataset(t_X_test, t_y_test), batch_size=batch_size, shuffle=False)
        
        model.eval()
        model.to(DEVICE)
        
        preds = []
        start_time = time.time()
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(DEVICE)
                logits = model(bx)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(batch_preds)
        total_time = time.time() - start_time
        avg_latency_ms = (total_time / len(test_texts)) * 1000.0

        metrics = compute_all_metrics(test_labels, preds)
        metrics['inference_latency_ms'] = round(avg_latency_ms, 3)
        metrics['total_samples'] = len(test_texts)
        
        # Calculate parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['param_count'] = param_count

        # Identify misclassified examples
        misclassified = []
        for i, (pred, actual) in enumerate(zip(preds, test_labels)):
            if pred != actual:
                misclassified.append({
                    "id": i,
                    "text": test_texts[i],
                    "predicted_label": pred,
                    "predicted_category": LABEL_NAMES[pred],
                    "actual_label": actual,
                    "actual_category": LABEL_NAMES[actual]
                })

        metrics['misclassified'] = misclassified
        return metrics

    @staticmethod
    def evaluate_baseline(
        baseline: BaselineModel,
        test_texts: List[str],
        test_labels: List[int],
        batch_size: int = 128
    ) -> Dict[str, Any]:
        X_test_tensor = baseline.transform(test_texts)
        y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
        
        model = baseline.model
        model.eval()
        model.to(DEVICE)
        
        preds = []
        start_time = time.time()
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(DEVICE)
                logits = model(bx)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(batch_preds)
        total_time = time.time() - start_time
        avg_latency_ms = (total_time / len(test_texts)) * 1000.0

        metrics = compute_all_metrics(test_labels, preds)
        metrics['inference_latency_ms'] = round(avg_latency_ms, 3)
        metrics['total_samples'] = len(test_texts)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['param_count'] = param_count

        misclassified = []
        for i, (pred, actual) in enumerate(zip(preds, test_labels)):
            if pred != actual:
                misclassified.append({
                    "id": i,
                    "text": test_texts[i],
                    "predicted_label": pred,
                    "predicted_category": LABEL_NAMES[pred],
                    "actual_label": actual,
                    "actual_category": LABEL_NAMES[actual]
                })

        metrics['misclassified'] = misclassified
        return metrics
