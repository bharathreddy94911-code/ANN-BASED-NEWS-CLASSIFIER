import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List
from app.config import LABEL_NAMES

def compute_all_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    macro_p = float(precision_score(y_t, y_p, average='macro', zero_division=0))
    macro_r = float(recall_score(y_t, y_p, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average='macro', zero_division=0))

    weighted_p = float(precision_score(y_t, y_p, average='weighted', zero_division=0))
    weighted_r = float(recall_score(y_t, y_p, average='weighted', zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average='weighted', zero_division=0))

    # Per-class metrics
    per_class_p = precision_score(y_t, y_p, average=None, zero_division=0).tolist()
    per_class_r = recall_score(y_t, y_p, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_t, y_p, average=None, zero_division=0).tolist()

    # Per-class report breakdown
    per_class_report = {}
    for idx, name in enumerate(LABEL_NAMES):
        support_count = int(np.sum(y_t == idx))
        per_class_report[name] = {
            'precision': round(float(per_class_p[idx]), 4),
            'recall': round(float(per_class_r[idx]), 4),
            'f1_score': round(float(per_class_f1[idx]), 4),
            'support': support_count
        }

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2, 3]).tolist()

    return {
        'accuracy': round(acc, 4),
        'macro_precision': round(macro_p, 4),
        'macro_recall': round(macro_r, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_precision': round(weighted_p, 4),
        'weighted_recall': round(weighted_r, 4),
        'weighted_f1': round(weighted_f1, 4),
        'per_class': per_class_report,
        'confusion_matrix': cm,
        'labels': LABEL_NAMES
    }
