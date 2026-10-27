import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.config import RESULTS_DIR

EXPERIMENTS_FILE = RESULTS_DIR / "experiments.json"
EVALUATION_FILE = RESULTS_DIR / "latest_evaluation.json"

def save_experiment(experiment_data: Dict[str, Any]):
    experiments = load_experiments()
    existing = False
    for i, exp in enumerate(experiments):
        if exp.get('id') == experiment_data.get('id'):
            experiments[i] = experiment_data
            existing = True
            break
    if not existing:
        experiments.append(experiment_data)
        
    with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiments, f, indent=2)

def load_experiments() -> List[Dict[str, Any]]:
    if not EXPERIMENTS_FILE.exists():
        return []
    try:
        with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def get_experiment_by_id(exp_id: str) -> Optional[Dict[str, Any]]:
    experiments = load_experiments()
    for exp in experiments:
        if exp.get('id') == exp_id:
            return exp
    return None

def save_evaluation(eval_data: Dict[str, Any]):
    with open(EVALUATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2)

def load_evaluation() -> Optional[Dict[str, Any]]:
    if not EVALUATION_FILE.exists():
        return None
    try:
        with open(EVALUATION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def export_experiments_csv() -> str:
    experiments = load_experiments()
    if not experiments:
        return ""
    
    headers = [
        'id', 'name', 'model_type', 'optimizer', 'lr', 'batch_size',
        'embed_dim', 'hidden_dim', 'dropout', 'pooling', 'status', 'val_accuracy',
        'val_loss', 'val_f1', 'test_accuracy', 'test_f1', 'training_time', 'created_at'
    ]
    
    output_lines = [','.join(headers)]
    for exp in experiments:
        cfg = exp.get('config', {})
        res = exp.get('results', {})
        val_m = res.get('val_metrics', {})
        test_m = res.get('test_metrics', {})
        
        row = [
            str(exp.get('id', '')),
            str(exp.get('name', '')).replace(',', ' '),
            str(cfg.get('model_type', '')),
            str(cfg.get('optimizer', '')),
            str(cfg.get('lr', '')),
            str(cfg.get('batch_size', '')),
            str(cfg.get('embed_dim', '')),
            str(cfg.get('hidden_dim', '')),
            str(cfg.get('dropout', '')),
            str(cfg.get('pooling', 'mean')),
            str(exp.get('status', '')),
            str(val_m.get('accuracy', '')),
            str(val_m.get('loss', '')),
            str(val_m.get('macro_f1', '')),
            str(test_m.get('accuracy', '')),
            str(test_m.get('macro_f1', '')),
            str(res.get('training_time', '')),
            str(exp.get('created_at', ''))
        ]
        output_lines.append(','.join(row))
        
    return '\n'.join(output_lines)
