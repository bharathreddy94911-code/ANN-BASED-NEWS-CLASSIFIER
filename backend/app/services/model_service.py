import torch
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from app.config import MODELS_DIR, VOCAB_DIR, DEVICE
from app.ml.vocabulary import Vocabulary
from app.ml.classifier import CBOWMLPClassifier
from app.ml.predict import Predictor

logger = logging.getLogger(__name__)

class ModelService:
    _instance: Optional['ModelService'] = None
    _predictor: Optional[Predictor] = None
    _model_info: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance

    def load_active_model(self) -> bool:
        model_path = MODELS_DIR / "best_cbow_mlp.pt"
        vocab_path = VOCAB_DIR / "vocab.json"
        meta_path = MODELS_DIR / "best_cbow_mlp_meta.json"

        if not model_path.exists() or not vocab_path.exists():
            return False

        try:
            vocab = Vocabulary.load(str(vocab_path))
            meta = {}
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

            embed_dim = meta.get("embed_dim", 100)
            hidden_dim = meta.get("hidden_dim", 128)
            num_layers = meta.get("num_layers", 1)
            dropout = meta.get("dropout", 0.3)
            pooling = meta.get("pooling", "mean")

            model = CBOWMLPClassifier(
                vocab_size=len(vocab),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=4,
                dropout=dropout,
                num_layers=num_layers,
                pooling=pooling
            )
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)

            self._predictor = Predictor(model, vocab)
            self._model_info = {
                "version": meta.get("version", "v1.0"),
                "architecture": "CBOW Embedding + MLP",
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "optimizer": meta.get("optimizer", "SGD (Mini-Batch)"),
                "trained_at": meta.get("trained_at", "unknown")
            }
            return True
        except Exception as e:
            logger.error(f"Error loading active model: {e}")
            return False

    def is_model_available(self) -> bool:
        if self._predictor is None:
            return self.load_active_model()
        return True

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_model_available():
            raise RuntimeError("Model is not loaded or has not been trained yet. Please train a model first.")
        res = self._predictor.predict(text)
        res["model_version"] = self._model_info.get("version", "v1.0") if self._model_info else "v1.0"
        return res

    def get_info(self) -> Optional[Dict[str, Any]]:
        self.is_model_available()
        return self._model_info

model_service = ModelService()
