import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from app.config import DEVICE, LABEL_NAMES
from app.ml.preprocessing import tokenize
from app.ml.vocabulary import Vocabulary
from app.ml.classifier import CBOWMLPClassifier

class Predictor:
    def __init__(self, model: CBOWMLPClassifier, vocab: Vocabulary, max_seq_len: int = 64):
        self.model = model
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("Input text contains no valid tokens.")
            
        token_ids = self.vocab.encode(tokens, max_len=self.max_seq_len)
        tensor_in = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).cpu().squeeze().tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())
        latency_ms = (time.time() - start_time) * 1000.0

        distribution = {
            LABEL_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(LABEL_NAMES))
        }

        return {
            "predicted_label": pred_idx,
            "predicted_category": LABEL_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": distribution,
            "latency_ms": round(latency_ms, 2)
        }
