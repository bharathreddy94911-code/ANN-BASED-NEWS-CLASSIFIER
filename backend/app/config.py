import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
VOCAB_DIR = ARTIFACTS_DIR / "vocab"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "Mikami063/ag_news"
DATASET_URL = "https://huggingface.co/datasets/Mikami063/ag_news"

# Verified Class Labels: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4

# Device selection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
