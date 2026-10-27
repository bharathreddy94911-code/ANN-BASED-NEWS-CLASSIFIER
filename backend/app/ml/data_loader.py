import logging
from typing import Dict, Any, Optional
from datasets import load_dataset, DatasetDict
from app.config import DATASET_NAME, DATASET_URL, LABEL_MAPPING, LABEL_NAMES

logger = logging.getLogger(__name__)

class DatasetManager:
    _instance: Optional['DatasetManager'] = None
    _dataset: Optional[DatasetDict] = None
    _status: str = "unloaded"
    _error_message: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
        return cls._instance

    def load(self, force_reload: bool = False) -> DatasetDict:
        if self._dataset is not None and not force_reload:
            return self._dataset

        try:
            self._status = "loading"
            logger.info(f"Loading dataset: {DATASET_NAME}")
            ds = load_dataset(DATASET_NAME)
            
            # Validate required splits and columns
            if "train" not in ds or "test" not in ds:
                raise ValueError(f"Dataset {DATASET_NAME} missing required 'train' or 'test' splits.")
            
            for split in ["train", "test"]:
                cols = ds[split].column_names
                if "text" not in cols or "label" not in cols:
                    raise ValueError(f"Split {split} is missing required 'text' or 'label' columns.")
            
            self._dataset = ds
            self._status = "loaded"
            self._error_message = None
            logger.info("Dataset successfully loaded and verified.")
            return self._dataset
        except Exception as e:
            self._status = "error"
            self._error_message = str(e)
            logger.error(f"Failed to load dataset: {e}")
            raise e

    def get_info(self) -> Dict[str, Any]:
        train_count = len(self._dataset["train"]) if self._dataset else 0
        test_count = len(self._dataset["test"]) if self._dataset else 0
        features = list(self._dataset["train"].features.keys()) if self._dataset else ["text", "label"]

        return {
            "dataset_name": DATASET_NAME,
            "dataset_url": DATASET_URL,
            "status": self._status,
            "error_message": self._error_message,
            "train_records": train_count,
            "test_records": test_count,
            "features": features,
            "label_mapping": LABEL_MAPPING,
            "classes": LABEL_NAMES,
            "num_classes": len(LABEL_NAMES)
        }

    def get_distribution(self) -> Dict[str, Dict[str, int]]:
        if self._dataset is None:
            self.load()
        
        distribution = {}
        for split in ["train", "test"]:
            counts = {label_name: 0 for label_name in LABEL_NAMES}
            labels = self._dataset[split]["label"]
            for l in labels:
                if l in LABEL_MAPPING:
                    counts[LABEL_MAPPING[l]] += 1
            distribution[split] = counts
        return distribution

    def get_examples(self, split: str = "train", page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        if self._dataset is None:
            self.load()
        
        if split not in self._dataset:
            raise ValueError(f"Invalid split '{split}'. Available: {list(self._dataset.keys())}")
        
        data_split = self._dataset[split]
        total = len(data_split)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        
        if start_idx >= total:
            return {"total": total, "page": page, "page_size": page_size, "items": []}
        
        slice_data = data_split.select(range(start_idx, end_idx))
        items = []
        for i, row in enumerate(slice_data):
            label_idx = int(row["label"])
            items.append({
                "id": start_idx + i,
                "text": row["text"],
                "label": label_idx,
                "category": LABEL_MAPPING.get(label_idx, "Unknown")
            })
            
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items
        }

dataset_manager = DatasetManager()
