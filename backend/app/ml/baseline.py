import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from typing import Optional

class TFIDFMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 4, dropout: float = 0.3):
        super(TFIDFMLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)

class BaselineModel:
    def __init__(self, max_features: int = 5000, hidden_dim: int = 128, dropout: float = 0.3):
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.model: Optional[TFIDFMLP] = None

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)
        self.model = TFIDFMLP(input_dim=len(self.vectorizer.vocabulary_), hidden_dim=self.hidden_dim, dropout=self.dropout)

    def transform(self, texts):
        matrix = self.vectorizer.transform(texts)
        return torch.tensor(matrix.toarray(), dtype=torch.float32)

    def save(self, filepath: str):
        data = {
            'max_features': self.max_features,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'vectorizer': self.vectorizer,
            'state_dict': self.model.state_dict() if self.model else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'BaselineModel':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(max_features=data['max_features'], hidden_dim=data['hidden_dim'], dropout=data['dropout'])
        instance.vectorizer = data['vectorizer']
        instance.model = TFIDFMLP(input_dim=len(instance.vectorizer.vocabulary_), hidden_dim=instance.hidden_dim, dropout=instance.dropout)
        if data['state_dict'] is not None:
            instance.model.load_state_dict(data['state_dict'])
        return instance
