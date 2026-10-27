import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from app.config import MODELS_DIR, VOCAB_DIR, DEVICE, NUM_CLASSES
from app.ml.preprocessing import tokenize
from app.ml.vocabulary import Vocabulary
from app.ml.cbow import CBOWModel, generate_cbow_pairs, CBOWDataset
from app.ml.classifier import CBOWMLPClassifier
from app.ml.baseline import BaselineModel, TFIDFMLP
from app.ml.metrics import compute_all_metrics

class TrainingPipeline:
    @staticmethod
    def train_cbow_embeddings(
        train_texts: List[str],
        vocab: Vocabulary,
        embed_dim: int = 100,
        window_size: int = 2,
        batch_size: int = 256,
        epochs: int = 3,
        lr: float = 0.01,
        max_pairs: int = 150000,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> torch.Tensor:
        if progress_callback:
            progress_callback("Extracting CBOW context-target pairs from corpus...")
            
        pairs = []
        for text in train_texts:
            tokens = tokenize(text)
            ids = [vocab.w2i.get(t, 1) for t in tokens if t in vocab.w2i]
            if len(ids) >= (2 * window_size + 1):
                for p in generate_cbow_pairs(ids, window_size=window_size):
                    pairs.append(p)
                    if len(pairs) >= max_pairs:
                        break
            if len(pairs) >= max_pairs:
                break
                
        if progress_callback:
            progress_callback(f"Training CBOW on {len(pairs)} context-target pairs for {epochs} epochs...")

        dataset = CBOWDataset(pairs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        cbow = CBOWModel(vocab_size=len(vocab), embed_dim=embed_dim).to(DEVICE)
        optimizer = optim.Adam(cbow.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        cbow.train()
        for ep in range(epochs):
            total_loss = 0.0
            for ctx, tgt in loader:
                ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
                optimizer.zero_grad()
                out = cbow(ctx)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if progress_callback:
                progress_callback(f"CBOW Epoch {ep+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

        return cbow.embeddings.weight.detach().cpu()

    @staticmethod
    def train_classifier(
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        vocab: Vocabulary,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "mean",
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 6,
        optimizer_name: str = "sgd",
        max_seq_len: int = 64,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[CBOWMLPClassifier, Dict[str, Any]]:
        # Tokenize and encode
        X_train = [vocab.encode(tokenize(t), max_len=max_seq_len) for t in train_texts]
        X_val = [vocab.encode(tokenize(t), max_len=max_seq_len) for t in val_texts]
        
        t_X_train = torch.tensor(X_train, dtype=torch.long)
        t_y_train = torch.tensor(train_labels, dtype=torch.long)
        t_X_val = torch.tensor(X_val, dtype=torch.long)
        t_y_val = torch.tensor(val_labels, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=batch_size, shuffle=False)
        
        model = CBOWMLPClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling,
            pretrained_embeddings=pretrained_embeddings
        ).to(DEVICE)
        
        # Select optimizer with sensible defaults
        opt_name = optimizer_name.lower()
        if opt_name == "adam":
            effective_lr = lr if lr is not None else 0.002
            optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=weight_decay)
        else: # Mini-Batch SGD default
            effective_lr = lr if lr is not None else 0.3
            optimizer = optim.SGD(model.parameters(), lr=effective_lr, momentum=0.9, weight_decay=weight_decay)
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=effective_lr * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        history = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
        best_val_f1 = -1.0
        best_state = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(by.cpu().numpy())
                    
            val_loss /= len(val_loader)
            from sklearn.metrics import f1_score
            val_acc = float(np.mean(np.array(val_preds) == np.array(val_targets)))
            val_f1 = float(f1_score(val_targets, val_preds, average='macro', zero_division=0))
            
            history["epochs"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_accuracy"].append(round(val_acc, 4))
            history["val_macro_f1"].append(round(val_f1, 4))
            
            # Primary model selection metric: Validation Macro F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                    "val_macro_f1": round(val_f1, 4)
                })

        if best_state is not None:
            model.load_state_dict(best_state)
            
        return model, history

    @staticmethod
    def train_baseline(
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        max_features: int = 5000,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 0.01,
        batch_size: int = 64,
        epochs: int = 5,
        optimizer_name: str = "sgd",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        baseline = BaselineModel(max_features=max_features, hidden_dim=hidden_dim, dropout=dropout)
        baseline.fit_vectorizer(train_texts)
        
        X_train_tensor = baseline.transform(train_texts)
        y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
        X_val_tensor = baseline.transform(val_texts)
        y_val_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
        
        model = baseline.model.to(DEVICE)
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            
        criterion = nn.CrossEntropyLoss()
        history = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
        
        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(by.cpu().numpy())
                    
            val_loss /= len(val_loader)
            val_acc = float(np.mean(np.array(val_preds) == np.array(val_targets)))
            
            history["epochs"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_accuracy"].append(round(val_acc, 4))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4)
                })
                
        if best_state is not None:
            baseline.model.load_state_dict(best_state)
            
        return baseline, history
