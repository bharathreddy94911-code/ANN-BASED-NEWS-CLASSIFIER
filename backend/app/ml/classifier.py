import torch
import torch.nn as nn
from typing import Optional

class CBOWMLPClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        num_layers: int = 1,
        pooling: str = "concat", # "mean", "max", or "concat"
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        super(CBOWMLPClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.pooling = pooling.lower()
        self.freeze_embeddings = freeze_embeddings

        # Embedding layer with padding_idx=0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Input dimension into MLP depends on pooling strategy
        mlp_in_dim = embed_dim * 2 if self.pooling == "concat" else embed_dim

        # First hidden layer
        self.fc1 = nn.Linear(mlp_in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Optional second hidden layer
        if num_layers > 1:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc2 = None
            self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        # Embeddings: [batch_size, seq_len, embed_dim]
        embeds = self.embedding(input_ids)
        mask = (input_ids != 0).float().unsqueeze(-1) # [batch_size, seq_len, 1]

        if self.pooling == "max":
            # Mask out padding with a large negative value
            masked_for_max = embeds.masked_fill(mask == 0, -1e9)
            pooled, _ = torch.max(masked_for_max, dim=1) # [batch_size, embed_dim]
        elif self.pooling == "concat":
            # Padding-aware mean pooling
            masked_embeds = embeds * mask
            sum_embeds = torch.sum(masked_embeds, dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1.0)
            mean_pooled = sum_embeds / lengths

            # Masked max pooling
            masked_for_max = embeds.masked_fill(mask == 0, -1e9)
            max_pooled, _ = torch.max(masked_for_max, dim=1)

            # Concatenate both features: [batch_size, 2 * embed_dim]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        else:
            # Default: standard padding-aware mean pooling
            masked_embeds = embeds * mask
            sum_embeds = torch.sum(masked_embeds, dim=1)
            lengths = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = sum_embeds / lengths # [batch_size, embed_dim]

        # Feed forward MLP
        h = self.fc1(pooled)
        h = self.relu(h)
        h = self.dropout(h)

        if self.fc2 is not None:
            h = self.fc2(h)
            h = self.relu(h)
            h = self.dropout(h)

        logits = self.out(h) # [batch_size, num_classes]
        return logits
