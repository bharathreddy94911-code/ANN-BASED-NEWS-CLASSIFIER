import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Generator, Tuple
import numpy as np

class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Context word embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Linear layer mapping average context embedding to target word vocabulary logits
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize embeddings with uniform distribution
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embeddings.weight[0].fill_(0)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        # context_ids shape: [batch_size, 2 * window_size]
        # Look up embeddings: [batch_size, 2 * window_size, embed_dim]
        embeds = self.embeddings(context_ids)
        # Calculate mean across context words: [batch_size, embed_dim]
        mean_embeds = torch.mean(embeds, dim=1)
        # Logits over vocabulary: [batch_size, vocab_size]
        logits = self.linear(mean_embeds)
        return logits

class CBOWNegativeSampling(nn.Module):
    """CBOW with Negative Sampling for scalable, high-throughput pre-training."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super(CBOWNegativeSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.in_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        nn.init.uniform_(self.in_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.out_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.in_embed.weight[0].fill_(0)
            self.out_embed.weight[0].fill_(0)

    def forward(self, context_ids: torch.Tensor, pos_targets: torch.Tensor, neg_targets: torch.Tensor) -> torch.Tensor:
        # context_ids: [B, 2*W], pos_targets: [B], neg_targets: [B, K]
        ctx_embeds = self.in_embed(context_ids).mean(dim=1) # [B, D]
        pos_embeds = self.out_embed(pos_targets) # [B, D]
        neg_embeds = self.out_embed(neg_targets) # [B, K, D]

        # Positive score: dot product ctx and pos
        pos_score = (ctx_embeds * pos_embeds).sum(dim=1) # [B]
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-7).mean()

        # Negative score: dot product ctx and neg
        # ctx_embeds: [B, 1, D], neg_embeds: [B, K, D]
        neg_score = torch.bmm(neg_embeds, ctx_embeds.unsqueeze(2)).squeeze(2) # [B, K]
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-7).sum(dim=1).mean()

        return pos_loss + neg_loss

def generate_cbow_pairs(token_ids: List[int], window_size: int = 2) -> Generator[Tuple[List[int], int], None, None]:
    """Generates (context, target) pairs without padding or unknown target words."""
    length = len(token_ids)
    for i in range(window_size, length - window_size):
        target = token_ids[i]
        # Ignore special tokens like PAD (0) as target
        if target == 0:
            continue
        context = [token_ids[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        yield context, target

class CBOWDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
