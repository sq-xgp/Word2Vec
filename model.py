"""Skip-gram with negative sampling model."""

import torch
from torch import nn
from torch.nn import functional as F


class SkipGramNegSampling(nn.Module):
    """Word2Vec model with configurable embedding tables and scores."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        embedding_mode: str = "dual",
        score_mode: str = "dot",
    ) -> None:
        super().__init__()
        if vocab_size < 1 or embedding_dim < 1:
            raise ValueError("vocab_size 和 embedding_dim 必须是正整数")
        if embedding_mode not in {"dual", "shared"}:
            raise ValueError("embedding_mode 必须是 'dual' 或 'shared'")
        if score_mode not in {"dot", "cosine"}:
            raise ValueError("score_mode 必须是 'dot' 或 'cosine'")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode
        self.score_mode = score_mode
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = (
            nn.Embedding(vocab_size, embedding_dim)
            if embedding_mode == "dual"
            else self.center_embeddings
        )

        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        if embedding_mode == "dual":
            nn.init.zeros_(self.context_embeddings.weight)

    def _score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "cosine":
            return F.cosine_similarity(left, right, dim=-1)
        return (left * right).sum(dim=-1)

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive [B] and negative [B, K] scores."""
        center_vectors = self.center_embeddings(centers)       # [B, D]
        context_vectors = self.context_embeddings(contexts)   # [B, D]
        negative_vectors = self.context_embeddings(negatives) # [B, K, D]

        positive_scores = self._score(center_vectors, context_vectors)  # [B]
        negative_scores = self._score(
            center_vectors.unsqueeze(1), negative_vectors
        )  # [B, K]
        return positive_scores, negative_scores

    def compute_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean SGNS loss for a batch."""
        positive_loss = -F.logsigmoid(positive_scores)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=-1)
        return (positive_loss + negative_loss).mean()
