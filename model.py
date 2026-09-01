"""模型：定义 SkipGramNegSampling，计算带负采样的 Skip-gram 损失。"""

import torch
from torch import nn
from torch.nn import functional as F


class SkipGramNegSampling(nn.Module):
    """保存 Skip-gram 负采样模型的两张词向量表。

    center_embeddings 用于中心词，context_embeddings 用于正上下文和负样本。
    两张表的形状均为 [vocab_size, embedding_dim]，每行对应一个词编号。
    forward 返回正负样本的原始点积分数；compute_loss 将分数组合为负采样损失。
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        """创建两张独立、可训练的 embedding 表。

        中心词表使用小范围均匀随机数初始化，上下文词表初始化为零。
        不在类内部固定种子；需要复现时，在创建模型前调用 torch.manual_seed。
        """
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocab_size 必须是正整数")
        if type(embedding_dim) is not int or embedding_dim < 1:
            raise ValueError("embedding_dim 必须是正整数")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对一个 batch 查表并打分，返回形状为 [B]、[B, K] 的分数。

        centers 和 contexts 是 [B]，negatives 是 [B, K]，均为 long 编号张量。
        B 和 K 均须大于 0，编号须在词表内，输入与两张表须位于同一设备。
        正负样本都与中心词做普通点积，不做 sigmoid、不取负号或计算损失。
        不重新抽取负样本、不更新参数，也不切断梯度计算图。
        """
        inputs = (("centers", centers), ("contexts", contexts), ("negatives", negatives))
        for name, word_ids in inputs:
            if not isinstance(word_ids, torch.Tensor) or word_ids.dtype != torch.long:
                raise TypeError(f"{name} 必须是 torch.long 类型的张量")
        if centers.ndim != 1 or contexts.ndim != 1 or negatives.ndim != 2:
            raise ValueError("centers、contexts、negatives 的形状必须分别为 [B]、[B]、[B, K]")

        batch_size = centers.shape[0]
        if contexts.shape[0] != batch_size or negatives.shape[0] != batch_size:
            raise ValueError("三个输入的 batch 大小必须一致")
        if batch_size == 0 or negatives.shape[1] == 0:
            raise ValueError("batch 大小和每条样本的负样本数量必须大于 0")

        device = self.center_embeddings.weight.device
        if self.context_embeddings.weight.device != device:
            raise ValueError("两张 embedding 表必须位于同一设备")
        for name, word_ids in inputs:
            if word_ids.device != device:
                raise ValueError(f"{name} 必须与模型位于同一设备")
            if torch.any(word_ids < 0) or torch.any(word_ids >= self.vocab_size):
                raise ValueError(f"{name} 中的词编号超出了词表范围")

        center_vectors = self.center_embeddings(centers)       # [B, D]
        context_vectors = self.context_embeddings(contexts)   # [B, D]
        negative_vectors = self.context_embeddings(negatives) # [B, K, D]

        positive_scores = (center_vectors * context_vectors).sum(dim=-1)
        negative_scores = (center_vectors.unsqueeze(1) * negative_vectors).sum(dim=-1)
        return positive_scores, negative_scores

    def compute_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """计算 SGNS 损失：先对每条样本的负例求和，再对 batch 取平均。

        输入为 [B] 和 [B, K] 的原始点积分数，不要预先做 sigmoid 或取负号。
        两组分数须有限、同设备、同类型，当前支持 float32 和 float64。
        B、K 均须大于 0；返回可反向传播的标量张量，形状为 []。
        使用 logsigmoid 保持数值稳定，本方法不执行 backward 或更新参数。
        """
        scores = (("positive_scores", positive_scores), ("negative_scores", negative_scores))
        for name, values in scores:
            if not isinstance(values, torch.Tensor) or values.dtype not in (
                torch.float32, torch.float64
            ):
                raise TypeError(f"{name} 必须是 float32 或 float64 张量")
        if positive_scores.ndim != 1 or negative_scores.ndim != 2:
            raise ValueError("正负样本分数的形状必须分别为 [B] 和 [B, K]")
        if positive_scores.shape[0] != negative_scores.shape[0]:
            raise ValueError("正负样本分数的 batch 大小必须一致")
        if positive_scores.shape[0] == 0 or negative_scores.shape[1] == 0:
            raise ValueError("batch 大小和每条样本的负样本数量必须大于 0")
        if positive_scores.device != negative_scores.device:
            raise ValueError("正负样本分数必须位于同一设备")
        if positive_scores.dtype != negative_scores.dtype:
            raise TypeError("正负样本分数的数据类型必须一致")
        for name, values in scores:
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} 不能包含 NaN 或无穷大")

        positive_loss = -F.logsigmoid(positive_scores)                         # [B]
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=-1)            # [B]
        loss = (positive_loss + negative_loss).mean()                         # []
        return loss
