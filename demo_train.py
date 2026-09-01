"""用小样本在 CPU 上训练多个 epoch，不下载语料或保存模型。"""

import torch
from torch.utils.data import DataLoader

from dataset import Word2VecDataset
from model import SkipGramNegSampling
from train import train_one_epoch


def main() -> None:
    # 与 DataLoader 演示使用相同的 5 条正样本和 4 个词编号。
    skipgram_pairs = [(0, 1), (0, 2), (0, 1), (1, 0), (2, 3)]
    negative_sampling_probs = [0.1, 0.2, 0.3, 0.4]
    dataset = Word2VecDataset(
        skipgram_pairs,
        negative_sampling_probs,
        num_negatives=3,
        seed=42,                     # 控制 Dataset 的负采样随机序列。
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,             # 最后一批只有 1 条，也参与训练。
    )

    torch.manual_seed(42)            # 控制模型初始化使用的随机数。
    model = SkipGramNegSampling(
        vocab_size=len(negative_sampling_probs),
        embedding_dim=128,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    num_epochs = 5

    print("Number of samples:", len(dataset))
    print("Number of batches:", len(dataloader))
    # 每轮继续使用同一个模型、优化器和 DataLoader，不重新设置种子。
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, dataloader)
        print(f"Epoch {epoch + 1} mean loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    main()
