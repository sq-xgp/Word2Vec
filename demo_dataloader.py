"""演示如何用 DataLoader 把正负样本组成 batch，不执行模型训练。"""

from torch.utils.data import DataLoader

from dataset import Word2VecDataset


def main() -> None:
    # 5 条正样本，编号范围为 0～3；重复正样本仍然参与读取。
    skipgram_pairs = [(0, 1), (0, 2), (0, 1), (1, 0), (2, 3)]
    negative_sampling_probs = [0.1, 0.2, 0.3, 0.4]
    dataset = Word2VecDataset(
        skipgram_pairs,
        negative_sampling_probs,
        num_negatives=3,
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,       # 每批最多包含 2 条正样本。
        shuffle=False,      # 演示时保留顺序，方便核对输入。
        num_workers=0,      # 在当前进程读取，暂不使用多个 worker。
        drop_last=False,    # 保留最后不足 2 条的 batch。
    )

    print("Number of samples:", len(dataset))
    print("Number of batches:", len(dataloader))
    for batch_index, (centers, contexts, negatives) in enumerate(dataloader, start=1):
        print(f"Batch {batch_index}")
        print("centers:", centers, "shape:", list(centers.shape))
        print("contexts:", contexts, "shape:", list(contexts.shape))
        print("negatives:", negatives, "shape:", list(negatives.shape))


if __name__ == "__main__":
    main()
