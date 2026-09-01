"""训练：创建训练组件，执行 batch/epoch 更新，并保存 checkpoint。"""

from pathlib import Path
from time import monotonic

import torch
from torch.utils.data import DataLoader

from dataset import SentenceWord2VecDataset
from model import SkipGramNegSampling
from prepare_data import prepare_training_data


SENTENCE_FILE = Path("data/raw/eng_wikipedia_2016_10K-sentences.txt")
CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")

MIN_COUNT = 5
SUBSAMPLING_THRESHOLD = 1e-4
WINDOW_SIZE = 2
EMBEDDING_DIM = 50
NUM_NEGATIVES = 5
BATCH_SIZE = 512
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
RANDOM_SEED = 42
NUM_WORKERS = 0
PROGRESS_EVERY_BATCHES = 100


def save_checkpoint(
    checkpoint_path: Path,
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    word_to_id: dict[str, int],
    config: dict,
    loss_history: list[float],
) -> None:
    """原子保存训练状态、词表和重建模型所需的配置。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "word_to_id": dict(word_to_id),
        "config": dict(config),
        "loss_history": list(loss_history),
    }
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def create_model_and_optimizer(
    vocab_size: int,
    embedding_dim: int = 50,
    learning_rate: float = 0.01,
    device: str | torch.device | None = None,
) -> tuple[SkipGramNegSampling, torch.optim.Optimizer, torch.device]:
    """创建模型和 Adam；未指定设备时优先使用可用的 CUDA。"""
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SkipGramNegSampling(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    ).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    return model, optimizer, selected_device


def create_training_dataloader(
    sentence_token_ids: list[list[int]],
    negative_sampling_probs: list[float],
    window_size: int = 2,
    num_negatives: int = 5,
    batch_size: int = 512,
    seed: int | None = 42,
    num_workers: int = 0,
) -> tuple[SentenceWord2VecDataset, DataLoader]:
    """创建按句子动态采样的数据集及可选多进程 DataLoader。"""
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers 必须是非负整数")

    dataset = SentenceWord2VecDataset(
        sentence_token_ids,
        negative_sampling_probs,
        window_size=window_size,
        num_negatives=num_negatives,
        seed=seed,
        shuffle_sentences=True,
    )
    dataloader_generator = torch.Generator()
    if seed is not None:
        dataloader_generator.manual_seed(seed)
    else:
        dataloader_generator.seed()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator,
    )
    return dataset, dataloader


def train_one_batch(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    centers: torch.Tensor,
    contexts: torch.Tensor,
    negatives: torch.Tensor,
) -> float:
    """用一个 batch 更新一次模型，返回更新前的损失数值。

    输入形状为 [B]、[B]、[B, K]，编号张量须与模型在同一设备。
    optimizer 应在函数外用该模型的全部参数创建，多个 batch 复用它。
    返回普通 Python 浮点数用于记录；反向传播使用转换前的 loss 张量。
    """
    model.train()                       # 切换到训练模式，本身不会更新参数。
    optimizer.zero_grad()              # 清空上一批留下的梯度，避免意外累加。

    positive_scores, negative_scores = model(centers, contexts, negatives)
    loss = model.compute_loss(positive_scores, negative_scores)

    loss.backward()                    # 计算损失对模型参数的梯度。
    optimizer.step()                   # 优化器根据梯度更新参数。

    return loss.item()                 # 取出本次前向计算得到的损失数值。


def train_one_epoch(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    progress_every_batches: int | None = None,
) -> float:
    """遍历 DataLoader 一次，返回按实际样本数加权的平均训练损失。

    每批复用同一个模型和优化器，并把 DataLoader 产生的张量移动到
    模型参数所在的设备。
    多 worker 时 Dataset 按 worker 编号分片句子并使用独立负采样 RNG。
    progress_every_batches 指定每隔多少批输出一次近似进度；None 表示不输出。
    保留尾批应设置 drop_last=False。
    损失统计来自各批更新前的计算，并非用最终参数重新评估的损失。
    如果没有读到任何样本，明确报错，不返回容易误解的零损失。
    """
    if (
        progress_every_batches is not None
        and (
            type(progress_every_batches) is not int
            or progress_every_batches < 1
        )
    ):
        raise ValueError("progress_every_batches 必须是正整数或 None")

    device = next(model.parameters()).device
    total_loss = 0.0
    total_samples = 0
    expected_batches = len(dataloader)
    epoch_start_time = monotonic()

    for batch_index, (centers, contexts, negatives) in enumerate(
        dataloader,
        start=1,
    ):
        centers = centers.to(device, non_blocking=True)
        contexts = contexts.to(device, non_blocking=True)
        negatives = negatives.to(device, non_blocking=True)

        loss_value = train_one_batch(model, optimizer, centers, contexts, negatives)
        batch_size = centers.shape[0]              # 使用这一批的实际样本数。
        total_loss += loss_value * batch_size     # 将批平均损失还原为批损失总和。
        total_samples += batch_size
        if (
            progress_every_batches is not None
            and batch_index % progress_every_batches == 0
        ):
            elapsed_seconds = monotonic() - epoch_start_time
            progress_percent = min(
                100.0,
                100.0 * batch_index / max(expected_batches, 1),
            )
            print(
                f"  Batch {batch_index} (~{progress_percent:.1f}%) - "
                f"elapsed {elapsed_seconds:.1f}s",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("DataLoader 没有产生训练样本，请检查数据集和 drop_last 设置")

    return total_loss / total_samples


def train_model(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    num_epochs: int,
    checkpoint_path: Path,
    word_to_id: dict[str, int],
    config: dict,
    progress_every_batches: int | None = None,
) -> list[float]:
    """训练指定轮数，并在每轮结束后覆盖保存最新 checkpoint。"""
    if type(num_epochs) is not int or num_epochs < 1:
        raise ValueError("num_epochs 必须是正整数")

    loss_history = []
    for epoch in range(1, num_epochs + 1):
        mean_loss = train_one_epoch(
            model,
            optimizer,
            dataloader,
            progress_every_batches=progress_every_batches,
        )
        loss_history.append(mean_loss)
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=epoch,
            word_to_id=word_to_id,
            config=config,
            loss_history=loss_history,
        )
        print(f"Epoch {epoch}/{num_epochs} - mean loss: {mean_loss:.6f}")

    return loss_history


def run_training(
    sentence_file: Path = SENTENCE_FILE,
    checkpoint_path: Path = CHECKPOINT_PATH,
    min_count: int = MIN_COUNT,
    subsampling_threshold: float = SUBSAMPLING_THRESHOLD,
    window_size: int = WINDOW_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    num_negatives: int = NUM_NEGATIVES,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    device: str | torch.device | None = None,
    num_workers: int = NUM_WORKERS,
    progress_every_batches: int | None = PROGRESS_EVERY_BATCHES,
) -> tuple[SkipGramNegSampling, list[float], torch.device]:
    """准备真实语料并执行完整训练，返回模型、loss 历史和设备。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prepared = prepare_training_data(
        Path(sentence_file),
        min_count=min_count,
        subsampling_threshold=subsampling_threshold,
        seed=seed,
    )
    dataset, dataloader = create_training_dataloader(
        prepared["sentence_token_ids"],
        prepared["negative_sampling_probs"],
        window_size=window_size,
        num_negatives=num_negatives,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    model, optimizer, selected_device = create_model_and_optimizer(
        vocab_size=len(prepared["word_to_id"]),
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        device=device,
    )
    config = {
        "vocab_size": len(prepared["word_to_id"]),
        "embedding_dim": embedding_dim,
        "min_count": min_count,
        "subsampling_threshold": subsampling_threshold,
        "window_size": window_size,
        "num_negatives": num_negatives,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "seed": seed,
        "num_workers": num_workers,
    }

    print(f"Device: {selected_device}")
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Positive samples per epoch: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    loss_history = train_model(
        model,
        optimizer,
        dataloader,
        num_epochs=num_epochs,
        checkpoint_path=Path(checkpoint_path),
        word_to_id=prepared["word_to_id"],
        config=config,
        progress_every_batches=progress_every_batches,
    )
    return model, loss_history, selected_device


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
