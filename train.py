"""Train configurable Skip-gram models with validation and early stopping."""

import argparse
import random
from pathlib import Path
from time import monotonic

import torch
from torch.utils.data import DataLoader

from dataset import (
    SentenceWord2VecDataset,
    build_center_to_positive_contexts,
    iter_skipgram_pairs_by_sentence,
)
from model import SkipGramNegSampling
from prepare_data import prepare_training_data


SENTENCE_FILE = Path("data/raw/eng_wikipedia_2016_10K-sentences.txt")


def model_name(config: dict) -> str:
    """Build a readable, collision-resistant name from important hyperparameters."""
    threshold = f"{config['subsampling_threshold']:.0e}".replace("-", "m")
    score_name = config["score_mode"]
    if score_name == "cosine":
        score_name += f"_t{config['temperature']:g}"
    return (
        f"w2v_{config['embedding_mode']}_{score_name}_"
        f"d{config['embedding_dim']}_w{config['window_size']}_"
        f"n{config['num_negatives']}_bs{config['batch_size']}_"
        f"lr{config['learning_rate']:g}_mc{config['min_count']}_ss{threshold}_"
        f"e{config['num_epochs']}_vf{config['validation_fraction']:g}_"
        f"p{config['patience']}_seed{config['seed']}"
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    word_to_id: dict[str, int],
    config: dict,
    train_loss_history: list[float],
    validation_loss_history: list[float],
    best_validation_loss: float,
) -> None:
    """Atomically save all state needed for inference or continued analysis."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "word_to_id": dict(word_to_id),
            "config": dict(config),
            "loss_history": list(train_loss_history),
            "train_loss_history": list(train_loss_history),
            "validation_loss_history": list(validation_loss_history),
            "best_validation_loss": best_validation_loss,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def create_model_and_optimizer(
    vocab_size: int,
    embedding_dim: int = 100,
    learning_rate: float = 0.01,
    embedding_mode: str = "dual",
    score_mode: str = "dot",
    temperature: float = 0.1,
    device: str | torch.device | None = None,
) -> tuple[SkipGramNegSampling, torch.optim.Optimizer, torch.device]:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SkipGramNegSampling(
        vocab_size,
        embedding_dim,
        embedding_mode=embedding_mode,
        score_mode=score_mode,
        temperature=temperature,
    ).to(selected_device)
    return model, torch.optim.Adam(model.parameters(), lr=learning_rate), selected_device


def create_dataloader(
    sentence_token_ids: list[list[int]],
    negative_sampling_probs: list[float],
    window_size: int,
    num_negatives: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    shuffle_sentences: bool,
    center_to_positive_contexts: dict[int, set[int]] | None = None,
) -> tuple[SentenceWord2VecDataset, DataLoader]:
    dataset = SentenceWord2VecDataset(
        sentence_token_ids,
        negative_sampling_probs,
        window_size=window_size,
        num_negatives=num_negatives,
        seed=seed,
        shuffle_sentences=shuffle_sentences,
        resample_negatives=shuffle_sentences,
        center_to_positive_contexts=center_to_positive_contexts,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    return dataset, loader


def create_training_dataloader(
    sentence_token_ids: list[list[int]],
    negative_sampling_probs: list[float],
    window_size: int = 2,
    num_negatives: int = 5,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[SentenceWord2VecDataset, DataLoader]:
    """Backward-compatible training-loader helper used by earlier lessons."""
    return create_dataloader(
        sentence_token_ids, negative_sampling_probs, window_size,
        num_negatives, batch_size, seed, num_workers, True, None,
    )


def split_sentences(
    sentences: list[list[int]], validation_fraction: float, seed: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Deterministically split whole sentences so no window crosses the split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")
    indices = list(range(len(sentences)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_size])
    train_sentences = [s for i, s in enumerate(sentences) if i not in validation_indices]
    validation_sentences = [s for i, s in enumerate(sentences) if i in validation_indices]
    return train_sentences, validation_sentences


def run_epoch(
    model: SkipGramNegSampling,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    progress_every_batches: int | None = None,
) -> float:
    """Run a training epoch when optimizer is given, otherwise a validation epoch."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total_loss = 0.0
    total_samples = 0
    started = monotonic()

    with torch.set_grad_enabled(training):
        for batch_index, (centers, contexts, negatives) in enumerate(dataloader, 1):
            centers = centers.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            positive_scores, negative_scores = model(centers, contexts, negatives)
            loss = model.compute_loss(positive_scores, negative_scores)
            if training:
                loss.backward()
                optimizer.step()

            sample_count = centers.shape[0]
            total_loss += loss.item() * sample_count
            total_samples += sample_count
            if progress_every_batches and batch_index % progress_every_batches == 0:
                percent = 100 * batch_index / max(len(dataloader), 1)
                print(
                    f"  Batch {batch_index} (~{percent:.1f}%) - "
                    f"elapsed {monotonic() - started:.1f}s",
                    flush=True,
                )

    if total_samples == 0:
        raise ValueError("DataLoader 没有产生样本")
    return total_loss / total_samples


def train_one_batch(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    centers: torch.Tensor,
    contexts: torch.Tensor,
    negatives: torch.Tensor,
) -> float:
    """Keep the small-demo API while using the same loss as the full trainer."""
    model.train()
    optimizer.zero_grad()
    positive_scores, negative_scores = model(centers, contexts, negatives)
    loss = model.compute_loss(positive_scores, negative_scores)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    progress_every_batches: int | None = None,
) -> float:
    """Backward-compatible name for a training epoch."""
    return run_epoch(model, dataloader, optimizer, progress_every_batches)


def train_model(
    model: SkipGramNegSampling,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    checkpoint_path: Path,
    word_to_id: dict[str, int],
    config: dict,
    patience: int = 3,
    min_delta: float = 0.0,
    progress_every_batches: int | None = 100,
) -> tuple[list[float], list[float]]:
    """Train, retain the best validation checkpoint, and stop after no improvement."""
    if patience < 1:
        raise ValueError("patience 必须是正整数")
    train_history: list[float] = []
    validation_history: list[float] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, progress_every_batches)
        validation_loss = run_epoch(model, validation_loader)
        train_history.append(train_loss)
        validation_history.append(validation_loss)
        improved = validation_loss < best_validation_loss - min_delta

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                word_to_id,
                config,
                train_history,
                validation_history,
                best_validation_loss,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{num_epochs} - train: {train_loss:.6f} - "
            f"validation: {validation_loss:.6f}" + (" - saved" if improved else "")
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping: validation loss {patience} 轮未改善")
            break

    return train_history, validation_history


def run_training(
    sentence_file: Path = SENTENCE_FILE,
    checkpoint_path: Path | None = None,
    min_count: int = 5,
    subsampling_threshold: float = 1e-4,
    window_size: int = 5,
    embedding_dim: int = 100,
    embedding_mode: str = "dual",
    score_mode: str = "dot",
    temperature: float = 0.1,
    num_negatives: int = 5,
    batch_size: int = 2048,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    validation_fraction: float = 0.1,
    patience: int = 3,
    min_delta: float = 0.0,
    seed: int = 42,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    progress_every_batches: int | None = 100,
) -> tuple[SkipGramNegSampling, list[float], list[float], torch.device, Path]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prepared = prepare_training_data(
        Path(sentence_file), min_count, subsampling_threshold, seed
    )
    train_sentences, validation_sentences = split_sentences(
        prepared["sentence_token_ids"], validation_fraction, seed
    )
    all_positive_contexts = build_center_to_positive_contexts(
        iter_skipgram_pairs_by_sentence(
            prepared["sentence_token_ids"], window_size=window_size
        )
    )
    loader_args = (
        prepared["negative_sampling_probs"], window_size, num_negatives,
        batch_size, seed, num_workers,
    )
    train_dataset, train_loader = create_dataloader(
        train_sentences, *loader_args, shuffle_sentences=True,
        center_to_positive_contexts=all_positive_contexts,
    )
    validation_dataset, validation_loader = create_dataloader(
        validation_sentences, *loader_args, shuffle_sentences=False,
        center_to_positive_contexts=all_positive_contexts,
    )
    model, optimizer, selected_device = create_model_and_optimizer(
        len(prepared["word_to_id"]), embedding_dim, learning_rate,
        embedding_mode, score_mode, temperature, device,
    )
    config = {
        "vocab_size": len(prepared["word_to_id"]),
        "embedding_dim": embedding_dim,
        "embedding_mode": embedding_mode,
        "score_mode": score_mode,
        "temperature": temperature,
        "min_count": min_count,
        "subsampling_threshold": subsampling_threshold,
        "window_size": window_size,
        "num_negatives": num_negatives,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "patience": patience,
        "min_delta": min_delta,
        "seed": seed,
        "num_workers": num_workers,
    }
    output_path = Path(checkpoint_path or Path("checkpoints") / f"{model_name(config)}.pt")
    print(f"Device: {selected_device}")
    print(f"Model: {model_name(config)}")
    print(f"Train/validation pairs: {len(train_dataset)}/{len(validation_dataset)}")
    train_history, validation_history = train_model(
        model, optimizer, train_loader, validation_loader, num_epochs,
        output_path, prepared["word_to_id"], config, patience, min_delta,
        progress_every_batches,
    )
    return model, train_history, validation_history, selected_device, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练可配置的 Word2Vec 模型")
    parser.add_argument("--sentence-file", type=Path, default=SENTENCE_FILE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--subsampling-threshold", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--embedding-mode", choices=("dual", "shared"), default="dual")
    parser.add_argument("--score-mode", choices=("dot", "cosine"), default="dot")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=100)
    args = parser.parse_args()
    run_training(
        sentence_file=args.sentence_file,
        checkpoint_path=args.checkpoint,
        min_count=args.min_count,
        subsampling_threshold=args.subsampling_threshold,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        embedding_mode=args.embedding_mode,
        score_mode=args.score_mode,
        temperature=args.temperature,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        progress_every_batches=args.progress_every_batches,
    )


if __name__ == "__main__":
    main()
