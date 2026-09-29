"""Run from the Word2Vec project root, alongside inference.py."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from inference import load_trained_model
from autoencoder_utils import DEFAULT_DATA


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed Word2Vec vectors for an autoencoder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, word_to_id, id_to_word, _ = load_trained_model(args.checkpoint)
    # Match the center table used by the project's similarity queries.
    vectors = model.center_embeddings.weight.detach().float().cpu()
    if vectors.ndim != 2 or vectors.shape[1] != 100:
        raise ValueError("Expected a [vocabulary_size, 100] embedding matrix")
    count = len(vectors)
    if len(word_to_id) != count or set(id_to_word) != set(range(count)):
        raise ValueError("Vocabulary IDs do not match embedding rows")
    if not torch.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or Inf")
    if (vectors.norm(dim=1) < 1e-12).any():
        raise ValueError("Found zero or near-zero vectors; inspect them before preparing data")

    # Each row is normalized independently; no statistics are fitted across splits.
    X = F.normalize(vectors, p=2, dim=1, eps=1e-12)
    words = [id_to_word[i] for i in range(count)]
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(count, generator=generator)
    n_train, n_val = int(count * 0.8), int(count * 0.1)
    if min(n_train, n_val, count - n_train - n_val) < 1:
        raise ValueError("Vocabulary is too small for three nonempty splits")
    splits = {
        "train_idx": indices[:n_train],
        "val_idx": indices[n_train:n_train + n_val],
        "test_idx": indices[n_train + n_val:],
    }
    payload = {
        "X": X,
        "words": words,
        **splits,
        "seed": args.seed,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "embedding_table": "center_embeddings",
        "normalization": "row_l2",
        "normalization_eps": 1e-12,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid accidentally replacing an existing prepared dataset.
    with args.output.open("xb") as output_file:
        torch.save(payload, output_file)
    print("Vector shape:", tuple(X.shape))
    for name, rows in splits.items():
        print(f"{name}: {len(rows)}")
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
