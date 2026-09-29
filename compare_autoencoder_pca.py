"""Compare train-fitted PCA with the selected autoencoder on held-out words."""
import argparse
import json
from pathlib import Path

import torch

from autoencoder_utils import DEFAULT_DATA, load_data, load_model


def neighbor_retention(X, Z, query_ids, k=10, batch_size=128):
    if not 1 <= k < len(X) or batch_size < 1:
        raise ValueError("Require 1 <= k < vocabulary size and positive batch size")
    scores = []
    for ids in query_ids.split(batch_size):
        similarity = X[ids] @ X.T
        # Direct distances avoid cancellation for very close 2D points.
        distance = torch.cdist(Z[ids], Z, compute_mode="donot_use_mm_for_euclid_dist")
        rows = torch.arange(len(ids))
        similarity[rows, ids] = -torch.inf
        distance[rows, ids] = torch.inf
        original = similarity.topk(k, dim=1).indices
        reduced = distance.topk(k, dim=1, largest=False).indices
        overlap = (original[:, :, None] == reduced[:, None, :]).any(dim=2)
        scores.append(overlap.float().mean(dim=1))
    return torch.cat(scores).mean().item()


def fit_pca(train_x, dimensions=2):
    mean = train_x.mean(dim=0)
    _, _, vh = torch.linalg.svd(train_x - mean, full_matrices=False)
    return mean, vh[:dimensions].T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = load_data(args.data)
    model, _ = load_model(args.checkpoint, args.data, data)
    X, test_ids = data["X"], data["test_idx"]
    with torch.no_grad():
        ae_z = model.encoder(X)
        test_x = X[test_ids]
        ae_mse = (model(test_x) - test_x).square().mean().item()
        mean, components = fit_pca(X[data["train_idx"]])
        pca_z = (X - mean) @ components
        pca_hat = pca_z[test_ids] @ components.T + mean
        pca_mse = (pca_hat - test_x).square().mean().item()
        results = {
            "test_words": len(test_ids), "k": args.k,
            "candidate_words": len(X), "exclude_self": True,
            "autoencoder": {
                "test_mse": ae_mse,
                "neighbor_retention": neighbor_retention(X, ae_z, test_ids, args.k),
            },
            "pca": {
                "test_mse": pca_mse,
                "neighbor_retention": neighbor_retention(X, pca_z, test_ids, args.k),
            },
        }
    output = args.output or args.checkpoint.parent / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(results, indent=2)
    output.write_text(content + "\n", encoding="utf-8")
    print(content)
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
