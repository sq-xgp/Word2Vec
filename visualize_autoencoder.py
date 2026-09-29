"""Plot saved coordinates and training history without refitting a model."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.run_dir / "embeddings_2d.pt",
                      map_location="cpu", weights_only=True)
    Z = data["Z"].numpy()
    words = data["words"]
    word_to_id = {word: i for i, word in enumerate(words)}
    selected = ["music", "song", "musical", "jazz", "city", "town",
                "village", "urban", "war", "battle", "army", "soldier"]
    ids = [word_to_id[w] for w in selected if w in word_to_id]
    missing = [w for w in selected if w not in word_to_id]
    output = args.run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.12, color="gray",
               rasterized=True, label="All words")
    if ids:
        ax.scatter(Z[ids, 0], Z[ids, 1], s=25, color="tab:blue",
                   label="Selected words")
        for i in ids:
            ax.annotate(words[i], Z[i], xytext=(4, 4),
                        textcoords="offset points", fontsize=9)
    ax.set(xlabel="Latent dimension 1", ylabel="Latent dimension 2",
           title="Word2Vec to 2D: Autoencoder")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "embeddings_2d.png", dpi=180)
    plt.close(fig)

    history = args.run_dir / "history.csv"
    if history.exists():
        with history.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(row["epoch"]) for row in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in [("train_mse", "Train"), ("val_mse", "Validation")]:
            ax.plot(epochs, [float(row[key]) for row in rows], label=label)
        ax.set(xlabel="Epoch", ylabel="MSE", title="Autoencoder training history")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "loss_curve.png", dpi=180)
        plt.close(fig)
    print("Missing selected words:", missing)
    print("Plots saved to:", output.resolve())


if __name__ == "__main__":
    main()
