"""Evaluate the chosen autoencoder and encode words in vocabulary order."""
import argparse
from pathlib import Path
import torch
from autoencoder_utils import DEFAULT_DATA, load_data, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = load_data(args.data)
    model, checkpoint = load_model(args.checkpoint, args.data, data)
    X = data["X"]
    with torch.no_grad():
        test_x = X[data["test_idx"]]
        test_mse = (model(test_x) - test_x).square().mean().item()
        Z = model.encoder(X)
    assert Z.shape == (len(data["words"]), 2)
    assert torch.isfinite(Z).all()
    output = args.checkpoint.parent / "embeddings_2d.pt"
    torch.save({
        "Z": Z, "words": data["words"], "test_idx": data["test_idx"],
        "test_mse": test_mse, "source_checkpoint": str(args.checkpoint.resolve()),
    }, output)
    print("Best epoch:", checkpoint["epoch"])
    print(f"Test MSE: {test_mse:.8f}")
    print("2D shape:", tuple(Z.shape))
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
