"""Train and validate; reserve the test split for final evaluation."""
import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import Autoencoder
from autoencoder_utils import DEFAULT_DATA, load_data


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for (x,) in loader:
            x = x.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), x)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            total_samples += len(x)
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path,
                        default=Path("checkpoints/autoencoder"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be finite and positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    data = load_data(args.data)
    X = data["X"]

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(X[data["train_idx"]]), batch_size=args.batch_size,
        shuffle=True, generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(X[data["val_idx"]]), batch_size=args.batch_size,
        shuffle=False,
    )
    config = {"input_dim": X.shape[1], "hidden_dim": 32, "latent_dim": 2}
    model = Autoencoder(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best.pt"
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    print("Device:", device)
    print("Run directory:", run_dir.resolve())

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse"])
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()
            improved = val_loss < best_val
            if improved:
                best_val, best_epoch, bad_epochs = val_loss, epoch, 0
                torch.save({
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "model_config": config,
                    "epoch": epoch, "val_mse": val_loss,
                    "seed": args.seed, "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size, "patience": args.patience,
                    "data_path": str(args.data.resolve()),
                    "data_sha256": data_sha256,
                    "normalization": data["normalization"],
                    "normalization_eps": data["normalization_eps"],
                    "source_checkpoint": data["source_checkpoint"],
                    "embedding_table": data["embedding_table"],
                }, best_path)
            else:
                bad_epochs += 1
            print(
                f"Epoch {epoch:03d} | train={train_loss:.8f} | "
                f"val={val_loss:.8f} | wait={bad_epochs}/{args.patience}"
                + (" * saved" if improved else ""), flush=True,
            )
            if bad_epochs >= args.patience:
                print("Early stopping")
                break
    print(f"Best epoch: {best_epoch}; validation MSE: {best_val:.8f}")
    print("Best checkpoint:", best_path.resolve())


if __name__ == "__main__":
    main()
