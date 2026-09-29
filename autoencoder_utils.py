"""Shared loading and validation for the autoencoder pipeline."""
import hashlib
import warnings
from pathlib import Path

import torch

from autoencoder_model import Autoencoder

DEFAULT_DATA = Path("data/autoencoder/dual_dot_d100_dataset.pt")


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    X = data["X"]
    if X.ndim != 2 or not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("Expected finite floating-point matrix X")
    words = data["words"]
    if len(words) != len(X) or len(set(words)) != len(words):
        raise ValueError("Words must be unique and aligned with rows")
    ids = [data[key] for key in ("train_idx", "val_idx", "test_idx")]
    if any(i.ndim != 1 or i.numel() == 0 or i.dtype != torch.long for i in ids):
        raise ValueError("Each split must contain nonempty one-dimensional long IDs")
    if not torch.equal(torch.cat(ids).sort().values, torch.arange(len(X))):
        raise ValueError("Splits must cover every row exactly once without overlap")
    if data["normalization"] != "row_l2":
        raise ValueError("Expected row_l2 preprocessing")
    if not torch.allclose(X.norm(dim=1), torch.ones(len(X)), atol=1e-5):
        raise ValueError("Expected unit-length input rows")
    return data


def load_model(checkpoint_path, data_path, data):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_hash = checkpoint.get("data_sha256")
    if expected_hash:
        actual = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError("Dataset hash differs from the training dataset")
    else:
        warnings.warn("Legacy checkpoint has no dataset hash; verify the original dataset.")
    for key in ("normalization", "normalization_eps", "embedding_table"):
        if key in checkpoint and checkpoint[key] != data[key]:
            raise ValueError(f"Preprocessing mismatch: {key}")
    config = checkpoint["model_config"]
    if config["input_dim"] != data["X"].shape[1]:
        raise ValueError("Model input dimension does not match data")
    model = Autoencoder(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
