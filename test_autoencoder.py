"""Behavioral regression tests for the autoencoder workflow."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import Autoencoder
from autoencoder_utils import load_data, load_model
from compare_autoencoder_pca import fit_pca, neighbor_retention
from model import SkipGramNegSampling
from train_autoencoder import run_epoch

ROOT = Path(__file__).resolve().parent


class AutoencoderTests(unittest.TestCase):
    def test_validation_is_weighted_and_does_not_update_parameters(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        X = torch.tensor([[1.0], [1.0], [4.0]])
        before = model.weight.detach().clone()
        actual = run_epoch(model, DataLoader(TensorDataset(X), batch_size=2),
                           torch.device("cpu"))
        self.assertAlmostEqual(actual, 6.0)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertIsNone(model.weight.grad)

    def test_neighbor_retention_excludes_self(self):
        # These distinct unit vectors have known nearest neighbors: 0<->1; 2->1.
        X = torch.tensor([[1., 0.], [0.8, 0.6], [0., 1.]])
        Z = X.clone()
        self.assertEqual(neighbor_retention(X, Z, torch.arange(3), k=1), 1.0)
        Z = torch.tensor([[0., 0.], [10., 0.], [0.1, 0.]])
        self.assertEqual(neighbor_retention(X, Z, torch.tensor([0]), k=1), 0.0)

    def test_pca_reconstructs_plane_and_preserves_discarded_error(self):
        train = torch.tensor([[3., 0., 7.], [-3., 0., 7.],
                              [0., 2., 7.], [0., -2., 7.]])
        mean, W = fit_pca(train)
        reconstructed = (train - mean) @ W @ W.T + mean
        torch.testing.assert_close(reconstructed, train)
        test = torch.tensor([[1., 1., 10.]])
        reconstructed = (test - mean) @ W @ W.T + mean
        self.assertAlmostEqual((reconstructed - test).square().mean().item(), 3.0)

    def test_cli_pipeline_and_dataset_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source, prepared = tmp / "word2vec.pt", tmp / "data.pt"
            torch.manual_seed(42)
            source_model = SkipGramNegSampling(40, 100, "dual", "dot")
            torch.save({
                "config": {"vocab_size": 40, "embedding_dim": 100,
                           "embedding_mode": "dual", "score_mode": "dot"},
                "word_to_id": {f"word_{i}": i for i in range(40)},
                "model_state_dict": source_model.state_dict(),
            }, source)

            def run(script, *args):
                result = subprocess.run(
                    [sys.executable, str(ROOT / script), *map(str, args)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("prepare_autoencoder_data.py", "--checkpoint", source,
                "--output", prepared)
            data = load_data(prepared)
            expected = torch.nn.functional.normalize(
                source_model.center_embeddings.weight.detach(), dim=1)
            torch.testing.assert_close(data["X"], expected)
            self.assertEqual(data["words"][5], "word_5")
            self.assertEqual([len(data[k]) for k in
                              ("train_idx", "val_idx", "test_idx")], [32, 4, 4])
            second = tmp / "second.pt"
            run("prepare_autoencoder_data.py", "--checkpoint", source,
                "--output", second)
            self.assertTrue(torch.equal(load_data(second)["train_idx"], data["train_idx"]))

            run("train_autoencoder.py", "--data", prepared, "--output-root",
                tmp / "runs", "--epochs", "2", "--device", "cpu")
            best = next((tmp / "runs").glob("*/best.pt"))
            model, checkpoint = load_model(best, prepared, data)
            run("infer_autoencoder.py", "--data", prepared, "--checkpoint", best)
            encoded = torch.load(best.parent / "embeddings_2d.pt", weights_only=True)
            with torch.no_grad():
                torch.testing.assert_close(encoded["Z"], model.encoder(data["X"]))
            self.assertEqual(encoded["words"], data["words"])
            run("compare_autoencoder_pca.py", "--data", prepared, "--checkpoint", best)
            result = json.loads((best.parent / "comparison.json").read_text())
            self.assertEqual(result["test_words"], 4)
            self.assertTrue(0 <= result["autoencoder"]["neighbor_retention"] <= 1)
            run("visualize_autoencoder.py", "--run-dir", best.parent)
            self.assertTrue((best.parent / "plots/embeddings_2d.png").is_file())
            self.assertTrue((best.parent / "plots/loss_curve.png").is_file())

            # A changed dataset must not silently be used with a hashed checkpoint.
            altered = tmp / "altered.pt"
            changed = dict(data)
            changed["words"] = list(reversed(data["words"]))
            torch.save(changed, altered)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_model(best, altered, load_data(altered))
            # Original server checkpoints did not have a hash: keep them loadable.
            checkpoint.pop("data_sha256")
            legacy = tmp / "legacy.pt"
            torch.save(checkpoint, legacy)
            with self.assertWarns(UserWarning):
                load_model(legacy, prepared, data)
            bad = dict(data)
            bad["val_idx"] = bad["train_idx"][:4]
            torch.save(bad, tmp / "bad.pt")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_data(tmp / "bad.pt")


if __name__ == "__main__":
    unittest.main()
