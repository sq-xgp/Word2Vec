"""Focused tests for configurable models and checkpoint comparison."""

import tempfile
import unittest
from pathlib import Path

import torch

from inference import find_similar_words, load_trained_model
from model import SkipGramNegSampling
from train import save_checkpoint, split_sentences
from dataset import SentenceWord2VecDataset


class ConfigurableWord2VecTests(unittest.TestCase):
    def test_all_model_modes_produce_expected_shapes_and_gradients(self):
        centers = torch.tensor([0, 1])
        contexts = torch.tensor([1, 2])
        negatives = torch.tensor([[2, 3, 4], [0, 3, 4]])
        for embedding_mode in ("dual", "shared"):
            for score_mode in ("dot", "cosine"):
                model = SkipGramNegSampling(5, 8, embedding_mode, score_mode)
                positive, negative = model(centers, contexts, negatives)
                self.assertEqual(positive.shape, (2,))
                self.assertEqual(negative.shape, (2, 3))
                model.compute_loss(positive, negative).backward()
                self.assertIsNotNone(model.center_embeddings.weight.grad)
                if embedding_mode == "shared":
                    self.assertIs(model.center_embeddings, model.context_embeddings)

    def test_cosine_temperature_scales_logits(self):
        left = torch.tensor([[1.0, 0.0]])
        right = torch.tensor([[0.5, 0.5]])
        unscaled = SkipGramNegSampling(2, 2, score_mode="cosine", temperature=1.0)
        scaled = SkipGramNegSampling(2, 2, score_mode="cosine", temperature=0.1)
        self.assertTrue(torch.allclose(scaled._score(left, right), 10 * unscaled._score(left, right)))

    def test_sentence_split_is_reproducible_and_disjoint(self):
        sentences = [[index] for index in range(20)]
        first = split_sentences(sentences, 0.2, 42)
        second = split_sentences(sentences, 0.2, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 16)
        self.assertEqual(len(first[1]), 4)
        self.assertFalse({s[0] for s in first[0]} & {s[0] for s in first[1]})

    def test_new_checkpoint_round_trip(self):
        model = SkipGramNegSampling(4, 6, "shared", "cosine")
        optimizer = torch.optim.Adam(model.parameters())
        config = {
            "vocab_size": 4,
            "embedding_dim": 6,
            "embedding_mode": "shared",
            "score_mode": "cosine",
            "temperature": 0.1,
        }
        words = {"red": 0, "blue": 1, "green": 2, "black": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, 1, words, config, [1.0], [0.9], 0.9)
            loaded, word_to_id, id_to_word, checkpoint = load_trained_model(path)
            result = find_similar_words("red", loaded, word_to_id, id_to_word, 2)
            self.assertEqual(len(result), 2)
            self.assertEqual(checkpoint["validation_loss_history"], [0.9])

    def test_validation_negatives_are_stable_between_epochs(self):
        dataset = SentenceWord2VecDataset(
            [[0, 1], [2, 3]], [0.25] * 4, window_size=1,
            num_negatives=2, seed=42, shuffle_sentences=False,
            resample_negatives=False,
        )
        first = [(c.item(), x.item(), n.tolist()) for c, x, n in dataset]
        second = [(c.item(), x.item(), n.tolist()) for c, x, n in dataset]
        self.assertEqual(first, second)

    def test_external_positive_contexts_are_never_sampled_as_negatives(self):
        all_contexts = {0: {1, 2}, 1: {0}, 2: {0}}
        dataset = SentenceWord2VecDataset(
            [[0, 1]], [0.25] * 4, window_size=1, num_negatives=20,
            seed=42, shuffle_sentences=False, resample_negatives=False,
            center_to_positive_contexts=all_contexts,
        )
        samples = list(dataset)
        negatives_for_zero = samples[0][2].tolist()
        self.assertFalse({0, 1, 2} & set(negatives_for_zero))


if __name__ == "__main__":
    unittest.main()
