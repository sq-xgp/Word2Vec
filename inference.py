"""Query one checkpoint or compare several Word2Vec checkpoints."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from model import SkipGramNegSampling


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")


def load_trained_model(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> tuple[SkipGramNegSampling, dict[str, int], dict[int, str], dict]:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=True
    )
    config = checkpoint["config"]
    word_to_id = checkpoint["word_to_id"]
    model = SkipGramNegSampling(
        config["vocab_size"],
        config["embedding_dim"],
        embedding_mode=config.get("embedding_mode", "dual"),
        score_mode=config.get("score_mode", "dot"),
        temperature=config.get("temperature", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, word_to_id, {i: w for w, i in word_to_id.items()}, checkpoint


def find_similar_words(
    query_word: str,
    model: SkipGramNegSampling,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    top_k: int = 10,
    similarity: str = "cosine",
) -> list[tuple[str, float]]:
    """Rank center-table vectors using cosine similarity or dot product."""
    if query_word not in word_to_id:
        raise KeyError(f"查询词不在词表中：{query_word!r}")
    embeddings = model.center_embeddings.weight
    query_id = word_to_id[query_word]
    with torch.no_grad():
        if similarity == "cosine":
            vectors = F.normalize(embeddings, p=2, dim=1)
            scores = vectors @ vectors[query_id]
        elif similarity == "dot":
            scores = embeddings @ embeddings[query_id]
        else:
            raise ValueError("similarity 必须是 'dot' 或 'cosine'")
        scores[query_id] = -torch.inf
        values, ids = torch.topk(scores, min(top_k, len(word_to_id) - 1))
    return [(id_to_word[i], value) for i, value in zip(ids.tolist(), values.tolist())]


def query_checkpoint(
    checkpoint_path: Path,
    query_word: str,
    top_k: int = 10,
    device: str | torch.device = "cpu",
    similarity: str = "cosine",
) -> list[tuple[str, float]]:
    model, word_to_id, id_to_word, _ = load_trained_model(checkpoint_path, device)
    return find_similar_words(
        query_word, model, word_to_id, id_to_word, top_k, similarity
    )


def compare_checkpoints(
    checkpoint_paths: list[Path],
    query_word: str,
    top_k: int = 10,
    similarity: str = "cosine",
) -> dict[Path, list[tuple[str, float]] | str]:
    """Run the same query against every checkpoint without requiring equal vocabularies."""
    results = {}
    for path in checkpoint_paths:
        try:
            results[path] = query_checkpoint(path, query_word, top_k, similarity=similarity)
        except (KeyError, FileNotFoundError) as error:
            results[path] = str(error)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="查询或对比 Word2Vec 模型")
    parser.add_argument("query_word")
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+", default=[DEFAULT_CHECKPOINT_PATH]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity", choices=("dot", "cosine"), default="cosine")
    args = parser.parse_args()

    for path, result in compare_checkpoints(
        args.checkpoint, args.query_word, args.top_k, args.similarity
    ).items():
        print(f"\n=== {path} ===")
        if isinstance(result, str):
            print(result)
        else:
            for word, score in result:
                print(f"{word}\t{score:.6f}")


if __name__ == "__main__":
    main()
