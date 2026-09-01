"""推理：使用训练好的词向量查询相似词。"""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from model import SkipGramNegSampling


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")


def load_trained_model(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> tuple[
    SkipGramNegSampling,
    dict[str, int],
    dict[int, str],
    dict,
]:
    """从 checkpoint 恢复推理模型、双向词表和训练记录。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    selected_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=True,
    )
    required_keys = {
        "model_state_dict",
        "word_to_id",
        "config",
    }
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError("checkpoint 缺少模型、词表或配置")

    config = checkpoint["config"]
    word_to_id = checkpoint["word_to_id"]
    if not isinstance(config, dict) or not isinstance(word_to_id, dict):
        raise ValueError("checkpoint 中的配置或词表格式不正确")
    if "vocab_size" not in config or "embedding_dim" not in config:
        raise ValueError("checkpoint 配置缺少 vocab_size 或 embedding_dim")
    if len(word_to_id) != config["vocab_size"]:
        raise ValueError("checkpoint 的词表大小与模型配置不一致")

    model = SkipGramNegSampling(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_word = {word_id: word for word, word_id in word_to_id.items()}
    return model, word_to_id, id_to_word, checkpoint


def find_similar_words(
    query_word: str,
    model: SkipGramNegSampling,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """按中心词向量的余弦相似度返回 top-k 相似词。"""
    if query_word not in word_to_id:
        raise KeyError(f"查询词不在词表中：{query_word!r}")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须是正整数")

    model.eval()
    with torch.no_grad():
        embeddings = model.center_embeddings.weight
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        query_id = word_to_id[query_word]
        query_vector = normalized_embeddings[query_id]
        similarities = normalized_embeddings @ query_vector

        similarities[query_id] = -torch.inf
        result_count = min(top_k, len(word_to_id) - 1)
        if result_count == 0:
            return []
        scores, word_ids = torch.topk(similarities, k=result_count)

    return [
        (id_to_word[word_id], score)
        for word_id, score in zip(word_ids.tolist(), scores.tolist())
    ]


def query_checkpoint(
    checkpoint_path: Path,
    query_word: str,
    top_k: int = 5,
    device: str | torch.device = "cpu",
) -> list[tuple[str, float]]:
    """加载一个 checkpoint，并查询指定单词的相似词。"""
    model, word_to_id, id_to_word, _ = load_trained_model(
        checkpoint_path,
        device=device,
    )
    return find_similar_words(
        query_word,
        model,
        word_to_id,
        id_to_word,
        top_k=top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 Word2Vec 相似词")
    parser.add_argument("query_word", help="要查询的英文单词")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="训练 checkpoint 路径",
    )
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    args = parser.parse_args()

    results = query_checkpoint(
        args.checkpoint,
        args.query_word,
        top_k=args.top_k,
    )
    for word, score in results:
        print(f"{word}\t{score:.6f}")


if __name__ == "__main__":
    main()
