"""可视化：使用 PCA / t-SNE 将训练好的词向量降至二维并绘图。"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from inference import load_trained_model
from model import SkipGramNegSampling


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")
DEFAULT_OUTPUT_DIR = Path("figures")


def reduce_embeddings_with_pca(
    model: SkipGramNegSampling,
    id_to_word: dict[int, str],
    num_words: int = 100,
) -> tuple[list[str], np.ndarray]:
    """选取编号最小的常见词，并把中心词向量降至二维。"""
    if type(num_words) is not int or num_words < 2:
        raise ValueError("num_words 必须是大于等于 2 的整数")
    if model.embedding_dim < 2:
        raise ValueError("PCA 二维可视化要求 embedding_dim 至少为 2")

    selected_count = min(num_words, model.vocab_size)
    selected_ids = list(range(selected_count))
    if any(word_id not in id_to_word for word_id in selected_ids):
        raise ValueError("id_to_word 缺少要可视化的词编号")

    words = [id_to_word[word_id] for word_id in selected_ids]
    embeddings = (
        model.center_embeddings.weight[selected_ids]
        .detach()
        .cpu()
        .numpy()
    )
    coordinates = PCA(n_components=2).fit_transform(embeddings)
    return words, coordinates


def save_pca_plot(
    model: SkipGramNegSampling,
    id_to_word: dict[int, str],
    output_path: Path,
    num_words: int = 100,
) -> Path:
    """生成带单词标签的 PCA 散点图并保存为图片。"""
    words, coordinates = reduce_embeddings_with_pca(
        model,
        id_to_word,
        num_words=num_words,
    )
    return _save_labeled_scatter(
        words,
        coordinates,
        output_path,
        title=f"Word2Vec embeddings — PCA ({len(words)} words)",
        x_label="Principal component 1",
        y_label="Principal component 2",
    )


def reduce_embeddings_with_tsne(
    model: SkipGramNegSampling,
    id_to_word: dict[int, str],
    num_words: int = 100,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """选取常见词，并使用固定随机种子的 t-SNE 降至二维。"""
    if type(num_words) is not int or num_words < 3:
        raise ValueError("t-SNE 的 num_words 必须是大于等于 3 的整数")

    selected_count = min(num_words, model.vocab_size)
    if selected_count < 3:
        raise ValueError("t-SNE 至少需要 3 个词")
    if (
        not isinstance(perplexity, (int, float))
        or isinstance(perplexity, bool)
        or not np.isfinite(perplexity)
        or not 0 < perplexity < selected_count
    ):
        raise ValueError("perplexity 必须大于 0 且小于实际词数")
    if type(random_state) is not int:
        raise ValueError("random_state 必须是整数")

    selected_ids = list(range(selected_count))
    if any(word_id not in id_to_word for word_id in selected_ids):
        raise ValueError("id_to_word 缺少要可视化的词编号")

    words = [id_to_word[word_id] for word_id in selected_ids]
    embeddings = (
        model.center_embeddings.weight[selected_ids]
        .detach()
        .cpu()
        .numpy()
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores",
            category=UserWarning,
        )
        coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=random_state,
            n_jobs=1,
        ).fit_transform(embeddings)
    return words, coordinates


def save_tsne_plot(
    model: SkipGramNegSampling,
    id_to_word: dict[int, str],
    output_path: Path,
    num_words: int = 100,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Path:
    """生成带单词标签的 t-SNE 散点图并保存为图片。"""
    words, coordinates = reduce_embeddings_with_tsne(
        model,
        id_to_word,
        num_words=num_words,
        perplexity=perplexity,
        random_state=random_state,
    )
    return _save_labeled_scatter(
        words,
        coordinates,
        output_path,
        title=f"Word2Vec embeddings — t-SNE ({len(words)} words)",
        x_label="t-SNE dimension 1",
        y_label="t-SNE dimension 2",
    )


def _save_labeled_scatter(
    words: list[str],
    coordinates: np.ndarray,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    """保存 PCA 和 t-SNE 共用的带标签散点图。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 9))
    axes.scatter(coordinates[:, 0], coordinates[:, 1], s=24, alpha=0.75)
    for word, (x_coordinate, y_coordinate) in zip(words, coordinates):
        axes.annotate(
            word,
            (x_coordinate, y_coordinate),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def visualize_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    num_words: int = 100,
    perplexity: float = 30.0,
) -> tuple[Path, Path]:
    """加载 checkpoint，并保存同一批常见词的 PCA 和 t-SNE 图片。"""
    model, _, id_to_word, checkpoint = load_trained_model(checkpoint_path)
    random_state = checkpoint.get("config", {}).get("seed", 42)
    output_dir = Path(output_dir)
    pca_path = save_pca_plot(
        model,
        id_to_word,
        output_dir / "word2vec_pca.png",
        num_words=num_words,
    )
    tsne_path = save_tsne_plot(
        model,
        id_to_word,
        output_dir / "word2vec_tsne.png",
        num_words=num_words,
        perplexity=perplexity,
        random_state=random_state,
    )
    return pca_path, tsne_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Word2Vec 词向量图")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="训练 checkpoint 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图片输出目录",
    )
    parser.add_argument("--num-words", type=int, default=100, help="绘制的常见词数量")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    args = parser.parse_args()

    pca_path, tsne_path = visualize_checkpoint(
        args.checkpoint,
        args.output_dir,
        num_words=args.num_words,
        perplexity=args.perplexity,
    )
    print(f"PCA figure: {pca_path}")
    print(f"t-SNE figure: {tsne_path}")


if __name__ == "__main__":
    main()
