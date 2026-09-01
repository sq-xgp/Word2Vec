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
from torch.nn import functional as F

from inference import load_trained_model
from model import SkipGramNegSampling


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")
DEFAULT_OUTPUT_DIR = Path("figures")


def _normalized_embeddings_for_ids(
    model: SkipGramNegSampling,
    selected_ids: list[int],
) -> np.ndarray:
    """返回指定词编号的 L2 归一化中心词向量。"""
    return (
        F.normalize(model.center_embeddings.weight[selected_ids], p=2, dim=1)
        .detach()
        .cpu()
        .numpy()
    )


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
    embeddings = _normalized_embeddings_for_ids(model, selected_ids)
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
        title=f"Word2Vec embeddings — normalized PCA ({len(words)} words)",
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
    embeddings = _normalized_embeddings_for_ids(model, selected_ids)
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
        title=f"Word2Vec embeddings — normalized t-SNE ({len(words)} words)",
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


def parse_word_groups(group_specs: list[str]) -> dict[str, list[str]]:
    """解析重复的 `标签:word1,word2` 参数，并拒绝空组和重复词。"""
    groups = {}
    seen_words = set()
    for group_spec in group_specs:
        label, separator, words_text = group_spec.partition(":")
        words = [word.strip() for word in words_text.split(",") if word.strip()]
        if not separator or not label.strip() or not words:
            raise ValueError("每个 --group 必须是 标签:word1,word2 格式")
        label = label.strip()
        if label in groups:
            raise ValueError(f"分组标签重复：{label!r}")
        if len(set(words)) != len(words):
            raise ValueError(f"分组 {label!r} 内有重复单词")
        repeated_words = seen_words.intersection(words)
        if repeated_words:
            raise ValueError(f"单词出现在多个分组中：{sorted(repeated_words)}")
        groups[label] = words
        seen_words.update(words)
    return groups


def _reduce_selected_words(
    model: SkipGramNegSampling,
    word_to_id: dict[str, int],
    words: list[str],
    method: str,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> np.ndarray:
    """按给定顺序选择词，并用 PCA 或 t-SNE 降到二维。"""
    missing_words = [word for word in words if word not in word_to_id]
    if missing_words:
        raise KeyError(f"以下单词不在词表中：{missing_words}")
    selected_ids = [word_to_id[word] for word in words]
    embeddings = _normalized_embeddings_for_ids(model, selected_ids)
    if method == "pca":
        if len(words) < 2:
            raise ValueError("PCA 分组图至少需要 2 个词")
        return PCA(n_components=2).fit_transform(embeddings)
    if method == "tsne":
        if len(words) < 3:
            raise ValueError("t-SNE 分组图至少需要 3 个词")
        if (
            not isinstance(perplexity, (int, float))
            or isinstance(perplexity, bool)
            or not np.isfinite(perplexity)
            or not 0 < perplexity < len(words)
        ):
            raise ValueError("perplexity 必须大于 0 且小于分组词总数")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not find the number of physical cores",
                category=UserWarning,
            )
            return TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                max_iter=1000,
                random_state=random_state,
                n_jobs=1,
            ).fit_transform(embeddings)
    raise ValueError("method 必须是 'pca' 或 'tsne'")


def _save_grouped_scatter(
    word_groups: dict[str, list[str]],
    coordinates: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """按语义组着色并保存带单词标签的二维散点图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 9))
    offset = 0
    colors = plt.get_cmap("tab10")
    for group_index, (label, words) in enumerate(word_groups.items()):
        group_coordinates = coordinates[offset : offset + len(words)]
        color = colors(group_index % 10)
        axes.scatter(
            group_coordinates[:, 0],
            group_coordinates[:, 1],
            s=42,
            alpha=0.8,
            color=color,
            label=label,
        )
        for word, (x_coordinate, y_coordinate) in zip(words, group_coordinates):
            axes.annotate(
                word,
                (x_coordinate, y_coordinate),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        offset += len(words)
    axes.set_title(title)
    axes.set_xlabel("Dimension 1")
    axes.set_ylabel("Dimension 2")
    axes.grid(alpha=0.2)
    axes.legend(title="Semantic group")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def visualize_word_groups(
    checkpoint_path: Path,
    output_dir: Path,
    word_groups: dict[str, list[str]],
    perplexity: float = 5.0,
) -> tuple[Path, Path]:
    """为用户指定的语义词组生成归一化 PCA 和 t-SNE 彩色图。"""
    if not word_groups:
        raise ValueError("word_groups 不能为空")
    model, word_to_id, _, checkpoint = load_trained_model(checkpoint_path)
    words = [word for group_words in word_groups.values() for word in group_words]
    random_state = checkpoint.get("config", {}).get("seed", 42)
    pca_coordinates = _reduce_selected_words(
        model,
        word_to_id,
        words,
        method="pca",
    )
    tsne_coordinates = _reduce_selected_words(
        model,
        word_to_id,
        words,
        method="tsne",
        perplexity=perplexity,
        random_state=random_state,
    )
    output_dir = Path(output_dir)
    pca_path = _save_grouped_scatter(
        word_groups,
        pca_coordinates,
        output_dir / "word2vec_grouped_pca.png",
        title=f"Word2Vec semantic groups — normalized PCA ({len(words)} words)",
    )
    tsne_path = _save_grouped_scatter(
        word_groups,
        tsne_coordinates,
        output_dir / "word2vec_grouped_tsne.png",
        title=f"Word2Vec semantic groups — normalized t-SNE ({len(words)} words)",
    )
    return pca_path, tsne_path


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
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="语义分组，格式为 标签:word1,word2；可以重复传入",
    )
    args = parser.parse_args()

    if args.group:
        pca_path, tsne_path = visualize_word_groups(
            args.checkpoint,
            args.output_dir,
            parse_word_groups(args.group),
            perplexity=args.perplexity,
        )
    else:
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
