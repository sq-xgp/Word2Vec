"""从训练 checkpoint 读取 loss_history 并生成训练曲线。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


DEFAULT_CHECKPOINT_PATH = Path("checkpoints/word2vec_latest.pt")
DEFAULT_OUTPUT_PATH = Path("figures/word2vec_loss.png")


def load_loss_history(checkpoint_path: Path) -> list[float]:
    """安全读取 checkpoint 中非空、有限的 loss 历史。"""
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    loss_history = checkpoint.get("loss_history")
    if not isinstance(loss_history, list) or not loss_history:
        raise ValueError("checkpoint 中没有非空 loss_history")
    losses = []
    for loss in loss_history:
        if not isinstance(loss, (int, float)) or isinstance(loss, bool):
            raise ValueError("loss_history 必须只包含数值")
        loss = float(loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError("loss_history 不能包含 NaN 或无穷值")
        losses.append(loss)
    return losses


def save_loss_curve(loss_history: list[float], output_path: Path) -> Path:
    """保存 epoch 平均 loss 曲线。"""
    if not loss_history:
        raise ValueError("loss_history 不能为空")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(loss_history) + 1))

    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.plot(epochs, loss_history, marker="o", linewidth=2, color="#2563eb")
    axes.scatter(
        [epochs[0], epochs[-1]],
        [loss_history[0], loss_history[-1]],
        color=["#dc2626", "#16a34a"],
        zorder=3,
    )
    axes.annotate(
        f"start: {loss_history[0]:.4f}",
        (epochs[0], loss_history[0]),
        xytext=(10, -18),
        textcoords="offset points",
        va="top",
    )
    axes.annotate(
        f"end: {loss_history[-1]:.4f}",
        (epochs[-1], loss_history[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
    )
    axes.set_title("Word2Vec training loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Mean SGNS loss")
    axes.set_xticks(epochs)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="从 checkpoint 生成训练 loss 曲线")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="训练 checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="图片输出路径",
    )
    args = parser.parse_args()

    losses = load_loss_history(args.checkpoint)
    output_path = save_loss_curve(losses, args.output)
    print(f"Loss figure: {output_path}")


if __name__ == "__main__":
    main()
