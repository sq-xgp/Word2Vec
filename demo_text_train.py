"""从一句英文文本开始，在 CPU 上完成预处理和 5 轮训练，不保存模型。"""

import torch
from torch.utils.data import DataLoader

from dataset import Word2VecDataset
from model import SkipGramNegSampling
from preprocess import clean_text, build_vocab, tokens_to_ids, generate_skipgram_pairs
from sampling import build_negative_distribution, subsample_frequent_words
from train import train_one_epoch


def main() -> None:
    # 只用一句话，暂不引入多句文本的句子边界处理。
    text = "Cats chase mice while dogs chase balls and birds build nests near trees."
    tokens = clean_text(text)
    word_to_id, id_to_word, word_counts = build_vocab(tokens, min_count=1)

    # 用降采样前的词频构建概率，之后不重新统计词频或建立词表。
    negative_sampling_probs = build_negative_distribution(word_counts, word_to_id)
    kept_tokens = subsample_frequent_words(
        tokens,
        word_counts,
        threshold=0.05,              # 仅用于这句短文本的演示，不是正式训练设置。
        seed=42,
    )
    token_ids = tokens_to_ids(kept_tokens, word_to_id)
    skipgram_pairs = generate_skipgram_pairs(token_ids, window_size=2)
    if not skipgram_pairs:
        raise ValueError("没有生成正样本，请检查文本、min_count 和降采样阈值")

    dataset = Word2VecDataset(
        skipgram_pairs,
        negative_sampling_probs,
        num_negatives=3,
        seed=42,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    torch.manual_seed(42)
    model = SkipGramNegSampling(vocab_size=len(word_to_id), embedding_dim=128)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    num_epochs = 5

    print("Tokens:", tokens)
    print("Vocabulary:", word_to_id)
    print("Kept tokens:", kept_tokens)
    print("Token IDs:", token_ids)
    print("Number of positive pairs:", len(dataset))
    print("Number of batches per epoch:", len(dataloader))
    # 把前几个编号对还原成单词，方便核对预处理结果。
    print("First positive word pairs:", [
        (id_to_word[center], id_to_word[context])
        for center, context in skipgram_pairs[:5]
    ])

    # 预处理只做一次；每轮继续训练，并在读取样本时重新抽取负样本。
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, dataloader)
        print(f"Epoch {epoch + 1} mean loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    main()
