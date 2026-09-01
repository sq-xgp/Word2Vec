"""把 Leipzig 原始句子语料准备成 Word2Vec 训练所需的数据。"""

from pathlib import Path

from analyze_corpus import read_leipzig_sentences
from preprocess import build_vocab, clean_text, tokens_to_ids
from sampling import FrequentWordSubsampler, build_negative_distribution


def prepare_training_data(
    sentence_file: Path,
    min_count: int = 5,
    subsampling_threshold: float = 1e-4,
    seed: int | None = 42,
) -> dict:
    """清洗语料、建立词表、降采样，并保留每个句子的 token ID。"""
    cleaned_sentences = []
    all_tokens = []
    for sentence in read_leipzig_sentences(sentence_file):
        tokens = clean_text(sentence)
        cleaned_sentences.append(tokens)
        all_tokens.extend(tokens)

    if not all_tokens:
        raise ValueError("语料清洗后没有 token，无法准备训练数据")

    word_to_id, id_to_word, word_counts = build_vocab(
        all_tokens,
        min_count=min_count,
    )
    if not word_to_id:
        raise ValueError("min_count 过滤后词表为空")

    negative_sampling_probs = build_negative_distribution(
        word_counts,
        word_to_id,
    )
    subsampler = FrequentWordSubsampler(
        word_counts,
        threshold=subsampling_threshold,
        seed=seed,
    )

    sentence_token_ids = []
    for tokens in cleaned_sentences:
        sampled_tokens = subsampler.sample(tokens)
        token_ids = tokens_to_ids(sampled_tokens, word_to_id)
        sentence_token_ids.append(token_ids)

    return {
        "sentence_token_ids": sentence_token_ids,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "word_counts": word_counts,
        "negative_sampling_probs": negative_sampling_probs,
    }
