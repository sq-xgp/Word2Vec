"""读取 Leipzig 句子语料并统计规模，不生成训练样本或训练模型。"""

from collections import Counter
from pathlib import Path

from preprocess import clean_text
from sampling import FrequentWordSubsampler


SENTENCE_FILE = Path("data/raw/eng_wikipedia_2016_10K-sentences.txt")
MIN_COUNTS = (2, 5, 10)
SELECTED_MIN_COUNT = 5
SUBSAMPLING_THRESHOLDS = (1e-3, 1e-4, 1e-5)
RANDOM_SEED = 42
WINDOW_SIZE = 2


def read_leipzig_sentences(sentence_file: Path):
    """逐行返回句子文本，并验证“句子编号 + 制表符 + 句子”的格式。"""
    seen_ids = set()
    with sentence_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = line.rstrip("\r\n")
            sentence_id_text, separator, sentence = row.partition("\t")
            if not separator or not sentence_id_text.isdigit() or not sentence.strip():
                raise ValueError(f"第 {line_number} 行不符合 Leipzig 句子格式")

            sentence_id = int(sentence_id_text)
            if sentence_id in seen_ids:
                raise ValueError(f"第 {line_number} 行的句子编号重复：{sentence_id}")
            seen_ids.add(sentence_id)
            yield sentence


def count_skipgram_pairs(sequence_length: int, window_size: int) -> int:
    """只根据句子长度计算有向 Skip-gram 正样本数，不创建样本列表。"""
    pair_count = 0
    for center_index in range(sequence_length):
        left_count = min(window_size, center_index)
        right_count = min(window_size, sequence_length - center_index - 1)
        pair_count += left_count + right_count
    return pair_count


def analyze_corpus(sentence_file: Path) -> dict:
    """流式读取语料，统计过滤和不同降采样阈值下的训练规模。"""
    if not sentence_file.is_file():
        raise FileNotFoundError(f"找不到句子文件：{sentence_file}")

    word_counts = Counter()
    sentence_count = 0
    empty_after_cleaning = 0
    total_tokens = 0
    shortest_sentence = None
    longest_sentence = 0

    # 第一遍：建立全语料词频并统计清洗后的句子长度。
    for sentence in read_leipzig_sentences(sentence_file):
        tokens = clean_text(sentence)
        token_count = len(tokens)
        sentence_count += 1
        total_tokens += token_count
        word_counts.update(tokens)
        if token_count == 0:
            empty_after_cleaning += 1
        shortest_sentence = (
            token_count
            if shortest_sentence is None
            else min(shortest_sentence, token_count)
        )
        longest_sentence = max(longest_sentence, token_count)

    if sentence_count == 0 or total_tokens == 0:
        raise ValueError("语料清洗后没有可统计的 token")

    retained_words = {
        min_count: {
            word for word, count in word_counts.items() if count >= min_count
        }
        for min_count in MIN_COUNTS
    }
    retained_tokens = {min_count: 0 for min_count in MIN_COUNTS}
    positive_pairs = {min_count: 0 for min_count in MIN_COUNTS}

    # 第二遍：保留句子边界，估算过滤后的 token 和正样本数量。
    for sentence in read_leipzig_sentences(sentence_file):
        tokens = clean_text(sentence)
        for min_count in MIN_COUNTS:
            filtered_length = sum(
                token in retained_words[min_count] for token in tokens
            )
            retained_tokens[min_count] += filtered_length
            positive_pairs[min_count] += count_skipgram_pairs(
                filtered_length,
                WINDOW_SIZE,
            )

    selected_word_counts = {
        word: count
        for word, count in word_counts.items()
        if count >= SELECTED_MIN_COUNT
    }
    subsamplers = {
        threshold: FrequentWordSubsampler(
            selected_word_counts,
            threshold=threshold,
            seed=RANDOM_SEED,
        )
        for threshold in SUBSAMPLING_THRESHOLDS
    }
    subsampled_tokens = {threshold: 0 for threshold in SUBSAMPLING_THRESHOLDS}
    subsampled_pairs = {threshold: 0 for threshold in SUBSAMPLING_THRESHOLDS}
    subsampled_short_sentences = {
        threshold: 0 for threshold in SUBSAMPLING_THRESHOLDS
    }

    # 第三遍：每个阈值各复用一个 RNG，逐句降采样并保留句子边界。
    for sentence in read_leipzig_sentences(sentence_file):
        tokens = clean_text(sentence)
        for threshold in SUBSAMPLING_THRESHOLDS:
            kept_tokens = subsamplers[threshold].sample(tokens)
            kept_count = len(kept_tokens)
            subsampled_tokens[threshold] += kept_count
            subsampled_pairs[threshold] += count_skipgram_pairs(
                kept_count,
                WINDOW_SIZE,
            )
            if kept_count < 2:
                subsampled_short_sentences[threshold] += 1

    return {
        "file_size_bytes": sentence_file.stat().st_size,
        "sentence_count": sentence_count,
        "empty_after_cleaning": empty_after_cleaning,
        "total_tokens": total_tokens,
        "unique_words": len(word_counts),
        "shortest_sentence": shortest_sentence or 0,
        "longest_sentence": longest_sentence,
        "mean_sentence_length": total_tokens / sentence_count,
        "top_words": word_counts.most_common(10),
        "min_count_stats": {
            min_count: {
                "vocab_size": len(retained_words[min_count]),
                "retained_tokens": retained_tokens[min_count],
                "retained_percent": 100 * retained_tokens[min_count] / total_tokens,
                "positive_pairs": positive_pairs[min_count],
            }
            for min_count in MIN_COUNTS
        },
        "subsampling_stats": {
            threshold: {
                "retained_tokens": subsampled_tokens[threshold],
                "retained_percent_of_filtered": (
                    100
                    * subsampled_tokens[threshold]
                    / retained_tokens[SELECTED_MIN_COUNT]
                ),
                "positive_pairs": subsampled_pairs[threshold],
                "sentences_with_fewer_than_two_tokens": (
                    subsampled_short_sentences[threshold]
                ),
            }
            for threshold in SUBSAMPLING_THRESHOLDS
        },
    }


def main() -> None:
    stats = analyze_corpus(SENTENCE_FILE)
    print("Sentence file:", SENTENCE_FILE)
    print("File size:", stats["file_size_bytes"], "bytes")
    print("Sentences:", stats["sentence_count"])
    print("Empty after cleaning:", stats["empty_after_cleaning"])
    print("Tokens:", stats["total_tokens"])
    print("Unique words:", stats["unique_words"])
    print(
        "Sentence length (min / mean / max):",
        stats["shortest_sentence"],
        f"/ {stats['mean_sentence_length']:.2f} /",
        stats["longest_sentence"],
    )
    print("Top 10 words:", stats["top_words"])
    print("min_count | vocab | retained tokens | retained % | positive pairs")
    for min_count, values in stats["min_count_stats"].items():
        print(
            f"{min_count:>9} | {values['vocab_size']:>5} |"
            f" {values['retained_tokens']:>15} |"
            f" {values['retained_percent']:>9.2f}% |"
            f" {values['positive_pairs']:>14}"
        )
    print(
        f"Subsampling after min_count={SELECTED_MIN_COUNT}"
        f" with shared RNG seed={RANDOM_SEED}"
    )
    print("threshold | retained tokens | retained % | positive pairs | short sentences")
    for threshold, values in stats["subsampling_stats"].items():
        print(
            f"{threshold:>9.0e} | {values['retained_tokens']:>15} |"
            f" {values['retained_percent_of_filtered']:>9.2f}% |"
            f" {values['positive_pairs']:>14} |"
            f" {values['sentences_with_fewer_than_two_tokens']:>15}"
        )


if __name__ == "__main__":
    main()
