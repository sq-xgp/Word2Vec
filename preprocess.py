"""数据预处理：数据清洗、词表、token → id 和 Skip-gram 正样本构造。"""

import re
from collections import Counter


def clean_text(text: str) -> list[str]:
    """把英文文本转成小写 token 列表。

    当前学习版只保留英文字母和数字，标点及其他字符作为分隔符。
    例如 word2vec 会保留为一个 token；暂不处理中文分词。
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def build_vocab(
    tokens: list[str], min_count: int = 1
) -> tuple[dict[str, int], dict[int, str], dict[str, int]]:
    """从清洗后的 token 列表建立词表，返回两个映射和保留词的词频。

    min_count 是保留词的最低出现次数，必须是正整数。
    编号从 0 开始，按词频从高到低分配；同频词按首次出现顺序排列。
    返回顺序为 word_to_id、id_to_word、word_counts。
    空输入或所有词被过滤时，返回三个空字典。
    """
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count 必须是大于等于 1 的整数")

    counts = Counter(tokens)
    word_counts = {
        word: count
        for word, count in counts.most_common()
        if count >= min_count
    }

    word_to_id = {word: index for index, word in enumerate(word_counts)}
    id_to_word = {index: word for word, index in word_to_id.items()}
    return word_to_id, id_to_word, word_counts


def tokens_to_ids(tokens: list[str], word_to_id: dict[str, int]) -> list[int]:
    """按原顺序把 token 转为词表中的编号，跳过不在词表中的词。

    保留重复词的每次出现，不修改输入列表或词表，也不新增未知词编号。
    空输入或没有词命中词表时，返回空列表。
    """
    token_ids = []
    for token in tokens:
        if token in word_to_id:
            token_ids.append(word_to_id[token])
    return token_ids


def generate_skipgram_pairs(
    token_ids: list[int], window_size: int = 2
) -> list[tuple[int, int]]:
    """从一个编号序列生成 (中心词编号, 上下文词编号) 正样本。

    window_size 是中心词左右各自最多查看的位置数，必须是正整数。
    使用固定窗口，在序列两端截断；仅排除中心位置本身。
    不同位置的相同词可以配对，重复出现的样本也会保留。
    按中心位置从左到右遍历，每个中心的上下文也从左到右遍历。
    空序列或只有一个词时，返回空列表。不修改输入序列。

    此学习版返回完整列表，适合小规模数据；不自动识别句子边界。
    """
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 1
    ):
        raise ValueError("window_size 必须是大于等于 1 的整数")

    pairs = []
    for center_index, center_id in enumerate(token_ids):
        start = max(0, center_index - window_size)
        end = min(len(token_ids), center_index + window_size + 1)
        for context_index in range(start, end):
            if context_index != center_index:
                pairs.append((center_id, token_ids[context_index]))
    return pairs
