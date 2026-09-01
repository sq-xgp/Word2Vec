"""数据集：封装 PyTorch Dataset，向训练流程提供样本。"""

import math
import random
from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from sampling import CumulativeNegativeSampler


def iter_skipgram_pairs_by_sentence(
    sentence_token_ids: Iterable[list[int]],
    window_size: int = 2,
) -> Iterator[tuple[int, int]]:
    """按句子边界逐个产生 Skip-gram 正样本，不保存完整样本列表。"""
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size 必须是正整数")

    for token_ids in sentence_token_ids:
        for center_index, center_id in enumerate(token_ids):
            start = max(0, center_index - window_size)
            end = min(len(token_ids), center_index + window_size + 1)

            for context_index in range(start, end):
                if context_index != center_index:
                    context_id = token_ids[context_index]
                    yield center_id, context_id


class SentencePairDataset(IterableDataset):
    """保存句子编号，并在每次遍历时重新生成 Skip-gram 正样本。"""

    def __init__(
        self,
        sentence_token_ids: list[list[int]],
        window_size: int = 2,
        shuffle_sentences: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size 必须是正整数")
        if type(shuffle_sentences) is not bool:
            raise ValueError("shuffle_sentences 必须是布尔值")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed 必须是整数或 None")

        sentences = []
        for token_ids in sentence_token_ids:
            if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
                raise ValueError("句子中的词编号必须是非负整数")
            sentences.append(list(token_ids))

        self.sentence_token_ids = sentences
        self.window_size = window_size
        self.shuffle_sentences = shuffle_sentences
        self.sentence_rng = random.Random(seed)
        self.num_positive_pairs = sum(
            1
            for _ in iter_skipgram_pairs_by_sentence(
                self.sentence_token_ids,
                window_size=self.window_size,
            )
        )

    def __len__(self) -> int:
        """返回一轮遍历会动态产生的正样本数量。"""
        return self.num_positive_pairs

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """为本轮遍历创建新的正样本生成过程。"""
        yield from iter_skipgram_pairs_by_sentence(
            self._iter_sentence_token_ids(),
            window_size=self.window_size,
        )

    def _iter_sentence_token_ids(self) -> Iterator[list[int]]:
        """按本轮顺序逐个返回句子；需要时只打乱句子编号。"""
        sentence_indices = list(range(len(self.sentence_token_ids)))
        if self.shuffle_sentences:
            self.sentence_rng.shuffle(sentence_indices)

        for sentence_index in sentence_indices:
            yield self.sentence_token_ids[sentence_index]


def build_center_to_positive_contexts(
    skipgram_pairs: Iterable[tuple[int, int]],
) -> dict[int, set[int]]:
    """把全部已生成的正样本整理为中心词编号到正上下文编号集合的映射。

    相同中心词的上下文合并到一个集合；不会删除原列表中的重复样本。
    只记录输入中的有向关系，不自动补反向关系或排除中心词本身。
    空输入返回空字典；每对样本必须包含两个非负整数编号。
    此处不接收词表，编号是否超出词表范围由后续 Dataset 检查。
    """
    center_to_positive_contexts = {}
    for pair in skipgram_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("每个正样本必须包含中心词和上下文词两个编号")
        center_id, context_id = pair
        if (
            type(center_id) is not int
            or type(context_id) is not int
            or center_id < 0
            or context_id < 0
        ):
            raise ValueError("中心词和上下文词编号必须是非负整数")

        if center_id not in center_to_positive_contexts:
            center_to_positive_contexts[center_id] = set()
        center_to_positive_contexts[center_id].add(context_id)
    return center_to_positive_contexts


class SentenceWord2VecDataset(SentencePairDataset):
    """按句子动态产生中心词、正上下文和负样本张量。"""

    def __init__(
        self,
        sentence_token_ids: list[list[int]],
        negative_sampling_probs: list[float],
        window_size: int = 2,
        num_negatives: int = 5,
        seed: int | None = None,
        shuffle_sentences: bool = True,
    ) -> None:
        super().__init__(
            sentence_token_ids,
            window_size=window_size,
            shuffle_sentences=shuffle_sentences,
            seed=seed,
        )
        if type(num_negatives) is not int or num_negatives < 1:
            raise ValueError("num_negatives 必须是正整数")

        self.negative_sampling_probs = list(negative_sampling_probs)
        self.num_negatives = num_negatives
        self.negative_sampler = CumulativeNegativeSampler(
            self.negative_sampling_probs,
            seed=seed,
        )
        self.rng = self.negative_sampler.rng

        vocab_size = len(self.negative_sampling_probs)
        if any(
            token_id >= vocab_size
            for token_ids in self.sentence_token_ids
            for token_id in token_ids
        ):
            raise ValueError("句子中的词编号超出了概率列表对应的词表范围")

        self.center_to_positive_contexts = build_center_to_positive_contexts(
            iter_skipgram_pairs_by_sentence(
                self.sentence_token_ids,
                window_size=self.window_size,
            )
        )
        for center_id, context_ids in self.center_to_positive_contexts.items():
            excluded_ids = {center_id} | context_ids
            if self.negative_sampler.positive_probability_ids.issubset(excluded_ids):
                raise ValueError(
                    f"中心词 {center_id} 在排除正上下文后没有可用负样本"
                )

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """每轮重新产生正样本，并为每个正样本即时抽取负样本。"""
        for center_id, context_id in iter_skipgram_pairs_by_sentence(
            self._iter_sentence_token_ids(),
            window_size=self.window_size,
        ):
            excluded_ids = {
                center_id,
                *self.center_to_positive_contexts[center_id],
            }
            negative_ids = self.negative_sampler.sample(
                num_negatives=self.num_negatives,
                excluded_ids=excluded_ids,
            )

            center = torch.tensor(center_id, dtype=torch.long, device="cpu")
            context = torch.tensor(context_id, dtype=torch.long, device="cpu")
            negatives = torch.tensor(negative_ids, dtype=torch.long, device="cpu")
            yield center, context, negatives


class Word2VecDataset(Dataset):
    """保存 Skip-gram 正样本及负采样所需数据。

    按索引返回中心词、正上下文词和新抽取的负样本的编号张量。
    当前按单进程读取设计；多进程 DataLoader 的随机状态管理尚未处理。
    """

    def __init__(
        self,
        skipgram_pairs: list[tuple[int, int]],
        negative_sampling_probs: list[float],
        num_negatives: int = 5,
        seed: int | None = None,
    ) -> None:
        """复制输入，建立上下文映射，并准备独立的随机数生成器。

        概率下标就是词编号，概率须有限且非负，总和约为 1。
        每个中心词在排除自身和全部已知正上下文后，须仍有正概率候选词。
        合法概率列表配合空正样本列表可创建长度为 0 的数据集。
        初始化不抽取负样本，不改变全局随机状态。
        """
        super().__init__()
        if type(num_negatives) is not int or num_negatives < 1:
            raise ValueError("num_negatives 必须是正整数")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed 必须是整数或 None")
        if not negative_sampling_probs:
            raise ValueError("负采样概率列表不能为空")
        for probability in negative_sampling_probs:
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not math.isfinite(probability)
                or probability < 0
            ):
                raise ValueError("负采样概率必须是有限的非负数值")
        if not math.isclose(
            sum(negative_sampling_probs), 1.0, rel_tol=1e-6, abs_tol=1e-8
        ):
            raise ValueError("负采样概率总和必须约等于 1")

        contexts = build_center_to_positive_contexts(skipgram_pairs)
        vocab_size = len(negative_sampling_probs)
        positive_probability_ids = {
            word_id
            for word_id, probability in enumerate(negative_sampling_probs)
            if probability > 0
        }
        for center_id, context_ids in contexts.items():
            if center_id >= vocab_size or any(
                context_id >= vocab_size for context_id in context_ids
            ):
                raise ValueError("正样本中的词编号超出了概率列表对应的词表范围")
            excluded_ids = {center_id} | context_ids
            if positive_probability_ids.issubset(excluded_ids):
                raise ValueError(f"中心词 {center_id} 在排除正上下文后没有可用负样本")

        self.skipgram_pairs = [tuple(pair) for pair in skipgram_pairs]
        self.negative_sampling_probs = list(negative_sampling_probs)
        self.num_negatives = num_negatives
        self.center_to_positive_contexts = contexts
        self.negative_sampler = CumulativeNegativeSampler(
            self.negative_sampling_probs,
            seed=seed,
        )
        # 保留这个名称，表示 Dataset 使用的独立随机状态。
        self.rng = self.negative_sampler.rng

    def __len__(self) -> int:
        """返回正样本对的数量，重复出现的正样本也计数。"""
        return len(self.skipgram_pairs)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (中心词, 正上下文词, 负样本) 三个 CPU long 张量。

        前两个张量是标量，形状为 []；负样本形状为 [num_negatives]。
        接受 Python 整数索引，支持负索引；越界抛出 IndexError，
        布尔值、切片或其他类型抛出 TypeError。无效索引不推进随机状态。
        每次读取都会继续使用 self.rng 抽样，结果可能与上次相同。
        固定种子并按相同顺序读取新建的数据集，可以复现整个抽样序列。
        """
        if type(index) is not int:
            raise TypeError("index 必须是 Python 整数，不支持布尔值或切片")
        center_id, context_id = self.skipgram_pairs[index]
        excluded_ids = {center_id} | self.center_to_positive_contexts[center_id]

        negative_ids = self.negative_sampler.sample(
            num_negatives=self.num_negatives,
            excluded_ids=excluded_ids,
        )
        center = torch.tensor(center_id, dtype=torch.long, device="cpu")
        context = torch.tensor(context_id, dtype=torch.long, device="cpu")
        negatives = torch.tensor(negative_ids, dtype=torch.long, device="cpu")
        return center, context, negatives
