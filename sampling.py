"""采样：高频词 subsampling、负采样分布与负样本生成。"""

import math
import random
from bisect import bisect_right


def build_negative_distribution(
    word_counts: dict[str, int], word_to_id: dict[str, int]
) -> list[float]:
    """按词编号返回负采样概率：P(w) = count(w)^0.75 / sum(count^0.75)。

    使用 build_vocab() 保留的词及其 subsampling 前的出现次数。
    返回列表的下标就是 word_id，不依赖输入字典的插入顺序。
    只统计词表中的词；word_counts 中的额外词不参与归一化。
    词表必须非空，编号必须是从 0 开始的连续整数，每个词的计数
    必须是正整数；不满足时抛出 ValueError。
    本函数只计算全词表的概率，不抽样，也不排除特定中心词或上下文词。
    """
    if not word_to_id:
        raise ValueError("词表不能为空，无法为空词表构建概率分布")

    word_ids = list(word_to_id.values())
    if any(type(word_id) is not int for word_id in word_ids):
        raise ValueError("词编号必须是整数")
    if sorted(word_ids) != list(range(len(word_to_id))):
        raise ValueError("词编号必须从 0 开始连续且不重复")

    weights = [0.0] * len(word_to_id)
    for word, word_id in word_to_id.items():
        count = word_counts.get(word)
        if type(count) is not int or count < 1:
            raise ValueError(f"词 {word!r} 缺少词频，或词频不是正整数")
        weights[word_id] = count ** 0.75

    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    return probabilities


def subsample_frequent_words(
    tokens: list[str],
    word_counts: dict[str, int],
    threshold: float = 1e-5,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    """按论文第 2.3 节的规则对 token 序列降采样，返回保留的词。

    word_counts 使用 build_vocab() 保留的词及其降采样前的计数。
    f(w) = count(w) / sum(word_counts.values())，即保留词表内的相对词频。
    P_keep(w) = min(1, sqrt(threshold / f(w)))，每次出现独立抽样。
    不在 word_counts 中的词视为已被词表过滤，直接跳过。

    保持剩余词的原顺序，不修改输入；空输入或空词表返回空列表。
    threshold 必须是 (0, 1] 内的有限数值；设为 1 可保留所有词表内词。
    seed 用于复现单次调用；逐句处理可传入同一个 rng 来延续随机序列。
    seed 和 rng 不能同时指定；两种方式都不改变 Python 全局随机状态。
    小语料使用默认阈值可能全部被丢弃；此时也返回空列表。
    """
    subsampler = FrequentWordSubsampler(
        word_counts,
        threshold=threshold,
        seed=seed,
        rng=rng,
    )
    return subsampler.sample(tokens)


class FrequentWordSubsampler:
    """预先计算每个词的保留概率，并在多次逐句调用间延续随机状态。"""

    def __init__(
        self,
        word_counts: dict[str, int],
        threshold: float = 1e-5,
        seed: int | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError("threshold 必须是 (0, 1] 内的有限数值")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed 必须是整数或 None")
        if rng is not None and not isinstance(rng, random.Random):
            raise ValueError("rng 必须是 random.Random 实例或 None")
        if seed is not None and rng is not None:
            raise ValueError("seed 和 rng 不能同时指定")
        for word, count in word_counts.items():
            if type(count) is not int or count < 1:
                raise ValueError(f"词 {word!r} 的词频必须是正整数")

        self.word_counts = dict(word_counts)
        self.threshold = threshold
        self.rng = rng if rng is not None else random.Random(seed)
        self.keep_probabilities = {}
        if self.word_counts:
            total_count = sum(self.word_counts.values())
            for word, count in self.word_counts.items():
                frequency = count / total_count
                self.keep_probabilities[word] = min(
                    1.0,
                    math.sqrt(threshold / frequency),
                )

    def sample(self, tokens: list[str]) -> list[str]:
        """按预先计算的概率逐个判断 token，并保持原顺序。"""
        kept_tokens = []
        for token in tokens:
            if token not in self.keep_probabilities:
                continue
            if self.rng.random() < self.keep_probabilities[token]:
                kept_tokens.append(token)
        return kept_tokens


def sample_negatives(
    negative_sampling_probs: list[float],
    num_negatives: int = 5,
    excluded_ids: set[int] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """按给定概率有放回地抽取负样本编号，跳过 excluded_ids 中的词。

    概率列表下标对应 word_id，元素须有限且非负，总和约为 1。
    num_negatives 必须是正整数；同一个词可以被重复抽中。
    排除后按剩余概率的相对比例抽样，不会把概率为 0 的词作为候选。
    无可用候选词时抛出 ValueError，不通过反复重抽等待结果。

    调用者负责传入排除集合（例如中心词及其全部已知正上下文词）。
    默认不排除任何词；本函数不推断正样本关系，也不修改输入。
    seed 用于复现单次调用；连续抽样可复用 rng，使随机序列继续向前。
    seed 和 rng 不能同时指定；rng 会推进自身状态，不影响全局随机状态。
    此学习版每次扫描词表，适合小规模数据；大语料阶段再优化。
    """
    if type(num_negatives) is not int or num_negatives < 1:
        raise ValueError("num_negatives 必须是正整数")
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
    if not math.isclose(sum(negative_sampling_probs), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("负采样概率总和必须约等于 1")

    excluded = set() if excluded_ids is None else set(excluded_ids)
    for word_id in excluded:
        if type(word_id) is not int or not 0 <= word_id < len(negative_sampling_probs):
            raise ValueError("排除集合中的编号必须是有效的词编号")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed 必须是整数或 None")
    if rng is not None and not isinstance(rng, random.Random):
        raise ValueError("rng 必须是 random.Random 实例或 None")
    if seed is not None and rng is not None:
        raise ValueError("seed 和 rng 不能同时指定")

    candidate_ids = []
    candidate_weights = []
    for word_id, probability in enumerate(negative_sampling_probs):
        if word_id not in excluded and probability > 0:
            candidate_ids.append(word_id)
            candidate_weights.append(probability)
    if not candidate_ids:
        raise ValueError("排除后没有概率大于 0 的候选词，无法生成负样本")

    sampler = rng if rng is not None else random.Random(seed)
    return sampler.choices(candidate_ids, weights=candidate_weights, k=num_negatives)


class CumulativeNegativeSampler:
    """预先构建累计概率，之后用二分查找和拒绝采样生成负样本。

    概率检查和累计概率构建只在初始化时执行一次。sample() 抽到排除编号时
    会重新抽取；如果合法概率太小或连续拒绝次数过多，则扫描一次词表完成
    剩余抽样，保证极端情况下也能结束。对象内部保存独立 RNG，连续调用会
    推进随机状态；相同种子和调用顺序可以复现结果。
    """

    def __init__(
        self,
        negative_sampling_probs: list[float],
        seed: int | None = None,
    ) -> None:
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
        if seed is not None and type(seed) is not int:
            raise ValueError("seed 必须是整数或 None")

        self.negative_sampling_probs = tuple(negative_sampling_probs)
        cumulative_probs = []
        running_total = 0.0
        for probability in self.negative_sampling_probs:
            running_total += probability
            cumulative_probs.append(running_total)

        self.cumulative_probs = tuple(cumulative_probs)
        self.total_probability = running_total
        self.positive_probability_ids = frozenset(
            word_id
            for word_id, probability in enumerate(self.negative_sampling_probs)
            if probability > 0
        )
        self.rng = random.Random(seed)

    def sample(
        self,
        num_negatives: int = 5,
        excluded_ids: set[int] | None = None,
    ) -> list[int]:
        """按累计概率有放回抽样，返回不在 excluded_ids 中的编号。"""
        if type(num_negatives) is not int or num_negatives < 1:
            raise ValueError("num_negatives 必须是正整数")

        excluded = set() if excluded_ids is None else set(excluded_ids)
        for word_id in excluded:
            if type(word_id) is not int or not 0 <= word_id < len(
                self.negative_sampling_probs
            ):
                raise ValueError("排除集合中的编号必须是有效的词编号")
        if self.positive_probability_ids.issubset(excluded):
            raise ValueError("排除后没有概率大于 0 的候选词，无法生成负样本")

        excluded_probability = sum(
            self.negative_sampling_probs[word_id] for word_id in excluded
        )
        acceptance_probability = (
            self.total_probability - excluded_probability
        ) / self.total_probability

        # 合法概率过小时直接扫描一次，避免大量拒绝；正常情况走累计概率查找。
        if acceptance_probability < 0.1:
            return self._sample_by_filtering(num_negatives, excluded)

        negative_ids = []
        max_attempts = max(100, num_negatives * 20)
        attempts = 0
        while len(negative_ids) < num_negatives and attempts < max_attempts:
            random_value = self.rng.random() * self.total_probability
            word_id = bisect_right(self.cumulative_probs, random_value)
            if (
                word_id not in excluded
                and self.negative_sampling_probs[word_id] > 0
            ):
                negative_ids.append(word_id)
            attempts += 1

        remaining = num_negatives - len(negative_ids)
        if remaining > 0:
            negative_ids.extend(self._sample_by_filtering(remaining, excluded))
        return negative_ids

    def _sample_by_filtering(
        self,
        num_negatives: int,
        excluded_ids: set[int],
    ) -> list[int]:
        """扫描词表构建合法候选，用于拒绝采样不合适的极端情况。"""
        candidate_ids = []
        candidate_weights = []
        for word_id, probability in enumerate(self.negative_sampling_probs):
            if word_id not in excluded_ids and probability > 0:
                candidate_ids.append(word_id)
                candidate_weights.append(probability)
        return self.rng.choices(
            candidate_ids,
            weights=candidate_weights,
            k=num_negatives,
        )
