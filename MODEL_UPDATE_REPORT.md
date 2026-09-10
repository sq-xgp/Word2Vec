# Word2Vec 可配置训练版本修改与实验报告

## 1. 报告概述

本次修改在原有 PyTorch Skip-gram with Negative Sampling（SGNS）项目上，增加了可配置模型结构、验证集、early stopping、余弦训练和多模型联合推理功能。同时重新整理了负采样代码，并通过四组100K句正式实验检查不同配置的训练行为和相似词质量。

开发工作最初在 `feature/configurable-embeddings-validation` 分支完成，经过测试与服务器训练后合并到 `main`。合并后的版本为提交 `5019427`，6项自动测试全部通过。

## 2. 本次修改内容

### 2.1 简化代码和负采样

原实现包含较多重复的边界检查和逐项采样逻辑。本次删除了调用链中重复、无法提供额外保护的判断，将参数检查集中在模块入口处；负采样统一交给累计概率采样器处理。

负样本仍然按词频的0.75次方分布抽取，并排除：

- 当前中心词；
- 该中心词在完整语料中出现过的所有正上下文词。

同一个负样本可以重复出现，这与 SGNS 的独立重复抽样含义一致，也避免了为了强制去重而增加复杂逻辑。

### 2.2 增加验证集和 early stopping

训练数据现在按完整句子划分为训练集和验证集，而不是先生成词对再随机拆分。这样可以保证同一句话产生的上下文窗口不会同时进入两个集合，降低数据泄漏风险。

本次正式实验使用10%验证集。训练集每轮重新采样负样本，验证集的负样本在训练开始时固定，使各轮验证 loss 在相同目标上计算。只有验证 loss 至少改善 `min_delta` 时才保存 checkpoint；连续 `patience` 轮没有改善便提前停止。

### 2.3 支持一张或两张 embedding table

新增 `--embedding-mode` 参数：

- `dual`：中心词和上下文分别使用一张 embedding table；
- `shared`：中心词和上下文共享同一张 embedding table。

`dual` 参数量更多，可以分别学习“作为中心词”和“作为上下文词”时的表示；`shared` 参数量较少，并对两种角色施加共享约束。

新增 `--embedding-dim` 参数，可直接指定词向量维度。本次四组正式实验统一使用100维。

### 2.4 支持 dot 和 cosine 训练打分

新增 `--score-mode` 参数：

- `dot`：使用原始点积作为 SGNS logits；
- `cosine`：使用余弦相似度作为 logits。

余弦相似度只位于 `[-1, 1]`，直接送入二元逻辑损失时动态范围不足。因此 cosine 模式新增 `--temperature`：

```text
logit = cosine_similarity / temperature
```

默认 `temperature=0.1`，使 logits 的范围扩大到约 `[-10, 10]`。dot 模式不使用该参数。

### 2.5 checkpoint 分开保存和多模型联合测试

默认 checkpoint 文件名现在包含 embedding 模式、打分模式、temperature、维度、窗口、负样本数、batch size、学习率、词频阈值、降采样阈值、最大 epoch、验证集比例、patience 和随机种子。不同超参数实验不会互相覆盖。

`inference.py` 支持一次传入多个 checkpoint。输入一个查询词后，各模型分别输出自己的 top-10 相似词，便于在同一推理条件下直接比较。加载器同时兼容旧 checkpoint；旧文件缺少新增字段时，按原来的 `dual + dot` 配置读取。

## 3. 训练中发现的问题与修正

### 3.1 验证集误采样已知正上下文

#### 问题表现

第一版验证实现分别根据训练子集和验证子集建立正上下文集合。某个词在训练句子中出现过的真实上下文，如果没有出现在验证句子中，就可能在验证阶段被抽成负样本。模型在训练时被要求提高这对词的分数，在验证时却被要求降低同一对词的分数，形成标签冲突。其表现是训练 loss 持续下降，但验证 loss 很快上升并过早停止。

#### 原因

句子拆分本身是正确的，但“哪些词对属于语料中的正关系”不应局限于某一个子集。负采样的排除信息必须来自完整语料。

#### 修正方法

在划分训练集和验证集后，先用完整语料建立全局 `center -> positive contexts` 映射，再将同一映射传给两个 Dataset。训练集和验证集都不会把全语料中已知的正上下文抽成负样本。

修正后增加了自动测试，专门验证外部正上下文不会作为负样本出现。10K句冒烟实验的验证 loss 由第1轮 `2.762801` 降到第2轮 `2.740203`，验证行为恢复正常。

### 3.2 未缩放 cosine 训练发生表示坍塌

#### 问题表现

最初直接用 cosine 相似度训练，即 `temperature=1`。模型虽然能够正常完成优化，验证 loss 也没有报错，但查询 `music` 时返回 `if、population、does、passed、known` 等无关词，而且大量相似度都接近 `0.9998`。

这说明词向量方向趋于一致，模型失去了区分词义的能力。仅看 loss 和程序是否正常运行，无法发现这一问题；相似词检查揭示了表示坍塌。

#### 原因

SGNS 使用逻辑损失区分正负样本，而未缩放 cosine logits 只能位于 `[-1, 1]`。这个范围过窄，模型难以为正负样本建立足够大的分数间隔，容易收敛到方向非常相似的退化表示。

#### 修正方法

在 cosine logits 上加入温度缩放，并设置 `temperature=0.1`。同时增加测试，验证温度参数确实按预期缩放 logits；checkpoint 名称和配置中也保存 temperature，避免混淆缩放与未缩放模型。

修正后的相似度恢复到约 `0.5～0.7` 的有区分度范围。`music`、`city` 和 `war` 的近邻均回到对应主题，说明坍塌问题已经解决。未缩放模型保留为失败对照，不纳入最终四模型结果。

### 3.3 最佳 checkpoint 与完整训练历史

当前 checkpoint 只在验证 loss 改善时覆盖，因此最终文件保存的是最佳 epoch 的模型状态和截至该轮的 history。例如 dual + dot 的最佳轮次为第1轮，虽然训练到第4轮才 early stop，但最佳 checkpoint 中的 history 只到第1轮。

这不影响恢复最佳模型和推理，但不能仅靠最佳 checkpoint 重建 early stopping 前的完整曲线。后续如果需要完整审计，可另外保存一个 `last` checkpoint 或独立的 JSON/CSV 训练日志。

## 4. 正式实验设置

四组模型使用相同的数据和基础超参数：

| 项目 | 设置 |
|---|---:|
| 语料 | Leipzig `eng_wikipedia_2016_100K` |
| 训练 / 验证正样本对 | 6,162,840 / 677,470 |
| 词向量维度 | 100 |
| 窗口半径 | 5 |
| 负样本数 | 5 |
| `min_count` | 10 |
| 降采样阈值 | `1e-4` |
| batch size | 2,048 |
| 学习率 | 0.01 |
| 最大 epoch | 20 |
| 验证集比例 | 0.1 |
| patience / min_delta | 3 / 0.001 |
| DataLoader workers | 8 |
| 随机种子 | 42 |
| 训练设备 | CUDA |

## 5. 训练结果

| Embedding 模式 | 训练打分 | Temperature | 最佳 epoch | 最佳验证 loss | 实际停止点 |
|---|---|---:|---:|---:|---:|
| dual | dot | 不适用 | 1 | 2.458485 | 4，early stop |
| shared | dot | 不适用 | 6 | 3.739783 | 9，early stop |
| dual | cosine | 0.1 | 20 | 2.250856 | 20 |
| shared | cosine | 0.1 | 20 | 3.799480 | 20 |

不同模型结构和 logit 计算方式对应不同的优化空间，因此不能把四个验证 loss 当作同一尺度并直接排名。验证 loss 的主要用途是判断同一个模型在不同 epoch 间是否改善，并选择该模型自己的最佳 checkpoint。

两个 dot 模型都触发了 early stopping。两个 temperature=0.1 的 cosine 模型训练到第20轮时验证 loss 仍在改善，其中 dual cosine 后期从第16轮的 `2.261984` 下降到第20轮的 `2.250856`，shared cosine 从第16轮的 `3.805124` 下降到第20轮的 `3.799480`。

## 6. 多模型相似词结果

推理阶段统一使用中心词向量的 cosine 相似度，以排除推理度量不同带来的影响。

### 6.1 `music`

| 模型 | 代表性 top-10 结果 |
|---|---|
| dual + dot | musical, folk, solo, dancing, dance, ragtime, songs, musicians, starred, genres |
| shared + dot | film, musical, series, popular, show, tv, played, works, artists, became |
| dual + cosine, T=0.1 | pop, folk, songs, dance, musical, jazz, guitar, albums, musicians, song |
| shared + cosine, T=0.1 | dance, songs, musical, pop, albums, artists, song, solo, film, featured |

两个 cosine 模型明显更集中于音乐类型、作品和表演相关概念，其中 dual cosine 的结果最完整。

### 6.2 `city`

| 模型 | 代表性 top-10 结果 |
|---|---|
| dual + dot | town, park, office, area, towns, outside, cities, opened, county, visitors |
| shared + dot | national, half, times, southern, town, downtown, street, home, country, area |
| dual + cosine, T=0.1 | town, county, downtown, area, park, district, tourist, located, san, centre |
| shared + cosine, T=0.1 | town, area, park, located, county, district, largest, road, river, san |

dual cosine 的城市与地理语义最集中；shared cosine 的前六项同样稳定，但后半部分稍微分散。

### 6.3 `war`

| 模型 | 代表性 top-10 结果 |
|---|---|
| dual + dot | military, forced, israel, fought, gorbachev, peace, iraq, italy, persia, civil |
| shared + dot | military, british, foreign, army, battle, france, russia, troops, german, revolution |
| dual + cosine, T=0.1 | army, civil, military, campaign, wars, soviet, vietnam, troops, france, allies |
| shared + cosine, T=0.1 | army, military, british, japanese, soviet, troops, civil, u, german, took |

dual cosine 再次给出了最集中的主题近邻。shared dot 在 `war` 上也表现良好；shared cosine 大部分结果合理，但出现了噪声词 `u`。

## 7. 最终结论

本次版本完成了原定的全部功能目标，并通过自动测试、10K句冒烟训练和100K句四模型正式训练验证。

根据 `music`、`city`、`war` 三个查询词的人工结果，当前综合表现可暂定为：

1. `dual + cosine + temperature=0.1`：三个主题上最稳定、语义最集中；
2. `shared + cosine + temperature=0.1`：使用较少参数仍能获得较好结果，但偶尔出现噪声；
3. `dual + dot`：整体合理，但近邻相对分散；
4. `shared + dot`：在战争主题上较强，在音乐和城市主题上较弱。

这个排序只基于三个查询词的定性观察，不能视为普遍的模型质量排名。更严格的下一步是加入 WordSim353、SimLex-999 或词类比数据集，分别报告 Spearman 相关系数或 analogy accuracy。还应保存独立训练日志，以完整保留最佳 epoch 之后直到 early stopping 的训练过程。

从问题定位过程来看，本次实验最关键的经验有两点：验证数据必须保持目标定义一致，不能把全语料中的真实正关系误作负样本；新的训练目标即使 loss 正常，也必须检查表示本身是否退化。自动测试、验证曲线和实际相似词查询需要结合使用，才能判断模型是否真正有效。
