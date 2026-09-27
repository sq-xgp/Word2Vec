# Word2Vec 单表与双表等参数量对比报告

## 1. 实验目的

本实验比较两种 Word2Vec embedding 结构：

- `dual + dot + dim=100`：中心词和上下文分别使用一张100维 embedding table；
- `shared + dot + dim=200`：中心词和上下文共享一张200维 embedding table。

设词表大小为 `V`，两种结构的 embedding 参数量相同：

```text
dual d100：   2 × V × 100 = 200V
shared d200： 1 × V × 200 = 200V
```

本实验用于观察参数总量相同时，两张低维表和一张高维共享表的表现差异。

## 2. 实验设置

两组模型使用相同的训练条件：

| 参数 | 设置 |
|---|---:|
| 训练语料 | 100K句英文 Wikipedia |
| 词表大小 | 14,951 |
| 窗口半径 | 5 |
| 负样本数 | 5 |
| batch size | 2,048 |
| 学习率 | 0.01 |
| 最大 epoch | 20 |
| 验证集比例 | 10% |
| patience | 3 |
| min_delta | 0.001 |
| 随机种子 | 42 |
| 训练打分 | dot |

两组模型的 embedding 参数量均为：

```text
2,990,200
```

## 3. 训练结果

| 模型 | 最佳 epoch | 最佳验证 loss | 停止位置 |
|---|---:|---:|---:|
| dual + dot + d100 | 1 | 2.458485 | 第4轮 early stop |
| shared + dot + d200 | 7 | 3.856090 | 第10轮 early stop |

在相同数据和 dot 训练目标下，双表100维模型获得了更低的验证 loss，并且更早达到最佳状态。共享表200维模型需要更多训练轮次才达到最佳结果。

## 4. 相似词对比

推理阶段统一使用 cosine similarity，并分别查询 `music`、`city` 和 `war`。

| 查询词 | dual + d100 | shared + d200 | 观察 |
|---|---|---|---|
| `music` | musical, folk, solo, dancing, dance | musical, film, songs, dance, pop | dual 的音乐主题更集中 |
| `city` | town, park, area, towns, cities | country, home, town, river, county | dual 略集中，shared 更偏地理环境 |
| `war` | military, fought, peace, civil, iraq | military, japanese, british, troops, battle | shared 的战争主题更集中 |

共享表从100维提升到200维后，能够返回更多合理的相关词，说明增加单表维度提升了它的表达能力。但是从三个查询词的整体表现看，双表100维仍然更加稳定。

## 5. 结论

在 embedding 参数总量相同的条件下：

1. `dual + dot + d100` 的验证 loss 明显更低；
2. 双表模型在 `music` 和 `city` 上的相似词更集中；
3. `shared + dot + d200` 在 `war` 上表现更好；
4. 单表增加到200维后有所改善，但没有在整体表现上超过双表100维；
5. 当前实验说明，将中心词和上下文角色分别建模，比单纯增加共享向量维度更适合本次语料和训练配置。

以上结论来自验证 loss 和三个查询词的人工检查，属于阶段性结果。若要得到更可靠的结论，后续应增加更多查询词，或使用 WordSim353、SimLex-999 等公开数据集进行定量评估。
