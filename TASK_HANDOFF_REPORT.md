# Word2Vec 项目任务交接报告

## 1. 项目概况

本项目使用 PyTorch 从头实现 Skip-gram with Negative Sampling（SGNS），覆盖文本预处理、词表构建、高频词降采样、正负样本生成、模型训练、验证集、early stopping、checkpoint 保存、多模型推理及 PCA/t-SNE 可视化。

- GitHub：`https://github.com/sq-xgp/Word2Vec`
- 本地路径：`C:\Users\86137\Documents\Codex\2026-08-31\referenced-chatgpt-conversation-this-is-an\outputs\word2vec`
- 新服务器路径：`/root/autodl-tmp/Word2Vec`
- 正式分支：`main`
- 原功能分支：`feature/configurable-embeddings-validation`，已合并到 `main`

主要文件：

| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口、验证、early stopping、模型保存 |
| `model.py` | SGNS 模型、embedding 结构和打分方式 |
| `dataset.py` | 按句子动态生成正样本和负样本 |
| `sampling.py` | 高频词降采样和负采样 |
| `preprocess.py` | 文本清洗、词表和 token/ID 转换 |
| `prepare_data.py` | 串联数据预处理流程 |
| `inference.py` | 单模型或多模型相似词查询 |
| `visualize.py` | PCA 和 t-SNE 可视化 |
| `test_pipeline.py` | 新功能自动测试 |
| `MODEL_UPDATE_REPORT.md` | 已完成版本的修改与实验报告 |

## 2. 已完成的功能修改

1. 精简重复的边界判断和负采样代码。
2. 按完整句子划分训练集和验证集。
3. 增加最佳 checkpoint 保存与 early stopping。
4. 支持 `dual` 和 `shared` 两种 embedding table 结构。
5. 支持通过 `--embedding-dim` 指定词向量维度。
6. 支持 `dot` 和 `cosine` 两种训练打分方式。
7. cosine 模式支持 `--temperature`，默认值为 `0.1`。
8. checkpoint 文件名包含主要超参数，避免不同实验互相覆盖。
9. `inference.py` 支持一次加载多个 checkpoint 并联合输出 top-k。
10. 保持对旧 checkpoint 的兼容加载。

当前自动测试共6项，已经在本地、旧服务器和新服务器运行通过。

## 3. 训练数据流与调用顺序

训练数据流：

```text
原始句子文件
→ 读取句子
→ clean_text()
→ build_vocab()
→ 计算负采样概率
→ 高频词降采样
→ tokens_to_ids()
→ 按句子划分训练集/验证集
→ 动态生成 Skip-gram 正样本
→ 生成负样本
→ DataLoader 组成 batch
→ embedding 查表
→ dot/cosine 打分
→ SGNS loss
→ 反向传播与参数更新
→ 验证
→ 保存最佳 checkpoint / early stopping
```

核心调用顺序：

```text
train.py main()
→ run_training()
→ prepare_training_data()
→ split_sentences()
→ build_center_to_positive_contexts()
→ create_dataloader()
→ create_model_and_optimizer()
→ train_model()
→ run_epoch()
→ SentenceWord2VecDataset.__iter__()
→ SkipGramNegSampling.forward()
→ _score()
→ compute_loss()
→ backward()
→ optimizer.step()
→ save_checkpoint()
```

推理调用顺序：

```text
inference.py main()
→ compare_checkpoints()
→ query_checkpoint()
→ load_trained_model()
→ find_similar_words()
→ 计算所有词的相似度
→ torch.topk()
→ 输出相似词
```

## 4. 模型配置说明

### 4.1 Embedding table

- `--embedding-mode dual`：中心词和上下文各使用一张 embedding table。
- `--embedding-mode shared`：中心词和上下文共享同一张 embedding table。

### 4.2 训练打分

- `--score-mode dot`：使用向量点积。
- `--score-mode cosine`：使用余弦相似度除以 temperature。

cosine 训练公式：

```text
logit = cosine_similarity / temperature
```

推理参数 `--similarity` 只控制相似词排序方式，与训练参数 `--score-mode` 相互独立。

## 5. 训练中发现的问题与修正

### 5.1 验证负样本冲突

初版代码分别使用训练子集和验证子集的正上下文关系。训练集中出现过的真实正上下文可能在验证集中被抽成负样本，造成标签冲突，表现为训练 loss 下降而验证 loss 快速上升。

修正方案：使用完整语料建立全局 `center → positive contexts` 映射，训练集和验证集共同使用该映射，保证全语料中的已知正上下文不会被当作负样本。同时固定验证负样本，确保不同 epoch 的验证结果可比较。

### 5.2 Cosine 表示坍塌

未缩放的 cosine 模型（`temperature=1`）训练后，大量单词之间的余弦相似度接近 `0.9998`，并返回许多与查询词无关的结果，说明词向量方向趋于一致。

修正方案：使用 `temperature=0.1` 放大 cosine logits 的范围。修正后，相似度恢复到约 `0.5～0.7`，`music`、`city`、`war` 的相似词重新集中到对应主题。未缩放模型仅作为失败对照，不纳入正式结果。

## 6. 已完成的四模型实验

统一训练条件：100K句英文 Wikipedia、100维、窗口半径5、5个负样本、batch size 2048、学习率0.01、10%验证集、patience 3、min_delta 0.001、seed 42、8个 DataLoader workers。

| 模型 | 最佳 epoch | 最佳验证 loss | 停止情况 |
|---|---:|---:|---|
| dual + dot | 1 | 2.458485 | 第4轮 early stop |
| shared + dot | 6 | 3.739783 | 第9轮 early stop |
| dual + cosine，T=0.1 | 20 | 2.250856 | 完成20轮 |
| shared + cosine，T=0.1 | 20 | 3.799480 | 完成20轮 |

不同模型的 loss 尺度不完全一致，不能仅依据数值大小横向排名。

人工查询 `music`、`city`、`war` 后的阶段性排序：

1. `dual + cosine + temperature=0.1`
2. `shared + cosine + temperature=0.1`
3. `dual + dot`
4. `shared + dot`

该排序只基于少量查询词的定性观察，尚未使用标准词相似度数据集进行定量评估。

## 7. PCA 学习要点

PCA 只用于训练后的可视化，不参与 Word2Vec 参数更新。项目中的处理过程为：

```text
100维词向量
→ L2归一化
→ PCA中心化
→ 寻找方差最大的两个主成分
→ 投影为二维坐标
→ 绘制散点图
```

PCA 二维图会丢失部分高维信息，因此精确的单词相似度仍应在原始 embedding 空间中计算。

## 8. 当前新实验目标

下一项实验比较：

```text
dual + dot + dim=100
        对比
shared + dot + dim=200
```

设词表大小为 `V`，二者 embedding 参数总量相同：

```text
dual d100：   2 × V × 100 = 200V
shared d200： 1 × V × 200 = 200V
```

实验目的：在 embedding 参数总量相同的条件下，比较“两张100维表”和“一张200维共享表”的效果。

除 embedding 模式和维度外，应保持语料、窗口、负样本数、batch size、学习率、验证集、early stopping、随机种子和 `score-mode=dot` 一致。

计划从以下方面比较：

1. 最佳验证 loss 和最佳 epoch；
2. `music`、`city`、`war` 的 top-10；
3. 相同 embedding 参数量下的语义质量；
4. 如条件允许，比较训练时间和 checkpoint 大小。

## 9. 新服务器当前状态

新服务器代码路径：

```text
/root/autodl-tmp/Word2Vec
```

代码已经从 GitHub `main` 克隆成功，6项自动测试全部通过。当前缺少100K句训练语料，也没有旧服务器生成的 `.pt` checkpoint，因为这些文件没有提交到 Git。

语料文件目标位置：

```text
data/raw/eng_wikipedia_2016_100K-sentences.txt
```

下载地址：

```text
https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_100K.tar.gz
```

历史校验信息：

```text
大小：25,566,797 bytes
SHA-256：04aa301072a612e0368f1a0abe5f6b011ab03df84961c29b80bd12683a5a6f0
```

## 10. 下一步任务

1. 在新服务器下载并校验100K句语料。
2. 解压 `eng_wikipedia_2016_100K-sentences.txt`。
3. 检查 GPU 和 CUDA 环境。
4. 训练 `shared + dot + dim=200`。
5. 为保证受控对比，从旧服务器复制 `dual + dot + dim=100` checkpoint，或在新服务器使用相同设置重新训练。
6. 提取两个模型的最佳 epoch 和验证 loss。
7. 联合查询 `music`、`city`、`war`。
8. 汇总训练时间、checkpoint 大小和语义结果。
9. 将新实验写入项目报告并提交 Git。

## 11. 新实验训练命令

```bash
cd /root/autodl-tmp/Word2Vec

python train.py \
  --sentence-file data/raw/eng_wikipedia_2016_100K-sentences.txt \
  --min-count 10 \
  --subsampling-threshold 0.0001 \
  --window-size 5 \
  --embedding-dim 200 \
  --embedding-mode shared \
  --score-mode dot \
  --num-negatives 5 \
  --batch-size 2048 \
  --epochs 20 \
  --learning-rate 0.01 \
  --validation-fraction 0.1 \
  --patience 3 \
  --min-delta 0.001 \
  --seed 42 \
  --num-workers 8 \
  --progress-every-batches 500
```

预期 checkpoint：

```text
checkpoints/w2v_shared_dot_d200_w5_n5_bs2048_lr0.01_mc10_ss1em04_e20_vf0.1_p3_seed42.pt
```
