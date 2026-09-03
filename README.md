# Word2Vec 学习项目

使用 PyTorch 分步实现带负采样的 Skip-gram（SGNS）。

正式100K句实验的配置、训练曲线、相似词和语义分组图见 [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)。

## 当前进度

已完成第 1 步：创建项目目录结构和基础文件。

已完成第 2 步的第一小步：在 `preprocess.py` 中实现 `clean_text()`。它把英文文本转为小写，并提取连续的英文字母和数字作为 token；标点及其他字符作为分隔符。此学习版暂不处理中文分词，后续再根据真实语料调整规则。

例如 `Hello, WORLD! I like word2vec.` 会得到 `['hello', 'world', 'i', 'like', 'word2vec']`。

已完成第 2 步的第二小步：添加 `build_vocab(tokens, min_count=1)`，统计词频并过滤低频词，返回 `word_to_id`（单词 → 编号）、`id_to_word`（编号 → 单词）和 `word_counts`（保留词的词频）。编号从 0 开始，按词频降序分配；同频词按首次出现顺序排列。空输入或全部被过滤时，返回三个空字典。

例如 `['cat', 'dog', 'cat', 'bird', 'dog', 'cat']` 在 `min_count=2` 时，会得到 `{'cat': 0, 'dog': 1}`、`{0: 'cat', 1: 'dog'}` 和 `{'cat': 3, 'dog': 2}`。`bird` 只出现一次，因此不进入词表。此函数不修改原始 token 列表。

已完成第 2 步的第三小步：添加 `tokens_to_ids(tokens, word_to_id)`，按原顺序把每个词转换为编号，保留重复出现的词，并跳过词表以外的词。不修改输入列表或词表，也不新增未知词编号；空输入或没有词命中词表时，返回空列表。

沿用上述例子，`tokens_to_ids()` 得到 `[0, 1, 0, 1, 0]`。`bird` 被跳过，而编号为 0 的 `cat` 正常保留。正样本构造基于过滤后的编号序列。

已完成第 2 步的第四小步：添加 `generate_skipgram_pairs(token_ids, window_size=2)`，返回 `(中心词编号, 上下文词编号)` 列表。`window_size` 表示中心词左右各自最多查看多少个位置，必须是正整数。采用固定窗口，序列两端自动截断，不会越界。

例如 `[0, 1, 2]` 在 `window_size=1` 时，得到 `[(0, 1), (1, 0), (1, 2), (2, 1)]`。只排除中心位置本身，不排除其他位置上的同一词：`[0, 0]` 会生成两个 `(0, 0)`。重复样本保留，用于反映实际共现次数。空序列或单词序列只有一个词时，返回空列表。

当前正样本函数面向小规模学习数据，一次返回全部样本；不自动识别句子边界。如果要避免跨句配对，需要逐句传入编号序列。接入大语料前，再处理句子边界和内存占用。这一小步没有实现采样、负样本或训练。

已完成第 3 步的第一小步：在 `sampling.py` 中添加 `build_negative_distribution(word_counts, word_to_id)`。采用词频的 0.75 次方作为权重，再除以权重总和得到概率，即 `P(w) = count(w)^0.75 / sum(count^0.75)`。这一分布来自 [Word2Vec 论文第 2.2 节](https://arxiv.org/pdf/1310.4546)。

函数使用 `build_vocab()` 保留的词及其 subsampling 前的计数，返回 Python 浮点数列表。列表下标严格对应 `word_id`，不依赖字典的插入顺序。例如 `word_counts={'cat': 3, 'dog': 2}`、`word_to_id={'cat': 0, 'dog': 1}`，得到约 `[0.5754, 0.4246]`，概率总和约等于 1。

空词表、缺失或非正整数词频、重复或不连续的编号会报错。词表以外的计数不参与计算。此函数只计算全词表的基础分布，不抽取负样本，也不排除某个训练样本的中心词或正上下文词；这些规则后续在负样本生成时处理。目前无需安装第三方依赖。

已完成第 3 步的第二小步：添加 `subsample_frequent_words(tokens, word_counts, threshold=1e-5, seed=None, *, rng=None)`，返回降采样后的单词列表。使用 [Word2Vec 论文第 2.3 节](https://arxiv.org/pdf/1310.4546) 的规则，把保留概率截断到 1：`P_keep(w) = min(1, sqrt(threshold / f(w)))`。本项目将 `f(w)` 定义为保留词表中该词的相对频率，分母为 `word_counts` 中全部计数之和；词频使用降采样前的统计。

高于阈值的词按概率丢弃部分出现次数；低于或等于阈值的词全部保留。每次出现独立抽样，不会把某个高频词从词表中删除。结果保持原始顺序，不修改输入列表或词频字典；词表外的词直接跳过。固定 `seed` 可复现结果，且不影响全局随机状态。

`threshold` 必须是 `(0, 1]` 内的有限数值，设为 1 可保留全部词表内词。空输入或空词表返回空列表。默认阈值较小，适合在大语料上尝试；学习时的小例子可能全部被丢弃，可临时调大阈值观察行为，但不能把演示值直接当作正式训练设置。该函数也可能返回不足两个词，后续正样本列表就会为空。

当前各函数的衔接顺序如下（这里只说明调用顺序，尚未创建训练入口）：

```text
clean_text → build_vocab
                  ├── 原始 word_counts + word_to_id → build_negative_distribution
                  └── tokens + 原始 word_counts → subsample_frequent_words
                                                   → tokens_to_ids
                                                   → generate_skipgram_pairs
```

降采样后不要重建词表或重新统计负采样分布。跨句处理和分批调用时的随机数管理将在数据管线阶段完善；不要在每一句上重复使用同一个种子重启随机数序列。

已完成第 3 步的第三小步：添加 `sample_negatives(negative_sampling_probs, num_negatives=5, excluded_ids=None, seed=None, *, rng=None)`，按概率抽取指定数量的负样本编号。这是有放回抽样，同一个词可以重复出现，因此负样本数量可以大于候选词数量。

`excluded_ids` 指定不能被抽中的词编号；函数默认不排除任何词，也不会自行识别正上下文。Dataset 按项目约定传入中心词和该中心词全部已知正上下文词的编号。排除后，候选词保持原概率的相对比例。例如原概率 `[0.1, 0.2, 0.3, 0.4]` 排除 `{0, 1}` 后，编号 2 和 3 被抽中的概率分别是 `3/7` 和 `4/7`。概率为 0 的词永远不会被抽中。

函数先筛选候选词，再一次抽取 `num_negatives` 个结果，不使用可能无法结束的反复重抽。若候选词全部被排除或仅剩零概率词，会明确报错；小语料下需要检查词表与正上下文集合，不能悄悄放回被排除的词。概率列表必须非空、数值有限且非负、总和约为 1；抽样数量必须为正整数，排除编号必须有效。输入概率和排除集合保持不变。

`seed` 用于复现单次调用。连续抽样时应只创建一次 `rng = random.Random(42)`，后续调用传入同一个 `rng`，避免每个样本都从相同随机起点重新开始。两者不能同时传入。复用 `rng` 会推进其自身状态，不修改 Python 全局随机状态。当前实现每次会扫描词表，面向小规模学习数据；大语料训练前再优化抽样效率。

已完成第 4 步的第一小步：在 `dataset.py` 中添加 `build_center_to_positive_contexts(skipgram_pairs)`，将全部已生成的正样本整理为“中心词编号 → 正上下文编号集合”。例如 `[(0, 1), (0, 2), (0, 1), (1, 0)]` 得到 `{0: {1, 2}, 1: {0}}`。同一个上下文编号在集合中只记录一次，但原始正样本列表不变，重复训练样本仍然保留。

映射只记录传入样本中的有向关系，不自动补反向关系；这里的“已知正上下文”仅指输入正样本中出现过的关系。空列表返回空字典；格式不正确或编号不是非负整数时会报错。由于本函数没有接收词表，编号上界将在后续 Dataset 中检查。

负采样时，中心词 `center_id` 的排除集合可以写成 `excluded_ids = {center_id} | center_to_positive_contexts[center_id]`，其中 `|` 创建两个集合的并集，不会修改已有映射。例如中心词 0 的正上下文为 `{1, 2}`，则排除集合为 `{0, 1, 2}`。需要先用全部正样本建立映射，再按单个样本抽取负样本。

已完成第 4 步的第二小步：确认现有 PyTorch 可用，并添加继承自 `torch.utils.data.Dataset` 的 `Word2VecDataset`，在这一小步中实现 `__init__()` 和 `__len__()`。初始化复制正样本与概率列表，自动建立正上下文映射，保存负样本数量，并创建一个独立的随机数生成器供读取样本时连续抽样使用；初始化时不会抽取负样本。

初始化会检查概率是否合法、正样本编号是否越界，以及每个中心词排除自身和全部已知正上下文后是否还有正概率候选词。若候选词耗尽，立即报错，避免迭代时才失败。合法概率列表配合空正样本列表可以创建长度为 0 的数据集。`len(dataset)` 返回原始正样本对数量，重复样本不会去重；修改外部输入列表也不会改变数据集保存的副本。

例如正样本 `[(0, 1), (0, 2), (0, 1), (1, 0)]` 配合概率 `[0.1, 0.2, 0.3, 0.4]` 创建数据集后，`len(dataset)` 为 4。编号 3 可作为中心词 0 的负样本候选，所以初始化通过。若概率只覆盖编号 0、1、2，则中心词 0 会排除全部候选词，初始化报错。

已完成第 4 步的第三小步：实现 `__getitem__()`。`center, context, negatives = dataset[index]` 先读取对应的正样本，排除中心词和全部已知正上下文，再调用 `sample_negatives(..., rng=self.rng)` 抽取负样本。返回三个 CPU 上的 `torch.long` 张量：中心词和正上下文词的形状都是 `[]`（标量），负样本形状为 `[num_negatives]`。它们仍然是词编号，不是词向量。

沿用上述 4 条正样本与 4 个词的概率列表，设置 `num_negatives=3` 后，`dataset[0]` 返回 `(tensor(0), tensor(1), tensor([3, 3, 3]))`。因为中心词 0 已知正上下文为 `{1, 2}`，排除 `{0, 1, 2}` 后只剩编号 3；允许重复抽取，所以能返回 3 个负样本。

支持普通 Python 整数索引及负索引，例如 `dataset[-1]` 读取最后一条正样本；越界抛出 `IndexError`，布尔值、切片或其他类型抛出 `TypeError`。无效索引不会推进随机状态。每次有效读取都会继续抽样，重复读取同一个索引的负样本可能相同，也可能不同；重新创建同种子的数据集并保持访问顺序相同，可复现抽样序列。

已完成第 4 步的第四小步：新增 `demo_dataloader.py`，演示用 DataLoader 把单条样本堆叠成 batch。使用 5 条正样本、`batch_size=2`、`num_negatives=3`、`shuffle=False`、`num_workers=0` 和 `drop_last=False`。因此 `len(dataset)` 为 5，`len(dataloader)` 为 3，三批实际样本数分别为 2、2、1。

每个 batch 返回 `centers`、`contexts` 和 `negatives`，形状分别为 `[B]`、`[B]`、`[B, K]`，其中 `B` 是这批实际样本数，`K` 是每条正样本的负样本数。前两批形状为 `[2]`、`[2]`、`[2, 3]`；最后一批为 `[1]`、`[1]`、`[1, 3]`。负样本矩阵的第 i 行对应第 i 个中心词，所有张量仍是 CPU 上的 `torch.long` 词编号。

演示不打乱正样本顺序，方便对照；负样本仍会按概率抽取。无需编写自定义 `collate_fn`，DataLoader 默认的组批逻辑就能完成这些张量的堆叠。`drop_last=False` 保留最后不足一批的样本，不通过填充补足数量。重新运行脚本会创建新的同种子 Dataset，复现演示结果。

已验证单进程批量读取。多进程 worker 的随机状态隔离尚未处理，不应直接改用多个 worker。

已完成第 5 步的第一小步：在 `model.py` 中添加继承自 `torch.nn.Module` 的 `SkipGramNegSampling(vocab_size, embedding_dim=128)`，这一小步实现初始化。`vocab_size` 和 `embedding_dim` 必须是正整数。模型保存两张独立、可训练的 `nn.Embedding` 表，每张形状均为 `[vocab_size, embedding_dim]`：`center_embeddings` 用于中心词，`context_embeddings` 同时用于正上下文词和负样本，不需要第三张表。

词编号直接对应行号，包括编号 0；这里没有设置 padding 词。若词表大小为 4、词向量维数为 128，两张表的形状分别都是 `[4, 128]`，总共包含 `2 * 4 * 128 = 1024` 个可训练参数。两张表没有共享参数，同一个词作为中心词和上下文词时会使用不同的表。

本项目选择将中心词表初始化为 `[-0.5/embedding_dim, 0.5/embedding_dim]` 范围内的小随机数，将上下文词表初始化为零。此时词向量尚未学到语义；上下文词表中的零值是初始化结果，不表示这张表被冻结。模型使用 PyTorch 的随机数生成器；需要复现初始化时，在创建模型前调用 `torch.manual_seed(42)`，这与 Dataset 中 Python 的 `random.Random` 是分别控制的随机状态。

已将 DataLoader 产生的编号张量直接送入两张 embedding 表验证查表：`model.center_embeddings(centers)` 的形状为 `[B, D]`，`model.context_embeddings(contexts)` 为 `[B, D]`，`model.context_embeddings(negatives)` 为 `[B, K, D]`。其中 `B` 是实际 batch 大小，`K` 是每条样本的负样本数量，`D` 是词向量维数。例子中前两批分别得到 `[2, 128]`、`[2, 128]`、`[2, 3, 128]`；最后一批保留 `B=1`。

已完成第 5 步的第二小步：实现 `forward(centers, contexts, negatives)`。可用 `positive_scores, negative_scores = model(centers, contexts, negatives)` 调用，输入三个 long 编号张量的形状分别为 `[B]`、`[B]`、`[B, K]`，返回正样本分数 `[B]` 和负样本分数 `[B, K]`。每条正样本只与自己的上下文和负样本计算点积，不会与其他 batch 行交叉配对。

正样本打分为 `(center_vectors * context_vectors).sum(dim=-1)`：对应维度相乘，再沿词向量维度求和。负样本打分先用 `unsqueeze(1)` 将中心词向量从 `[B, D]` 变为 `[B, 1, D]`，与 `[B, K, D]` 的负样本向量相乘后，沿最后一维求和，得到 `[B, K]`。`B=1` 或 `K=1` 时仍保留这些维度。

这些分数是原始点积，可以为正或负，不是概率。`forward()` 不做 sigmoid，也不对负样本分数取负号；负号在 `compute_loss()` 中处理。上下文表刚初始化为零时，两组分数都为零是预期结果，并不代表打分代码未执行。验证时也使用了人工指定的非零词向量核对数值。

`forward()` 检查输入类型、batch 形状、编号范围和设备一致性。空 batch、零个负样本、标量输入或 batch 大小不一致都会报错；单条样本要先组成 `B=1` 的 batch 再调用。此函数不会重新抽样或更新参数，保留了供后续反向传播使用的计算图。

已完成第 5 步的第三小步：添加 `model.compute_loss(positive_scores, negative_scores)`。先调用模型获得原始分数，再计算损失：

```python
positive_scores, negative_scores = model(centers, contexts, negatives)
loss = model.compute_loss(positive_scores, negative_scores)
```

采用 [Word2Vec 论文第 2.2 节](https://arxiv.org/pdf/1310.4546) 的负采样目标，取负号后对 batch 求平均。每条样本的损失为 `-log(sigmoid(positive_score)) - sum(log(sigmoid(-negative_scores)))`。实现使用数值更稳定的 `F.logsigmoid()`，不直接计算 `log(sigmoid(x))`。K 个负样本的损失求和，不对 K 取平均；之后才对 B 条样本取平均，结果是形状为 `[]` 的标量张量。

正样本分数越高、负样本分数越低，损失越小。比如全部分数为 0、每条样本有 3 个负样本时，损失为 `4 * log(2)`，约等于 `2.772589`，与 batch 大小无关。这个初始数值是公式的结果，不代表模型已经训练过，也不能单凭它衡量词向量质量。

两组分数必须是同设备、同类型的有限 float32 或 float64 张量，形状为 `[B]`、`[B, K]`。空 batch、零个负样本、错误形状或非有限值会报错。当前没有实现 float16/bfloat16 混合精度支持，不要预先给分数做 sigmoid 或取负号，也不要在传入损失前调用 `.detach()` 或 `.item()`。

损失保留了反向传播所需的计算图。已核对手算数值、零分数基准、极端分数的有限结果，以及正负分数的梯度方向，并验证模型两张表与损失的梯度连接。上下文表初始为零时，首次反向传播的中心词表梯度为零是预期情况；上下文表仍可获得非零梯度。此步只做梯度检查，没有优化器更新或训练循环。

已完成第 6 步的第一小步：在 `train.py` 中添加 `train_one_batch(model, optimizer, centers, contexts, negatives)`。它先用 `model.train()` 切换到训练模式，再清空已有梯度、计算分数与损失、调用 `loss.backward()` 计算梯度，最后用 `optimizer.step()` 更新一次参数。训练模式本身不更新参数；本函数没有 epoch 循环，不读取语料或保存模型。

优化器在函数外用当前模型的全部参数创建，后续 batch 复用同一个优化器，不在函数内反复创建。输入须为与模型同设备的编号张量，形状仍为 `[B]`、`[B]`、`[B, K]`。函数返回 `loss.item()`，即普通 Python 浮点数，便于打印和记录。该值来自参数更新前的前向计算；虽然在 `step()` 之后返回，但没有重新计算更新后的损失。

以下是调用方式，假设已有前面创建的 `dataset`；这段代码没有添加为自动执行入口：

```python
import torch
from torch.utils.data import DataLoader
from model import SkipGramNegSampling
from train import train_one_batch

torch.manual_seed(42)
model = SkipGramNegSampling(vocab_size=len(dataset.negative_sampling_probs))
optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
centers, contexts, negatives = next(iter(dataloader))  # 只取第一个 batch。
loss_value = train_one_batch(model, optimizer, centers, contexts, negatives)
print(loss_value)
```

这里用简单的 SGD 演示参数更新，`lr=0.025` 是学习示例中的学习率，并非对所有语料都合适。Dataset 不能为空。每次调用会真实修改内存中的模型参数，但不会自动写入磁盘。初始上下文表为零时，第一次更新中中心词表保持不变、上下文表可以改变；随后中心词表也可以获得非零梯度并更新。

已完成第 6 步的第二小步：添加 `train_one_epoch(model, optimizer, dataloader)`，遍历 DataLoader 一次，每个 batch 调用一次 `train_one_batch()`。整个过程复用同一个模型与优化器，不重新初始化词向量或随机种子，不缓存全部 batch。正常按顺序或打乱遍历数据、并设置 `drop_last=False` 时，一个 epoch 会将所有正样本训练一遍；重复出现的正样本仍参与训练。

返回值是按实际训练样本数加权的平均损失：先累加 `loss_value * batch_size`，再除以实际处理的样本总数。不能直接平均各批损失，因为最后一批可能较小。例如三批样本数为 2、2、1，平均损失分别为 1、3、9，则这一轮平均损失为 `(1*2 + 3*2 + 9*1) / 5 = 3.4`，而不是 `(1+3+9)/3`。这些数值只用于解释统计方式。

沿用已经创建的模型、优化器和数据集，可这样训练一轮；若此前调用过 `train_one_batch()`，这里会在已有参数基础上继续训练：

```python
from train import train_one_epoch

dataloader = DataLoader(
    dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False
)
epoch_loss = train_one_epoch(model, optimizer, dataloader)
print(epoch_loss)
```

当前要求 `num_workers=0`，避免尚未处理的多进程负采样随机状态问题。`drop_last=False` 保留最后不足一批的样本；如果调用者设置 `drop_last=True`，函数只统计 DataLoader 实际提供的样本，不会补回被丢弃的尾批。空数据集或因丢弃尾批而完全没有 batch 时会报错，不执行参数更新。输入仍须与模型位于同一设备，没有添加自动 GPU 搬运。

这一轮平均损失记录的是训练过程中各批更新前的损失；各批对应的模型参数会逐步变化，因此它不是用最终模型重新评估整个数据集得到的损失。本步骤没有添加多 epoch 循环、自动执行的训练入口或 checkpoint 保存。

已完成第 6 步的第三小步：新增可直接运行的 `demo_train.py`。沿用 DataLoader 演示中的 5 条正样本、4 个词编号和示例负采样概率，创建 Dataset、DataLoader、128 维模型和 SGD 优化器，然后调用一次 `train_one_epoch()`。每批最多 2 条，保留尾批，因此一轮执行 3 次参数更新，实际批大小为 2、2、1。

Dataset 的 `seed=42` 控制负采样，创建模型前的 `torch.manual_seed(42)` 控制模型初始化。两个种子各设置一次；训练批次之间不重新设置种子或创建模型。优化器接收 `model.parameters()`，管理模型的两张词向量表；`lr=0.025` 是本演示使用的学习率。

直接运行会打印样本数、batch 数和这一轮的平均训练损失。脚本在 CPU 上真实更新内存中的参数，但不保存文件；每次重新运行都会从初始化开始，不会接着上一次脚本运行的参数训练。通过 `import demo_train` 导入时不会自动创建数据或训练，只有直接运行脚本或显式调用 `main()` 才执行演示。

这一小步只串起现有的采样、组批、模型和训练函数。正样本与概率仍是手写的学习示例，没有接入 `preprocess.py` 或真实文本；训练一轮并不代表这些词向量已学到有用的语义。当时尚未添加多轮训练或模型保存。

已完成第 6 步的第四小步：将 `demo_train.py` 扩展为 5 个 epoch。设置 `num_epochs = 5`，通过 `for epoch in range(num_epochs)` 每轮调用一次 `train_one_epoch()`，并打印该轮的平均损失。`epoch` 依次为 0、1、2、3、4，打印时使用 `epoch + 1`，因此显示为第 1 到第 5 轮。

模型、优化器、Dataset 和 DataLoader 都只在循环外创建一次，种子也只在循环前设置。下一轮接着上一轮的参数训练，Dataset 的负采样随机状态持续推进；不会在每轮重新初始化词向量或重置随机序列。每轮从头遍历 DataLoader，保留 5 条正样本（含重复样本），执行 3 次参数更新；5 轮共执行 15 次更新。当前 `shuffle=False`，正样本顺序不变，但负样本继续抽取，不保证与上一轮不同。

每轮损失是训练期间的统计值。由于参数和抽到的负样本会变化，它不保证每轮都下降；这个小例子的数值也不能说明词向量已有语义质量。重新运行整个脚本仍会从相同种子重新初始化，不会恢复上次进程的参数。本步仍只使用本地 CPU，没有连接老师的 RTX 5090 机器或保存模型。

已完成第 6 步的第五小步：新增 `demo_text_train.py`，从一句英文 `Cats chase mice while dogs chase balls and birds build nests near trees.` 开始，完整串起 `clean_text()`、`build_vocab()`、负采样分布构建、降采样、编号转换、正样本构造、Dataset、DataLoader 和 5 轮训练。原来的手写编号演示保留不变。

本例设置 `min_count=1`，保留所有出现过的词；负采样分布使用降采样前的词频。降采样设置 `threshold=0.05, seed=42`，让短文本也能演示丢弃部分词的流程，这不是正式大语料的推荐设置。预处理只在训练循环前执行一次，后续各轮复用正样本，在 Dataset 读取时持续抽取负样本，不在每轮重新降采样或设置种子。

清洗后有 13 个 token、12 个词表项；降采样后保留 11 个 token。其中第二次出现的 `chase` 和唯一一次出现的 `balls` 被丢弃，但这两个词仍在原词表中，词频仍分别为 2 和 1。`balls` 因此不再参与本例的正样本构造，但仍可被抽为负样本。词表大小应使用 `len(word_to_id)`，不能改为降采样后剩余词的数量。

固定窗口半径为 2，从降采样后的编号序列生成 38 条有向正样本。设置 `batch_size=8, drop_last=False`，每轮实际批大小为 8、8、8、8、6，5 轮共执行 25 次参数更新。演示打印 token、词表、降采样结果、编号、正样本数量、前 5 对正样本对应的单词，以及每轮平均训练损失，便于逐步核对。

这里只用一句文本，尚未实现多句边界处理或外部语料文件读取。如果降采样后无法生成正样本，会明确报错；如果更换成候选词不足的极小文本，Dataset 仍会按原有规则报告负样本候选耗尽，不会悄悄放宽排除规则。完成此演示表示小文本的模块衔接已验证，不表示已验证大语料性能、词向量语义质量、GPU 训练或模型保存与加载。

正式语料的首选候选是 [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/eng) 的 English Wikipedia 句子语料。该项目已经进行标记清理、语言识别、句子切分和重复句处理，并提供按句子数量划分的下载规模；句子文件采用 UTF-8，每行是“句子编号 + 制表符 + 完整句子”。计划先用 10K 句子版本测量 token、词表和正样本规模，覆盖度不足时再扩展到 30K；当前尚未下载。

按每句平均 20 个 token、窗口半径 2 粗略估算，10K 句子约有 20 万 token，并在过滤和降采样前产生约 80 万个有向正样本。原 `sample_negatives()` 会为每条正样本重新扫描词表；若词表有 2 万项，数量级可达 160 亿次概率位置检查。这个 CPU Python 瓶颈不会因模型放到 RTX 5090 而自动消失，因此需要在真实语料训练前优化。

已完成第 7 步的第一小步：在 `sampling.py` 中新增 `CumulativeNegativeSampler`。初始化时只检查一次概率，并把 `[p0, p1, ...]` 预先转换为累计概率。正常抽样时生成 `[0, 1)` 内的随机数，通过二分查找定位词编号；词表大小为 V 时，每次定位约为 `O(log V)`，避免每条样本都重新创建完整候选列表。

抽中排除编号时采用拒绝采样，重新从原分布抽取，因此合法词保持条件概率 `P(w | w 不在 excluded_ids)`。如果合法概率低于 0.1，或者正常拒绝达到有界次数仍未取够，采样器会扫描一次词表完成剩余抽样；这样兼顾常见情况的速度和极端情况的可终止性。排除全部正概率候选时仍明确报错，不放回中心词或正上下文。

采样器保存自己的独立随机数生成器；相同概率、种子和调用顺序可以复现，连续调用会继续推进状态，不影响 Python 全局随机状态。概率、累计概率和正概率编号集合保存为不可变副本，外部修改原列表不会改变采样器。

这一小步只实现独立采样器，尚未修改 `Word2VecDataset`；所以现有训练演示仍使用原来的逐样本扫描函数，性能还没有改变。下一小步才把 Dataset 接到累计概率采样器上，并重新验证负样本规则与完整训练流程。

已完成第 7 步的第二小步：`Word2VecDataset` 初始化时创建一个 `CumulativeNegativeSampler`，所有 `__getitem__()` 调用都复用这个对象。读取样本时仍先构造 `{中心词} ∪ {该中心词全部已知正上下文}`，然后把排除集合交给 `negative_sampler.sample()`；负样本规则和返回的三个 CPU long 张量保持不变。

Dataset 不再把概率列表、排除集合和 RNG 反复传给旧的 `sample_negatives()`。累计概率只在 Dataset 初始化时构建一次；每个合法索引读取会推进采样器内部 RNG。为保持前面学习过的接口，`dataset.rng` 仍指向同一个内部 RNG；无效索引在调用采样器之前报错，因此不会推进随机状态。相同种子和访问顺序仍可复现，但优化前后的具体随机编号序列不承诺完全相同。

原 `sample_negatives()` 暂时保留，便于独立调用和对照；训练用 Dataset 已切换到累计概率采样器。常见排除比例下不再为每一条正样本扫描完整词表；合法概率过低或连续拒绝较多时仍可能触发一次可靠回退。当前优化解决了主要的逐样本 Python 扫描问题，但 Dataset 仍逐条生成负样本、正样本仍全部保存在内存中，多进程 worker 也尚未支持，所以还不能直接把 10K 句子语料投入正式训练。

已完成第 8 步的第一小步：从 Leipzig 官方下载 `eng_wikipedia_2016_10K.tar.gz`，文件大小 2,516,875 字节，SHA-256 为 `d0c803b7b10d7b42e2da0a3990ded945143594be89f135266da0e8998ffe4edc`。下载服务器上没有对应的 2021 文件，因此使用经过实际响应确认的 2016 版。压缩包和语料文件保存在 `data/raw` 且继续被 Git 忽略，只让 `data/raw/README.md` 记录来源、校验值、格式与使用条款。

下载后先检查 gzip 文件签名和 tar 成员路径，只从压缩包中读取并提取 `eng_wikipedia_2016_10K-sentences.txt`，没有执行其中的 SQL 或展开其他文件。句子文件大小为 1,340,756 字节，采用 UTF-8；每行经检查是 `Sentence_ID<TAB>Sentence`，句子边界已经保留。

新增 `analyze_corpus.py`，以流式方式读取句子文件。第一遍使用现有 `clean_text()` 统计句子、token 和全语料词频；第二遍保持每句话的边界，分别估算 `min_count=2、5、10` 时的保留词表、保留 token 和窗口半径 2 的有向正样本数量。正样本只按过滤后句子长度计数，不建立几十万个 `(center, context)` 元组，也不执行降采样、负采样、Dataset 或模型训练。

实际统计得到 10,000 句、213,575 个 token、26,576 个不同单词，清洗后没有空句；每句最少 2 个、平均 21.36 个、最多 51 个 token。`min_count=2` 保留 12,460 个词和 93.39% token，估算 737,836 对正样本；`min_count=5` 保留 5,208 个词和 84.56% token，估算 662,416 对；`min_count=10` 保留 2,727 个词和 77.03% token，估算 598,066 对。以上正样本数均为降采样前估算值。

统计公式已与真实创建正样本的函数逐长度核对，人工小语料的词频、保留比例和正样本数也与手算一致；10,000 行真实数据全部通过 UTF-8、整数 ID、制表符、非空句子和 ID 不重复检查。当前工作目录尚未初始化为 Git 仓库，所以只配置了 `.gitignore` 规则，尚未执行 Git 跟踪状态验证或 GitHub 同步。

根据统计结果，第一版正式实验选择 `min_count=5`：相比 2，大幅缩小低频词词表但仍保留 84.56% token；相比 10，又保留更多出现 5～9 次的词。这个值是首轮实验配置，不是固定真理；若以后关注罕见专业词，应结合更多领域语料考虑降低阈值，而不能只靠降低阈值凭空增加上下文证据。

已完成第 8 步的第二小步：`subsample_frequent_words()` 新增关键字参数 `rng`。处理多句话时可以先创建一次 `rng = random.Random(42)`，随后每句话传入同一个对象；随机状态跨句继续推进，不再让每句话从相同种子的同一序列起点重新开始。原有的 `seed=42` 单次调用方式继续有效，`seed` 和 `rng` 不能同时传入。

共享 RNG 只解决跨句随机状态问题，不会把句子拼接起来。降采样仍逐句执行，每句话得到自己的保留 token 列表，正样本也必须逐句生成；词频和负采样分布则继续使用降采样前的全语料统计。函数不修改 token、词频或全局随机状态，相同种子、句子顺序和调用顺序仍可复现。

已完成第 8 步的第三小步：扩展 `analyze_corpus.py`，在 `min_count=5` 的过滤结果上，对 `threshold=1e-3、1e-4、1e-5` 分别创建一个 `random.Random(42)`。第三遍流式读取时，每个阈值在全部 10,000 句话之间复用自己的 RNG；每句话独立降采样并按保留长度计算正样本数量，不连接句子首尾。

除了保留 token 和有向正样本数，统计还记录降采样后不足两个 token 的句子数量，因为这些句子无法产生正样本。三个阈值从同一种子开始并按相同 token 顺序消耗随机数，便于比较阈值影响；这只用于确定首轮配置，没有生成或保存正样本列表。

实际结果如下：`1e-3` 保留 121,983 个 token（过滤后 token 的 67.54%）、427,988 对正样本，26 句话不足两个 token；`1e-4` 保留 78,893 个 token（43.68%）、256,028 对正样本，178 句话不足两个 token；`1e-5` 只保留 29,254 个 token（16.20%）、65,560 对正样本，并使 2,586 句话不足两个 token。统计使用相同种子可复现，阈值降低时保留 token 和正样本单调减少。

这次三组统计还发现一个新的性能问题：`subsample_frequent_words()` 每次调用都会重新遍历 5,208 个词计算保留概率，而逐句处理会调用 10,000 次。三组阈值的本地统计因此耗时约一分钟。正式预处理前应把每个词的保留概率预计算一次并跨句复用；共享 RNG 本身不能解决这部分重复计算。

第一版正式实验选择 `threshold=1e-4`。它比 `1e-3` 更明显地降低高频词和训练量，同时仍保留 78,893 个 token、256,028 对正样本，只有 178 句话不足两个 token。选择依据是本次小语料的规模平衡，并不表示降采样越强越好；后续仍可用 `1e-3` 做对照。

已完成第 8 步的第四小步：在 `sampling.py` 中新增 `FrequentWordSubsampler`。初始化时根据全语料过滤后的词频只计算一次 `keep_probabilities`，并保存一个连续 RNG；之后每句话调用 `sample(tokens)` 时只遍历当前句子的 token，不再遍历整个词表。原 `subsample_frequent_words()` 保持兼容，它现在创建一个临时 `FrequentWordSubsampler` 后完成单次调用。

`analyze_corpus.py` 已改为每个候选阈值创建一个 `FrequentWordSubsampler`，10,000 句话复用三个对象。优化只缓存确定性的词频概率，不缓存 token 或正样本，不改变降采样公式、随机数消费顺序、句子边界或固定种子的统计结果。

同一台本地机器上，包含三个阈值的完整统计由优化前约一分钟降到约 0.64 秒；全部 token、正样本和短句统计与优化前逐项相同。这个计时用于确认重复计算已消除，不代表服务器或其他语料的固定性能。

已完成按句子动态训练的数据管线。`SentencePairDataset` 只保存句子 token ID，每次进入 `__iter__()` 都重新生成正样本；`SentenceWord2VecDataset` 在此基础上即时执行累计概率负采样。初始化时只遍历动态生成器统计正样本数并建立“中心词到不重复正上下文”的集合，不保存 256,028 个 Python 样本元组。

每个 epoch 会打乱句子编号，句子内部词序和窗口关系保持不变。单进程时，句子顺序 RNG 只在 Dataset 初始化时创建一次；多进程时，每个 worker 只遍历按 worker 编号分配给自己的句子，并使用 PyTorch 提供的独立 worker seed 打乱该分片和抽取负样本。因此多 worker 不会重复或漏掉句子，相同 seed 的新 DataLoader 可以复现完整样本序列。`DataLoader` 仍使用 `shuffle=False`，因为句子分片和局部打乱由 Dataset 负责；组批形状为 `[B]、[B]、[B, K]`。

新增 `prepare_data.py`，按“清洗与全语料词频 → min_count=5 → 负采样分布 → threshold=1e-4 降采样 → 逐句 token ID”的顺序准备数据。真实结果为词表 5,208、降采样后 78,893 个 token、每轮动态产生 256,028 个正样本；负采样分布使用降采样前的稳定词频。

`train.py` 已加入正式配置和入口：embedding_dim=50、num_negatives=5、batch_size=512、epochs=5、Adam learning_rate=0.01。训练循环会自动把 batch 移到模型参数所在设备，每轮记录平均 loss 并原子覆盖最新 checkpoint；checkpoint 包含模型、优化器、epoch、word_to_id、配置和 loss 历史。SGD learning_rate=0.05 的服务器诊断中，5 轮 loss 基本停在初始化基准 4.158883；改用 Adam learning_rate=0.01 后，同一真实语料的本地诊断 loss 在 3 轮中从 3.002734 降至 2.292595，因此正式入口采用 Adam。

为扩展到 100K 句语料，`train.py` 和 `SentenceWord2VecDataset` 已支持多个 DataLoader worker。句子先按 worker 编号分片，再在各自分片内打乱；每个 worker 使用独立负采样 RNG。训练循环可按固定 batch 间隔打印近似进度，并使用 pinned memory 与 non-blocking GPU 搬运。两 worker 测试已验证正样本计数和多重集合与单进程完全一致、负样本排除规则保持不变、相同 seed 可复现。服务器单 worker 的 100K 基线为每轮 6,840,310 个正样本、3,340 个 batch、22 分 10 秒，后续用该基线衡量多 worker 加速效果。

`inference.py` 已能用 `weights_only=True` 从 checkpoint 重建模型和双向词表，使用中心词向量的余弦相似度排除查询词自身并返回 top-k，也提供可直接运行的命令行入口。100K 句正式模型已经得到清晰结果：`music` 邻近 `pop、dance、musical、albums、musicians`，`city` 邻近 `town、area、park、county、river`，`war` 邻近 `force、military、forces、soviet、navy、campaign、civil、allied`。

`visualize.py` 已实现 PCA 和 t-SNE 二维降维、带标签散点图保存以及 checkpoint 命令行入口。降维前先对每个词向量做 L2 归一化，使图中的输入与相似词查询使用的余弦方向一致。默认模式绘制最常见的词；重复传入 `--group 标签:word1,word2` 可以选择有意义的语义词并按组着色，避免最高频功能词遮住主题结构。PCA 用于观察整体线性方向，t-SNE 固定训练 seed 并侧重局部邻域；二维 t-SNE 的全局距离不能当作精确比例。分组图用于解释模型结果，不作为独立的量化评估。

100K 句正式实验在 RTX 5090 上使用 8 个 DataLoader worker。配置为词表 14,951、100 维、窗口半径 5、5 个负样本、batch size 2,048、Adam 学习率 0.01，共训练 10 轮。每轮有 6,840,310 个正样本和约 3,340 个 batch；总耗时 34 分 49 秒。平均 loss 从 2.600747 降至 2.229495，checkpoint 为 `checkpoints/word2vec_100K.pt`（约 35 MB）。

## 当前编辑位置

后续编辑以本文件所在的原工作区项目为准。`D:\code\word2vec` 保留为此前的副本，不会自动同步后续修改。

## 目录结构

```text
word2vec/
├── README.md             # 项目说明与学习进度
├── requirements.txt      # 计划使用的第三方依赖
├── .gitignore            # 忽略缓存、本地数据和训练产物
├── preprocess.py         # 数据清洗、词表、token → id、Skip-gram 正样本
├── sampling.py           # 高频词 subsampling、负采样分布与负样本生成
├── dataset.py            # PyTorch Dataset
├── demo_dataloader.py    # DataLoader 组批演示，不训练
├── demo_train.py         # 小样本多 epoch 训练演示，不保存
├── demo_text_train.py    # 英文文本到多轮训练的完整流程演示，不保存
├── analyze_corpus.py     # 流式检查正式语料格式并统计训练规模
├── prepare_data.py       # 准备逐句 token ID、词表和负采样概率
├── model.py              # SkipGramNegSampling 模型
├── train.py              # 优化器、epoch、训练循环、保存 checkpoint
├── inference.py          # 相似词查询
├── visualize.py          # PCA / t-SNE 可视化
├── plot_loss.py           # 从 checkpoint 生成训练 loss 曲线
├── EXPERIMENT_REPORT.md   # 100K 句正式实验报告
├── data/
│   ├── raw/              # 原始语料
│   └── processed/        # 处理后的数据与词表
├── checkpoints/          # 模型检查点
└── figures/              # 可视化结果
```

空目录中的 `.gitkeep` 仅用于在 Git 中保留目录。

## 后续步骤

1. 增加定量评估，例如人工相似词小测试或公开的词相似度数据集，避免只凭几次 top-k 查询判断质量。
2. 如需继续扩大语料，先记录新的数据版本和 checksum，再用当前100K句结果作为基线对照。

`preprocess.py` 负责正样本构造逻辑，`sampling.py` 负责采样逻辑；数据管线将按上面的顺序衔接，降采样发生在正样本构造之前。

## 运行当前演示

在 `word2vec` 项目目录中运行：

```powershell
& 'D:\anaconda\python.exe' -B .\demo_dataloader.py
```

脚本会打印 3 个 batch 的内容和形状。它使用代码中的小样本，不下载语料，也不会训练、保存模型或修改项目数据。被其他文件导入时不会自动执行演示。

运行多 epoch 训练演示（当前为 5 轮）：

```powershell
& 'D:\anaconda\python.exe' -B .\demo_train.py
```

它会打印 5 条样本、每轮 3 个 batch，以及从 `Epoch 1 mean loss` 到 `Epoch 5 mean loss` 的 5 条记录。每条记录是对应轮次按实际样本数加权的训练损失。训练只影响本次进程中模型的参数，不下载语料、不保存 checkpoint，也不会改动 `D:\code\word2vec` 中的旧副本。

运行英文文本的完整流程演示：

```powershell
& 'D:\anaconda\python.exe' -B .\demo_text_train.py
```

它使用代码中的一句英文，不需要准备或下载文件。会打印预处理结果，以及每轮 38 条正样本、5 个 batch 对应的训练损失。运行时仅更新内存中的 CPU 模型，导入模块不会自动训练，不保存 checkpoint。

## 环境说明

`requirements.txt` 只列出计划使用的依赖。当前验证使用现有的 `D:\anaconda\python.exe`（Python 3.13.11）及已安装的 PyTorch `2.9.0+cu130`。已下载并检查 Leipzig 10K 句子语料，使用真实数据完成准备、动态 DataLoader 和 3 个 CPU batch 的冒烟测试；小型临时语料已验证多轮训练和 checkpoint 保存。尚未验证 GPU、运行真实 5 轮、生成正式 checkpoint 或同步 GitHub。换用其他 Python 环境时，需要另外确认依赖是否齐全。
