# Word2Vec 与 Autoencoder 学习项目

使用 PyTorch 实现 Skip-gram with Negative Sampling（SGNS），并将训练好的词向量通过 Autoencoder 降到二维，包含训练、推理、可视化与 PCA 对照评估。

## 安装与检查

```bash
pip install -r requirements.txt
python -m unittest discover -v
```

本地验证环境为 PyTorch 2.9.0+cu130；本次服务器实验为 PyTorch 2.12.1+cu130。
根据设备安装适配的 PyTorch；数据和 checkpoint 不随代码提交。

## 模块入口

| 环节 | 文件 |
|---|---|
| 文本处理与语料准备 | preprocess.py、prepare_data.py、analyze_corpus.py |
| 正负样本与采样 | dataset.py、sampling.py |
| Word2Vec 模型及训练 | model.py、train.py |
| 相似词查询 | inference.py |
| Word2Vec 绘图 | visualize.py、plot_loss.py |
| Autoencoder 数据准备 | prepare_autoencoder_data.py |
| Autoencoder 网络 | autoencoder_model.py |
| 公共数据校验、模型加载 | autoencoder_utils.py |
| Autoencoder 训练 | train_autoencoder.py |
| 二维推理与测试误差 | infer_autoencoder.py |
| 二维图与训练曲线 | visualize_autoencoder.py |
| PCA 公平对照 | compare_autoencoder_pca.py |

全部命令从项目根目录运行。查看 Word2Vec 参数：
```bash
python train.py --help
python inference.py --help
python visualize.py --help
```

小规模入门演示保留 demo_dataloader.py（理解 batch）和 demo_text_train.py（文本到训练）。
重复的固定编号训练演示 demo_train.py 已删除。

## Autoencoder 快速开始

[运行指南](AUTOENCODER_README.md) 给出完整命令与产物说明。
[详细实验报告](AUTOENCODER_REPORT.md) 记录本次数据、数学定义、模型、训练曲线、测试与局限。

本次实验：14,951 个 100 维词向量，压缩为 2 维。
Autoencoder 测试 MSE 0.00746061、top-10 邻居保留率 1.02%；PCA 分别为 0.00776938、0.58%。
本次 Autoencoder 相对更好，但两者都没有良好保留精确邻居；精确检索仍使用 100 维向量。

## 历史资料

- [100K 句实验](EXPERIMENT_REPORT.md)
- [模型配置与四模型对照](MODEL_UPDATE_REPORT.md)
- [Embedding 表参数量对照](EMBEDDING_TABLE_COMPARISON_REPORT.md)
- [旧任务交接记录](TASK_HANDOFF_REPORT.md)（历史状态）
- [分步学习记录](docs/LEARNING_HISTORY.md)（归档，不作为当前运行指南）
