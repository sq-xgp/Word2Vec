# Autoencoder 运行指南

## 1. 准备数据

```bash
python prepare_autoencoder_data.py --checkpoint checkpoints/w2v_dual_dot_d100_w5_n5_bs2048_lr0.01_mc10_ss1em04_e20_vf0.1_p3_seed42.pt
```

提取 center_embeddings，断开梯度，逐行 L2 归一化，固定 seed=42，
按 80% / 10% / 余数划分词。输入与重建目标相同，不需要正负词对。
默认输出 data/autoencoder/dual_dot_d100_dataset.pt。
若文件已存在，使用已有文件，或用 --output 指定新路径。
文件包含 X、words、train_idx、val_idx、test_idx 及预处理信息。

## 2. 训练

```bash
python train_autoencoder.py
```

默认结构 100→32→2→32→100，仅两个隐藏层后有 ReLU。
Adam，lr=0.001，batch=64，最多 200 轮，patience=15。
严格更低的验证 MSE 算作改善，min_delta=0。
自动使用 CUDA（可用时）；用 --device cpu 可强制 CPU。
支持 --data、--output-root、--epochs、--patience、--batch-size、--learning-rate、--seed。
每次运行在 checkpoints/autoencoder/时间戳/ 创建 best.pt 与 history.csv。
best.pt 是推理 checkpoint，不包含精确续训所需的优化器和随机状态。

## 3. 推理与绘图

将以下时间戳替换为自己训练打印出的目录：
```bash
python infer_autoencoder.py --checkpoint checkpoints/autoencoder/20260929_170321_581954/best.pt
python visualize_autoencoder.py --run-dir checkpoints/autoencoder/20260929_170321_581954
```

推理输出 embeddings_2d.pt（Z、words、测试 MSE 等）。
绘图输出 plots/embeddings_2d.png 与 plots/loss_curve.png，无图形界面的服务器也可生成。
推理不打乱词序，words[i] 对应 Z[i]；只需编码器生成坐标。
--data 可指定迁移后的数据路径，不依赖 checkpoint 内记录的旧绝对路径。
新 checkpoint 会校验数据文件哈希；本次旧服务器 checkpoint 无哈希，
兼容加载时会提醒核对原数据，同时检查结构和预处理元数据。

## 4. PCA 对照

```bash
python compare_autoencoder_pca.py --checkpoint checkpoints/autoencoder/20260929_170321_581954/best.pt
```

只用训练词拟合 PCA，使用相同测试词评估 MSE 和平均 top-10 邻居保留率。
候选为全词表，排除自身；原空间余弦，二维欧氏。
输出 comparison.json，可用 --k、--output、--data 定制。
测试集用于最终报告，不用于选 epoch 或反复调参。

## 5. 测试与产物管理

```bash
python -m unittest discover -v
```

数据、训练权重及大规模产物由 .gitignore 排除；报告中的小型实验图片保存在 docs/figures/autoencoder/。
原始服务器训练产物仍在服务器，报告图已从本机下载目录归档。
数学解释、实验事实与限制见 AUTOENCODER_REPORT.md。
