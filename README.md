# ml-leetcode

手写 ML 算法的刷题库。每道题都有题面、代码模板、参考答案和单元测试，部分题目还配了和代码联动的训练动画。

## 网页版

在线打开：**https://laulauraluo.github.io/ml-leetcode/**

- 左边：题目描述和动画。动画播放时，右边编辑器会高亮正在执行的那几行代码；改代码里的超参数（比如 `n_estimators`、`max_depth`），动画会按新参数重放。
- 右边：写代码，点「运行测试」直接在浏览器里跑 Python（[Pyodide](https://pyodide.org)，不用装环境）。代码会自动存在你的浏览器里。

本地打开（页面要用 `fetch` 读题目文件，不能直接双击 `index.html`）：

```bash
python3 -m http.server
```

然后访问 http://localhost:8000。

## 命令行刷

```bash
pip install -r requirements.txt
python problems/001_rf_vs_decision_tree/test_solution.py              # 测你的 solution.py
SOL=reference python problems/001_rf_vs_decision_tree/test_solution.py  # 测参考答案
```

## 题目

003 起还没做，按面经里出现的公司数排序，公司数相同再看笔记数。

| # | 题 | 难度 | 考点 | 来源 |
|---|---|---|---|---|
| 001 | [Random forest vs decision tree（sklearn）](problems/001_rf_vs_decision_tree/README.md) | Easy | sklearn API、bagging 为什么比单棵树准 | — |
| 002 | [手写 random forest（numpy）](problems/002_random_forest_from_scratch/README.md) | Medium | Gini、bootstrap、随机 feature、投票 | — |
| 003 | Cosine similarity Top-K 检索 | Easy | BoW 计数向量、向量化 cosine（零向量）、argsort 取 top-k、排除已看文档 | yahoo面经, workday面经, apple面经 |
| 004 | 手写 logistic regression | Medium | sigmoid、BCE 梯度、mini-batch GD；follow-up：加 L1 后 loss 还是 convex 吗 | apple面经, intuit面经, 非面经 |
| 005 | 手写 KNN | Easy | broadcast 算距离（欧氏 / 曼哈顿 / cosine）、argpartition 取 k 近邻、多数投票 | apple面经, workday面经 |
| 006 | 手写 linear regression | Easy | MSE 梯度、mini-batch GD、和 closed form（lstsq）对照 | apple面经, 非面经 |
| 007 | Softmax、cross entropy 和 KL | Easy | 减 max 防溢出、log-softmax、ignore_index（prompt 位置设 -100）、KL | apple面经, 非面经 |
| 008 | Scaled dot-product attention 和 multi-head attention | Medium | QKᵀ/√d_k、mask、softmax 沿哪一维、分头 reshape / transpose 再拼回 | apple面经, 非面经 |
| 009 | 手写 backprop：MLP + residual | Hard | Linear / ReLU 反向、residual 梯度相加、gradient check；follow-up：fp16 loss scaling | msft面经, 非面经 |
| 010 | Bootstrap 置信区间和 z-test | Easy | 有放回重采样、percentile CI、two-sample z-score 和 p-value | apple面经, 非面经 |
| 011 | Beam search | Medium | 累积 log-prob、beam × vocab 展开再取 top-B、和 greedy 对比、长度归一化 | apple面经, 非面经 |
| 012 | Transformer block（pre-LN） | Medium | sinusoidal 位置编码、LayerNorm、FFN（4× hidden）、残差 | apple面经 |
| 013 | KV cache 增量解码 | Hard | prefill vs decode、K/V 沿 seq 拼接、增量输出和整段 causal forward 一致 | apple面经 |
| 014 | Confusion matrix | Easy | K×K 计数、行是真实列是预测（sklearn 约定）、由它算 precision / recall / F1 | apple面经 |
| 015 | 手写 k-means | Easy | broadcast 距离矩阵、分配 / 更新交替、收敛判断、空簇 | apple面经 |
| 016 | 手写 multinomial naive Bayes | Medium | Laplace smoothing、log 空间防下溢、一次矩阵乘法算所有类 | workday面经 |
| 017 | 文档排序：多级 tie-break + query cache | Medium | doc length / keyword overlap / cosine 三个 feature、不用 sorted() 取 top-k、cache key 设计 | workday面经 |
| 018 | Tiny next-token 模型 | Medium | inputs = seq[:, :-1]、targets = seq[:, 1:]、flatten 后算 CE、embedding → MLP | msft面经 |
| 019 | FlashAttention（tiling + online softmax） | Hard | 分块算 QKᵀ、running max 和 sum、旧结果按新 max 缩放、和标准 attention 数值一致 | apple面经 |
| 020 | In-batch negatives（InfoNCE）+ Recall@K | Medium | U·Vᵀ/τ 相似度矩阵、对角线是正样本、CE、Recall@K | 非面经 |
| 021 | AUC | Medium | 按分数排序算、并列算 0.5、和梯形 ROC 面积一致 | 非面经 |
| 022 | 手写 LoRA linear | Medium | 冻结 W₀、ΔW = (α/r)·BA、B 初始化为 0、参数量、merge 后输出不变 | 非面经 |

## 加新题

1. 复制一个题目文件夹，改四个文件：`README.md`（题面）、`solution.py`（模板）、`reference.py`（答案）、`test_solution.py`（测试）
2. 在 `index.html` 的 `PROBLEMS` 里加一行
