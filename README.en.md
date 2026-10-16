# ml-leetcode

[中文](README.md) | English

A LeetCode-style practice set for implementing ML algorithms by hand. Each problem has a statement, a code template, a reference solution, and unit tests. Some problems also have a training animation synced with the code.

The web UI and the problem statements are in Chinese for now.

## Web version

Open online: **https://laulauraluo.github.io/ml-leetcode/**

- Left: the problem statement and the animation. While the animation plays, the editor on the right highlights the lines being executed. Change a hyperparameter in the code (for example `n_estimators` or `max_depth`) and the animation replays with the new value.
- Right: write your code and click 运行测试 (Run tests) to run Python in the browser ([Pyodide](https://pyodide.org), nothing to install). Your code is saved in your browser.

To open it locally, serve the repo root (the page uses `fetch` to read problem files, so double-clicking `index.html` does not work):

```bash
python3 -m http.server
```

Then visit http://localhost:8000.

## Command line

```bash
pip install -r requirements.txt
python problems/001_rf_vs_decision_tree/test_solution.py              # test your solution.py
SOL=reference python problems/001_rf_vs_decision_tree/test_solution.py  # test the reference solution
```

## Problems

Problems 003 and later are planned but not written yet. They are ordered by how many companies' interview notes mention them, then by how many notes do.

| # | Problem | Difficulty | Key points | Source |
|---|---|---|---|---|
| 001 | [Random forest vs decision tree (sklearn)](problems/001_rf_vs_decision_tree/README.md) | Easy | sklearn API, why bagging beats a single tree | — |
| 002 | [Random forest from scratch (numpy)](problems/002_random_forest_from_scratch/README.md) | Medium | Gini, bootstrap, random features, voting | — |
| 003 | Cosine similarity top-K retrieval | Easy | bag-of-words count vectors, vectorized cosine (zero vectors), argsort for top-k, excluding seen documents | Yahoo, Workday, Apple |
| 004 | Logistic regression from scratch | Medium | sigmoid, BCE gradient, mini-batch GD; follow-up: is the loss still convex with L1? | Apple, Intuit, practice notes |
| 005 | KNN from scratch | Easy | broadcast distances (Euclidean / Manhattan / cosine), argpartition for the k nearest, majority vote | Apple, Workday |
| 006 | Linear regression from scratch | Easy | MSE gradient, mini-batch GD, check against the closed form (lstsq) | Apple, practice notes |
| 007 | Softmax, cross entropy and KL | Easy | subtract the max for stability, log-softmax, ignore_index (-100 on prompt tokens), KL | Apple, practice notes |
| 008 | Scaled dot-product and multi-head attention | Medium | QKᵀ/√d_k, mask, which axis softmax runs over, split heads with reshape / transpose and merge back | Apple, practice notes |
| 009 | Backprop by hand: MLP + residual | Hard | Linear / ReLU backward, residual gradients add up, gradient check; follow-up: fp16 loss scaling | Microsoft, practice notes |
| 010 | Bootstrap confidence interval and z-test | Easy | resampling with replacement, percentile CI, two-sample z-score and p-value | Apple, practice notes |
| 011 | Beam search | Medium | cumulative log-prob, expand beam × vocab then keep the top B, compare with greedy, length normalization | Apple, practice notes |
| 012 | Transformer block (pre-LN) | Medium | sinusoidal positional encoding, LayerNorm, FFN (4× hidden), residual | Apple |
| 013 | Incremental decoding with a KV cache | Hard | prefill vs decode, concatenate K/V along the sequence, incremental output matches a full causal forward | Apple |
| 014 | Confusion matrix | Easy | K×K counts, rows are true labels and columns are predictions (sklearn convention), precision / recall / F1 from it | Apple |
| 015 | K-means from scratch | Easy | broadcast distance matrix, alternate assign / update, convergence check, empty clusters | Apple |
| 016 | Multinomial naive Bayes from scratch | Medium | Laplace smoothing, log space against underflow, all classes in one matrix multiply | Workday |
| 017 | Document ranking: multi-key tie-breaks + query cache | Medium | doc length / keyword overlap / cosine features, top-k without sorted(), cache key design | Workday |
| 018 | Tiny next-token model | Medium | inputs = seq[:, :-1], targets = seq[:, 1:], flatten before CE, embedding → MLP | Microsoft |
| 019 | FlashAttention (tiling + online softmax) | Hard | blockwise QKᵀ, running max and sum, rescale old results to the new max, matches standard attention numerically | Apple |
| 020 | In-batch negatives (InfoNCE) + Recall@K | Medium | U·Vᵀ/τ similarity matrix, positives on the diagonal, CE, Recall@K | practice notes |
| 021 | AUC | Medium | compute from sorted scores, ties count 0.5, matches the trapezoidal ROC area | practice notes |
| 022 | LoRA linear layer | Medium | frozen W₀, ΔW = (α/r)·BA, B initialized to 0, parameter count, same output after merging | practice notes |

## Adding a problem

1. Copy a problem folder and edit its four files: `README.md` (statement), `solution.py` (template), `reference.py` (solution), `test_solution.py` (tests).
2. Add a line to `PROBLEMS` in `index.html`.
