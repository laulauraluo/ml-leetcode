# ml-leetcode

手写 ML 算法的刷题库。每道题都有题面、代码模板、参考答案和单元测试，部分题目还配了和代码联动的训练动画。

## 网页版

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

| # | 题 | 难度 | 考点 |
|---|---|---|---|
| 001 | [Random forest vs decision tree（sklearn）](problems/001_rf_vs_decision_tree/README.md) | Easy | sklearn API、bagging 为什么比单棵树准 |
| 002 | [手写 random forest（numpy）](problems/002_random_forest_from_scratch/README.md) | Medium | Gini、bootstrap、随机 feature、投票 |

## 加新题

1. 复制一个题目文件夹，改四个文件：`README.md`（题面）、`solution.py`（模板）、`reference.py`（答案）、`test_solution.py`（测试）
2. 在 `index.html` 的 `PROBLEMS` 里加一行
