# 002 手写 random forest（numpy）

只用 numpy 实现一个二分类 random forest。

## 树的数据结构

- 叶子：一个 int，就是预测的类别（0 或 1）
- 内部节点：tuple `(f, t, left, right)`，`x[f] <= t` 走 `left`，否则走 `right`

## 要实现的函数

| 函数 | 做什么 |
|---|---|
| `gini(y)` | `y` 是 0/1 的 int 数组，返回 Gini impurity：1 − Σ p_k² |
| `build(X, y, n_feats=None, rng=None)` | 递归长树，直到每个叶子都纯。`n_feats=None` 时每个节点试所有 feature（就是普通 decision tree）；否则每个节点用 `rng` 随机抽 `n_feats` 个 feature，只在这几个里找最好的切分 |
| `predict(node, x)` | 单棵树预测一个样本 |
| `bootstrap(n, rng)` | 有放回地抽 n 个下标 |
| `random_forest(X, y, n_trees=25, n_feats=1, rng=None)` | 每棵树用一份 bootstrap 样本，调 `build(..., n_feats, rng)` |
| `forest_predict(trees, x)` | 多数投票，平票返回 0 |

## 测试会检查

- `gini` 的数值
- `predict` 按 `(f, t, left, right)` 正确走分支
- 长满的单棵树在训练集上 100% 正确
- `bootstrap` 抽到的不同下标比例 ≈ 1 − 1/e ≈ 63.2%
- 投票和平票规则
- 森林里的树互不相同
- 在 moons 数据上森林的 test acc ≥ 0.85（3 个 seed 取平均）

## 做完再想

1. 为什么 bootstrap 抽到的不同点比例会趋近 1 − 1/e？
2. 把 `n_feats` 改成 `None`（只剩 bagging），test acc 会变多少？为什么？
3. 单棵树 train acc 100%、test 只有 ~87%，差掉的这部分是 bias 还是 variance？
