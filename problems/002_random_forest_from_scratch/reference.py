import numpy as np


def gini(y):
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - (p ** 2).sum()


def build(X, y, n_feats=None, rng=None):
    if len(set(y)) == 1:                          # 纯了就停：叶子 = 这个类
        return int(y[0])
    feats = range(X.shape[1]) if n_feats is None else rng.choice(X.shape[1], n_feats, replace=False)
    best = None
    for f in feats:                               # DT: 所有 feature 都试；RF: 只试随机抽的几个
        for t in np.unique(X[:, f])[:-1]:
            l = X[:, f] <= t
            score = l.mean() * gini(y[l]) + (~l).mean() * gini(y[~l])
            if best is None or score < best[0]:
                best = (score, f, t, l)
    if best is None:                              # 抽到的 feature 切不动
        return int(np.bincount(y).argmax())
    _, f, t, l = best
    return (f, t, build(X[l], y[l], n_feats, rng), build(X[~l], y[~l], n_feats, rng))


def predict(node, x):
    while isinstance(node, tuple):
        f, t, left, right = node
        node = left if x[f] <= t else right
    return node


def bootstrap(n, rng):
    return rng.integers(0, n, n)


def random_forest(X, y, n_trees=25, n_feats=1, rng=None):
    trees = []
    for _ in range(n_trees):
        idx = bootstrap(len(X), rng)                        # 区别 1: 每棵树只看一份 bootstrap 样本
        trees.append(build(X[idx], y[idx], n_feats, rng))   # 区别 2: 每个节点只看随机 n_feats 个 feature
    return trees


def forest_predict(trees, x):
    return int(np.bincount([predict(t, x) for t in trees], minlength=2).argmax())  # 区别 3: 投票，平票取 0
