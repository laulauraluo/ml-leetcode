import importlib
import os

import numpy as np

sol = importlib.import_module(os.environ.get("SOL", "solution"))


def make_moons(n, noise, rng):
    t, y = rng.uniform(0, np.pi, n), rng.integers(0, 2, n)
    X = np.where(y[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    return X + rng.normal(0, noise, X.shape), y.astype(np.intp)  # 网页里的 Pyodide 是 32 位，bincount 只收 intp


def test_gini():
    assert sol.gini(np.array([0, 0, 0])) == 0
    assert np.isclose(sol.gini(np.array([0, 1, 0, 1])), 0.5)
    assert np.isclose(sol.gini(np.array([0, 0, 0, 1])), 0.375)


def test_predict_follows_split():
    tree = (0, 0.5, 0, (1, 2.0, 1, 0))  # x0 <= 0.5 → 0；否则 x1 <= 2.0 → 1，> 2.0 → 0
    assert sol.predict(tree, np.array([0.2, 9.0])) == 0
    assert sol.predict(tree, np.array([0.8, 1.0])) == 1
    assert sol.predict(tree, np.array([0.8, 3.0])) == 0


def test_full_tree_memorizes_train():
    X, y = make_moons(100, 0.3, np.random.default_rng(0))
    tree = sol.build(X, y)
    assert all(sol.predict(tree, x) == t for x, t in zip(X, y))


def test_bootstrap_unique_ratio():
    idx = sol.bootstrap(10_000, np.random.default_rng(0))
    assert len(idx) == 10_000
    assert abs(len(set(idx)) / 10_000 - (1 - 1 / np.e)) < 0.01


def test_vote_and_tie():
    assert sol.forest_predict([1, 1, 0], np.zeros(2)) == 1  # 一个叶子本身也是一棵树
    assert sol.forest_predict([1, 0], np.zeros(2)) == 0


def test_trees_differ():
    rng = np.random.default_rng(0)
    X, y = make_moons(100, 0.3, rng)
    trees = sol.random_forest(X, y, n_trees=5, rng=rng)
    assert len(trees) == 5
    assert len({str(t) for t in trees}) > 1


def test_forest_accuracy():
    accs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Xtr, ytr = make_moons(200, 0.3, rng)
        Xte, yte = make_moons(1000, 0.3, rng)
        trees = sol.random_forest(Xtr, ytr, rng=rng)
        accs.append(np.mean([sol.forest_predict(trees, x) == t for x, t in zip(Xte, yte)]))
    assert np.mean(accs) >= 0.85, accs


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("PASS", name)
        except Exception as e:
            print("FAIL", name, "-", type(e).__name__, e)
    print(f"{passed}/{len(tests)} passed")
    raise SystemExit(passed != len(tests))
