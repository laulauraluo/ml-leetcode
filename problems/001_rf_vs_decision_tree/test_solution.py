import os
import runpy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
_ns = None


def ns():  # 跑一遍 solution 脚本，拿到它留下的变量
    global _ns
    if _ns is None:
        _ns = runpy.run_path(os.path.join(HERE, os.environ.get("SOL", "solution") + ".py"))
    return _ns


def split():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_data_split():
    X_train, X_test, y_train, y_test = split()
    s = ns()
    assert s["X"].shape == (1000, 4), s["X"].shape
    assert np.array_equal(s["X_train"], X_train) and np.array_equal(s["y_test"], y_test), "划分和 random_state=42 的结果对不上"


def test_decision_tree():
    s = ns()
    m = s["dt_model"]
    assert isinstance(m, DecisionTreeClassifier), type(m)
    assert m.random_state == 42, m.random_state
    assert hasattr(m, "tree_"), "dt_model 还没 fit"
    X_train, X_test, y_train, y_test = split()
    want = DecisionTreeClassifier(random_state=42).fit(X_train, y_train).score(X_test, y_test)
    assert np.isclose(s["dt_score"], want), (s["dt_score"], want)


def test_random_forest():
    s = ns()
    m = s["rf_model"]
    assert isinstance(m, RandomForestClassifier), type(m)
    assert m.n_estimators == 100 and m.random_state == 42, (m.n_estimators, m.random_state)
    assert hasattr(m, "estimators_"), "rf_model 还没 fit"
    X_train, X_test, y_train, y_test = split()
    want = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).score(X_test, y_test)
    assert np.isclose(s["rf_score"], want), (s["rf_score"], want)


def test_forest_beats_tree():
    s = ns()
    assert s["rf_score"] > s["dt_score"], (s["dt_score"], s["rf_score"])


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
