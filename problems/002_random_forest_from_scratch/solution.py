import numpy as np


def gini(y):
    raise NotImplementedError


def build(X, y, n_feats=None, rng=None):
    raise NotImplementedError


def predict(node, x):
    raise NotImplementedError


def bootstrap(n, rng):
    raise NotImplementedError


def random_forest(X, y, n_trees=25, n_feats=1, rng=None):
    raise NotImplementedError


def forest_predict(trees, x):
    raise NotImplementedError
