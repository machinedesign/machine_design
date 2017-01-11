import numpy as np
from machinedesign.layers import ksparse
from machinedesign.layers import winner_take_all_spatial
from machinedesign.layers import axis_softmax

import keras.backend as K


def test_k_sparse():
    act = ksparse(zero_ratio=0.7)
    X = K.placeholder(shape=(None, 10))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 10))
    y = pred([x])
    assert (y == 0).sum() == 7 * nb


def test_winner_take_all_spatial():
    for nb_active in (0, 1, 2, 3, 100, 101):
        act = winner_take_all_spatial(nb_active=nb_active)
        X = K.placeholder(shape=(None, 1, 10, 10))
        pred = K.function([X], act.call(X))
        np.random.seed(42)
        nb = 100
        x = np.random.uniform(-1, 1, size=(nb, 1, 10, 10))
        y = pred([x])
        assert np.all((y != 0).sum(axis=(2, 3)) == min(nb_active, 100))


def test_axis_softmax():
    act = axis_softmax(axis=2)
    X = K.placeholder(shape=(None, 2, 10, 10))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 2, 10, 10))
    y = pred([x])
    assert np.allclose(y.sum(axis=2), 1)
