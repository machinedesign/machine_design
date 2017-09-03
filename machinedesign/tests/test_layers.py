import numpy as np
from machinedesign.layers import ksparse
from machinedesign.layers import winner_take_all_fc
from machinedesign.layers import winner_take_all_spatial
from machinedesign.layers import winner_take_all_channel
from machinedesign.layers import winner_take_all_lifetime
from machinedesign.layers import axis_softmax
from machinedesign.layers import SaltAndPepper
from machinedesign.layers import ZeroMasking

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
    assert np.all((y == 0).sum(axis=1) == 7)


def test_winner_take_all_fc():
    act = winner_take_all_fc(zero_ratio=0.7)
    X = K.placeholder(shape=(None, 10))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 10))
    y = pred([x])
    assert (y == 0).sum() == 7 * nb
    assert np.all((y == 0).sum(axis=0) == 0.7*nb)


def test_winner_take_all_lifetime():
    act = winner_take_all_lifetime(zero_ratio=0.7)
    X = K.placeholder(shape=(None, 10, 5, 5))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 10, 5, 5))
    y = pred([x])
    assert (y == 0).sum() == 7 * 5 * 5 * nb
    assert np.all((y == 0).sum(axis=(0, 2, 3)) == 0.7 * nb * 5 * 5)

def test_winner_take_all_kchannel():
    act = winner_take_all_lifetime(zero_ratio=0.7)
    X = K.placeholder(shape=(None, 10, 5, 5))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 10, 5, 5))
    y = pred([x])
    assert (y == 0).sum() == 7 * 5 * 5 * nb
    assert np.all((y == 0).sum(axis=(0, 2, 3)) == 0.7 * nb * 5 * 5)



def test_salt_and_pepper():
    proba = 0.3
    act = SaltAndPepper(proba)
    X = K.placeholder(shape=(None, 10))
    pred = K.function([X, K.learning_phase()], act.call(X))
    np.random.seed(42)
    nb = 10000
    x = np.random.uniform(0, 1, size=(nb, 50))
    y = pred([x, True])
    assert np.allclose((y == 1).mean(axis=0), 0.5 * proba, atol=1e-2, rtol=1e-2)
    assert np.allclose((y == 0).mean(axis=0), 0.5 * proba, atol=1e-2, rtol=1e-2)


def test_zero_masking():
    proba = 0.3
    act = ZeroMasking(proba)
    X = K.placeholder(shape=(None, 10))
    pred = K.function([X, K.learning_phase()], act.call(X))
    np.random.seed(42)
    nb = 10000
    x = np.random.uniform(0, 1, size=(nb, 50))
    y = pred([x, True])
    assert np.allclose((y == 0).mean(axis=0), proba, atol=1e-2, rtol=1e-2)
 

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
