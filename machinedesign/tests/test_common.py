import numpy as np
from machinedesign.common import minibatcher
from machinedesign.common import iterate_minibatches
from machinedesign.common import ksparse

import keras.backend as K

def test_minibatcher():
    func = lambda x:x**2
    func = minibatcher(func, batch_size=10)

    # test empty input
    y = func([])
    assert len(y) == 0

    # test when nb of elements is not
    # prop. to batch_size
    X = np.arange(0, 51)
    func = minibatcher(func, batch_size=10)
    y = func(X)
    assert len(y) == len(X)
    assert np.all(y == func(X))

def test_iterate_minibatches():
    assert list(iterate_minibatches(0, 10)) == []

    mb = iterate_minibatches(55, 10)
    assert list(mb) == [slice(0, 10), slice(10, 20), slice(20, 30), slice(30, 40), slice(40, 50), slice(50, 55)]

def test_k_sparse():
    act = ksparse(0.3)
    X = K.placeholder(shape=(None, 10))
    pred = K.function([X], act.call(X))
    np.random.seed(42)
    nb = 100
    x = np.random.uniform(-1, 1, size=(nb, 10))
    y = pred([x])
    assert (y==0).sum() == 7 * nb
