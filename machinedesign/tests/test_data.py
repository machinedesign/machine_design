import numpy as np
from six.moves import map

from machinedesign.data import get_nb_samples
from machinedesign.data import get_shapes
from machinedesign.data import get_nb_minibatches
from machinedesign.data import minibatcher
from machinedesign.data import iterate_minibatches
from machinedesign.data import pipeline_load

toy_pipeline = [
    {"name": "toy", "params": {"nb": 50, "w": 8, "h": 8,
                               "pw": 2, "ph": 2, "nb_patches": 2, "random_state": 42}},
    {"name": "shuffle", "params": {"random_state": 42}},
    {"name": "normalize_shape", "params": {}},
    {"name": "divide_by", "params": {"value": 255}},
    {"name": "order", "params": {"order": "th"}}
]


def test_get_nb_samples():
    data = pipeline_load(toy_pipeline)
    data = map(lambda d: d['X'], data)
    assert get_nb_samples(data) == 50
    assert get_nb_samples([]) == 0


def test_get_shapes():
    assert get_shapes({}) == {}
    assert get_shapes({'X': np.random.uniform(size=(1, 2, 3))}) == {'X': (2, 3)}
    assert get_shapes({'X': np.random.uniform(size=(1, 2, 3)), 'y': np.random.uniform(size=(4, 5))}) == {
        'X': (2, 3), 'y': (5,)}


def test_get_nb_minibatches():
    assert get_nb_minibatches(0, 10) == 0
    assert get_nb_minibatches(0, 1) == 0
    assert get_nb_minibatches(0, 0) == 0
    assert get_nb_minibatches(10, 0) == 0
    assert get_nb_minibatches(25, 1) == 25
    assert get_nb_minibatches(25, 10) == 3


def test_minibatcher():
    def func(x):
        return x**2
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
    assert list(mb) == [slice(0, 10), slice(10, 20), slice(
        20, 30), slice(30, 40), slice(40, 50), slice(50, 55)]
