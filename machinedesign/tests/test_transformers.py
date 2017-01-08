import numpy as np
import pytest

from machinedesign.transformers import Standardize


def test_standardize():
    np.random.seed(42)
    t = Standardize(axis=0)
    for _ in range(10000):
        g = np.random.normal(5, 3, size=(1000, 2))
        t.partial_fit(g)

    assert np.all(np.isclose(t.mean_, np.array([5., 5.]), atol=1e-3))
    assert np.all(np.isclose(t.std_, np.array([3., 3.]), atol=1e-3))

    X = np.random.uniform(0, 1, size=(100, 2))
    assert np.all(np.isclose(t.transform(X), (X - 5.) / 3., atol=1e-3))

    X = np.random.uniform(0, 1, size=(100, 2))
    assert np.all(np.isclose(t.inverse_transform(X), X * 3. + 5, atol=1e-3))

    assert np.all(np.isclose(X, t.inverse_transform(t.transform(X)), atol=1e-3))

    with pytest.raises(AssertionError):
        t = Standardize(axis=0)
        x = np.random.uniform(size=(10, 2))
        t.transform(x)

    with pytest.raises(AssertionError):
        t = Standardize(axis=0)
        x = np.random.uniform(size=(10, 2))
        t.inverse_transform(x)
