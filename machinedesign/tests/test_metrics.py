import numpy as np

from machinedesign.metrics import compute_metric

def test_compute_metric():
    np.random.seed(42)
    X = np.random.uniform(size=(10,))
    Y = X + 1
    it = lambda: [(X, Y), (X+1, Y+1), (X+2, Y+2)]
    metric = lambda real, pred: real-pred

    res = compute_metric(it, metric)
    assert res.shape == (30,)
    assert np.allclose(res, -1)

    it = lambda: []
    res = compute_metric(it, metric)
    assert len(res) == 0
