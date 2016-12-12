import numpy as np
try:
    from itertools import izip
except  ImportError:
    izip = zip

def mean_squared_error(y_true, y_pred):
    axes = tuple(range(1, len(y_true.shape)))
    score = ((y_true - y_pred)**2).mean(axis=axes)
    return score

def compute_metric(real, pred, metric):
    vals = []
    for pred, real in izip(pred, real):
        vals.append(metric(real, pred))
    vals = np.concatenate(vals, axis=0)
    return vals
