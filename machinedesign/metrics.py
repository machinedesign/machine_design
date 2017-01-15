"""
This module contains a list of metrics that are
common to models.
"""
from functools import partial
import numpy as np


def mean_squared_error(y_true, y_pred):
    """mean squared error (mean over all axes except the first)"""
    axes = tuple(range(1, len(y_true.shape)))
    score = ((y_true - y_pred)**2).mean(axis=axes)
    return score


def binary_crossentropy(y_true, y_pred, eps=1e-5):
    """ binary cross entropy (mean over all axes except the first)"""
    axes = tuple(range(1, len(y_true.shape)))
    y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid nans, like done in keras
    score = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return score.mean(axis=axes)


def categorical_crossentropy(y_true, y_pred):
    """
    categorical crossentropy that can work for any tensors
    y_true and y_pred should have the same shape.
    y_true and y_pred are first squeezed to be a matrix
    e.g if they are (nb_examples * timesteps, features) then they
    are converted to (nb_examples * timesteps, features).
    After that, the classical categorical crossentropy is applied on the matrices.
    """
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    return -np.log(y_pred[np.arange(len(y_pred)), y_true])


metrics = {
    'mean_squared_error': mean_squared_error,
    'binary_crossentropy': binary_crossentropy,
    'categorical_crossentropy': categorical_crossentropy
}


def get_metric(metric, metrics=metrics):
    if isinstance(metric, dict):
        name = metric['name']
        params = metric.get('params', {})
        func = metrics[name]
        orig_name = func.__name__
        func = partial(func, **params)
        func.__name__ = orig_name
        # keras requests and uses the __name__ of the func (for serialization), this
        # is why I do this.
        return func
    # assume metric is a str
    else:
        func = metrics[metric]
        return func


def compute_metric(get_true_and_pred, metric):
    """
    compute a metric using an iterator and true and predicted values

    Parameters
    ----------

    get_true_and_pred: callable
        should return an iterator of tuple of (true, pred) values
    metric: callable
        callable that take the true and predicted values and returns a score
        available metrics : `mean_squared_error`

    Returns
    -------

    numpy array
        the metric value for each individual element.
    """
    vals = []
    for real, pred in get_true_and_pred():
        vals.append(metric(real, pred))
    if len(vals) == 0:
        return []
    vals = np.concatenate(vals, axis=0)
    return vals
