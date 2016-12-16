"""
this module contain commonly used objective functions and some
helpers. the difference with the module `metrics` is that
the functions here are used as objective functions (loss), rather
than metrics, so they are written in terms of keras backend (symbolic),
rather than numpy.
"""
from keras import backend as K
from keras import objectives

__all__ = [
    "mean_squared_error",
    "get_loss",
    "custom_objectives"
]

def mean_squared_error(y_true, y_pred):
    """mean squared error (mean over all axes)"""
    return K.mean(K.square(y_pred - y_true))

custom_objectives = {
    'mean_squared_error': mean_squared_error
}

def get_loss(name, objectives=objectives):
    try:
        func = custom_objectives[name]
    except KeyError:
        return getattr(objectives, name)
    else:
        return func
