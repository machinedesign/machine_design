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
    "custom_objectives",
    "axis_categorical_crossentropy"
]

def mean_squared_error(y_true, y_pred):
    """mean squared error (mean over all axes)"""
    return K.mean(K.square(y_pred - y_true))

def axis_categorical_crossentropy(y_true, y_pred, axis=1):
    yt = y_true.argmax(axis=axis) # supposed to be onehot in the axis 'axis'
    yt = yt.flatten()#convert it to a vector
    perm = list(range(y_pred.ndim))
    # permute 'axis' and the first axis
    perm[axis], perm[0] = perm[0], perm[axis]
    perm = tuple(perm)
    ypr = y_pred.transpose(perm)
    ypr = ypr.reshape((ypr.shape[0], -1))
    ypr = ypr.T
    return K.categorical_crossentropy(ypr, yt)

custom_objectives = {
    'axis_categorical_crossentropy': axis_categorical_crossentropy
}

def get_loss(name, objectives=objectives):
    if isinstance(name, dict):
        #TODO do that
        raise NotImplementedError()
    else:
        try:
            func = custom_objectives[name]
        except KeyError:
            return getattr(objectives, name)
        else:
            return func
