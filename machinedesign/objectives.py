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
    y_true = y_true.flatten(2)
    y_pred = y_pred.flatten(2)
    return K.mean(K.square(y_pred - y_true), axis=1)

def axis_categorical_crossentropy(y_true, y_pred, axis=1):
    """
    categorical crossentropy where y_pred can be tensor of any dimension.
    keras categorical crossentropy only supports y_pred as 2D tensor.

    Paramaters
    ----------

    y_true : int tensor of order D - 1
        true labels
        e.g can be (nb_examples, height width)
    y_pred : float tensor of order D
        predicted probabilities.
        e.g can be (nb_examples, nb_channels, height, widths)
    axis : int(default=1)
        axis where the probabilities of categories are defined

    Returns
    -------

    scalar
    """
    yt = y_true.argmax(axis=axis) # supposed to be onehot in the axis 'axis'
    yt = yt.flatten()#convert it to a vector
    perm = list(range(y_pred.ndim))
    # permute 'axis' and the first axis
    perm[axis], perm[0] = perm[0], perm[axis]
    perm = tuple(perm)
    ypr = y_pred.transpose(perm)
    ypr = ypr.reshape((ypr.shape[0], -1))
    ypr = ypr.T
    return K.categorical_crossentropy(ypr, yt).mean()

custom_objectives = {
    'axis_categorical_crossentropy': axis_categorical_crossentropy
}

def get_loss(name, objectives=objectives):
    """
    get loss function given a `name`.
    If the name is one of the keys defined in `custom_objectives`,
    then use it. if not, search in `objectives`.
    if it does not find in either of these, it will return an exception.

    Parameters
    ----------

    name : str
        name of the loss function
    objectives : object
        object (e.g module) containing objectives (by default its `keras.objectives`)
    """
    if isinstance(name, dict):
        #TODO do that (handle params for loss function)
        raise NotImplementedError()
    else:
        try:
            func = custom_objectives[name]
        except KeyError:
            return getattr(objectives, name)
        else:
            return func
