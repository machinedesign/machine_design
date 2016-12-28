"""
this module contain commonly used objective functions and some
helpers. the difference with the module `metrics` is that
the functions here are used as objective functions (loss), rather
than metrics, so they are written in terms of keras backend (symbolic),
rather than numpy.
"""
from functools import partial

from keras import backend as K
from keras import objectives
from keras.models import load_model
from keras.models import Model

__all__ = [
    "mean_squared_error",
    "get_loss",
    "custom_objectives",
    "axis_categorical_crossentropy"
]

def mean_squared_error(y_true, y_pred):
    """mean squared error (mean over all axes except the first)"""
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

    vector
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

def feature_space_mean_squared_error(y_true, y_pred, model_filename=None, layer=None):
    """mean squared error on a feature space defined by a model"""
    assert model_filename is not None, 'Please specify `model_filename` in the parameters of the loss'
    assert layer is not None, 'Please specifiy `layer` in the parameters of the loss'
    model = load_model(model_filename)
    model_layer = Model(input=model.layers[0].input, output=model.get_layer(layer).output)
    model_layer.trainable = False
    return mean_squared_error(model_layer(y_true), model_layer(y_pred))

def objectness(y_true, y_pred, model_filename=None):
    """
    Compute the objectness [1] score based on a classifier in `model_filename`.
    this loss does not use `y_true`, it only uses the classifier and `y_pred`.

    References
    ----------

    [1] Improved techniques for training gans
        T.Salimans, I.Goodfellow, W.Zaremba, V.Cheung, A.Radford, X.Chen
        Advances in Neural Information Processing Systems, 2226-2234
    """
    assert model_filename is not None, 'Please specify `model_filename` in the parameters of the loss'
    model = load_model(model_filename)
    model.trainable = False
    probas = model(y_pred)
    score = _compute_objectness(probas)
    return score

def _compute_objectness(probas):
    pr = probas
    marginal = pr.mean(axis=0, keepdims=True)
    score = pr * K.log(pr / marginal)
    score = score.sum(axis=1)
    return score.mean()

def loss_sum(y_true, y_pred, terms=[]):
    """
    a loss which is a weighted sum of several losses

    parameters
    ----------

    terms: list of dict
        each dict should have a key 'loss' and a key 'coef'.
        'loss' can be either a dict or a str. It defines the loss.
        See `get_loss` to know when to use a str and when to use a dict.
        'coef' is a float defining the importance given to the loss.
    """
    assert len(terms), 'Please specify `terms` in the parameters of the loss'
    total = 0
    for loss_def in terms:
        total += get_loss(loss_def['loss'])(y_true, y_pred) * loss_def['coef']
    return total

custom_objectives = {
    'axis_categorical_crossentropy': axis_categorical_crossentropy,
    'feature_space_mean_squared_error': feature_space_mean_squared_error,
    'objectness': objectness,
    'sum': loss_sum
}

def get_loss(loss, objectives=objectives):
    """
    get loss function given a `loss`.
    `loss` can either be a string, and in that case, it will correspond
    the name of the loss. In case the loss has itself parameters, a dict
    can be passed instead of a str, thus `loss` can be a dict.
    If `loss` is a dict, then it should have a key `name` and a keys `params`.
    On both cases, if the name is one of the keys defined in `custom_objectives`,
    then use it. if not, search in `objectives`.
    if it does not find in either of these, it will return an exception.

    Parameters
    ----------

    loss : str or dict
        loss to use, see the description above to know when to use
        str and when to use dict.
    objectives : object
        object (e.g module) containing objectives (by default it is `keras.objectives`)
    """
    if isinstance(loss, dict):
        name = loss['name']
        params = loss['params']
        try:
            func = custom_objectives[name]
        except KeyError:
            func = getattr(objectives, name)
        orig_name = func.__name__
        func = partial(func, **params)
        func.__name__ = orig_name
        # keras requests the __name__ of the func, this
        # is why I do this.
        return func
    # assume loss is a str
    else:
        try:
            func = custom_objectives[loss]
        except KeyError:
            return getattr(objectives, loss)
        else:
            return func
