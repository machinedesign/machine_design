"""
this module contain commonly used objective functions and some
helpers. the difference with the module `metrics` is that
the functions here are used as objective functions (loss), rather
than metrics, so they are written in terms of keras backend (symbolic),
rather than numpy.
"""
import warnings
from functools import partial

import numpy as np

from keras import backend as K
from keras import losses as keras_objectives
from keras.models import load_model
from keras.models import Model

from .layers import ReverseColorChannel
from .layers import Normalize
from .data import floatX
from .utils import object_to_dict
from .utils import get_axis


def dummy(y_true, y_pred):
    """
    dummy loss outputing a scalar of zero. it is used to
    specify a dummy loss when a keras model is loaded and the
    original loss function  can't be loaded with keras load_model
    """
    return (y_pred * 0).mean()


def squared_error(y_true, y_pred):
    y_true = y_true.flatten(2)
    y_pred = y_pred.flatten(2)
    return K.sum(K.square(y_pred - y_true), axis=1)


def mean_squared_error(y_true, y_pred):
    """mean squared error (mean over all axes except the first)"""
    y_true = y_true.flatten(2)
    y_pred = y_pred.flatten(2)
    return K.mean(K.square(y_pred - y_true), axis=1)


def categorical_crossentropy(y_true, y_pred):
    """categorical crossentropy assuming the last axis is the axis where categories reside"""
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    return -K.log(y_pred[K.arange(0, y_true.shape[0]), y_true]).mean()


def feature_space_mean_squared_error(y_true, y_pred, model_filename=None, layer=None, bias=None, scale=None, reverse_color_channel=False):
    """
    mean squared error on a feature space defined by a model
    usually those models need some preprocessing. This is why bias, scale, and reverse_color_channel are for.

    Parameters
    ----------

    model_filename : str
        filename of the model to load (should be a .h5 keras model)
    layer : str
        layer to use. raises an exception when the layer does not exist.
    bias : None or list
        list of real numbers with as many elements as number of image channels (e.g. usually 3).
        it is used to transform the input to `input * scale + bias` on the channel axis.
    scale : None or list
        list of real numbers with as many elements as number of image channels (e.g. usually 3).
        it is used to transform the input to `input * scale + bias` on the channel axis.
    reverse_color_channel : bool
        if True, do X = X[:, ::-1, :, :] before normalizing with bias and scale.
        Used in imagenet models.
    """
    if model_filename is None or layer is None:
        warnings.warn('In case you are willing to train this model, please specify `model_filename` and `layer` in the parameters of the loss.'
                      'In case you will just use the model, it is fine. When loading a model with keras through `load_model` '
                      'the parameters of the loss functions do not get passsed, so if you are just willing to use the model'
                      'it is fine', RuntimeWarning)
        return dummy(y_true, y_pred)
    model = load_model(model_filename)
    layer_names = set([lay.name for lay in model.layers])
    if layer not in layer_names:
        raise ValueError(
            'layer {} does not exist, available layers are : {}'.format(layer, layer_names))
    model_layer = Model(inputs=model.layers[0].input, outputs=model.get_layer(layer).output)
    model_layer.trainable = False
    transforms = []
    if reverse_color_channel:
        transforms.append(ReverseColorChannel())
    if bias and scale:
        nb_channels = model_layer.input_shape[1]
        bias = floatX(bias)
        scale = floatX(scale)
        assert len(bias.shape) == 1 and bias.shape[0] == nb_channels, 'bias should have {} elements'.format(nb_channels)
        assert len(scale.shape) == 1 and scale.shape[0] == nb_channels, 'scale should have {} elements'.format(nb_channels)
        bias = bias[np.newaxis, :, np.newaxis, np.newaxis]
        scale = scale[np.newaxis, :, np.newaxis, np.newaxis]
        norm = Normalize(bias=bias, scale=scale)
        transforms.append(norm)
    transforms.append(model_layer)
    for tf in transforms:
        y_true = tf(y_true)
        y_pred = tf(y_pred)
    return mean_squared_error(y_true, y_pred).mean()


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
    yt = y_true.argmax(axis=get_axis(axis))  # supposed to be onehot in the axis 'axis'
    yt = yt.flatten()  # convert it to a vector
    perm = list(range(y_pred.ndim))
    # permute 'axis' and the first axis
    perm[axis], perm[0] = perm[0], perm[axis]
    perm = tuple(perm)
    ypr = y_pred.transpose(perm)
    ypr = ypr.reshape((ypr.shape[0], -1))
    ypr = ypr.T
    return K.categorical_crossentropy(yt, ypr).mean()


def objectness(y_true, y_pred, model_filename=None):
    """
    Compute the objectness [1] score based on a classifier in `model_filename`.
    this loss does not use `y_true`, it only uses the classifier and `y_pred`.

    Parameters
    ----------

    model_filename : str
        filename of the model to load (should be a .h5 keras model)

    References
    ----------

    [1] Improved techniques for training gans
        T.Salimans, I.Goodfellow, W.Zaremba, V.Cheung, A.Radford, X.Chen
        Advances in Neural Information Processing Systems, 2226-2234
    """
    if model_filename is None:
        warnings.warn('In case you are willing to train this model, please specify `model_filename` in the parameters of the loss.'
                      'In case you will just use the model, it is fine. When loading a model with keras through `load_model` '
                      'the parameters of the loss functions do not get passsed, so if you are just willing to use the model'
                      'it is fine', RuntimeWarning)
        return dummy(y_true, y_pred)

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
    if not len(terms):
        warnings.warn('In case you are willing to train this model, please specify `terms` in the parameters of the loss.'
                      'In case you will just use the model, it is fine. When loading a model with keras through `load_model` '
                      'the parameters of the loss functions do not get passsed, so if you are just willing to use the model'
                      'it is fine', RuntimeWarning)
        return dummy(y_true, y_pred)
    total = 0
    for loss_def in terms:
        total += get_loss(loss_def['loss'])(y_true, y_pred) * loss_def['coef']
    return total


objectives = {
    'squared_error': squared_error,
    'axis_categorical_crossentropy': axis_categorical_crossentropy,
    'feature_space_mean_squared_error': feature_space_mean_squared_error,
    'objectness': objectness,
    'sum': loss_sum,
    'loss_sum': loss_sum,
    # 'loss_aggregate' is used to multi_interface as evaluator loss.
    #  it is set to dummy when calling keras load_model
    'loss_aggregate': dummy,
    'categorical_crossentropy': categorical_crossentropy
}

objectives_ = object_to_dict(keras_objectives)
objectives.update(objectives_)


def get_loss(loss, objectives=objectives):
    """
    get loss function given a `loss`.
    `loss` can either be a string, and in that case, it will correspond
    the name of the loss. In case the loss has itself parameters, a dict
    can be passed instead of a str, thus `loss` can be a dict.
    If `loss` is a dict, then it should have a key `name` and a key `params`.
    On both cases (`loss` is a str or dict), if the name is one of the keys
    defined in `objectives`, then use it, if not, raise an exception.

    Parameters
    ----------

    loss : str or dict
        loss to use, see the description above to know when to use
        str and when to use dict.
    objectives : dict
        dict containing available objective functions, keys are
        objective names, values are objective functions.
        By default, `objectives` contain all keras objectives with additional
        custom objectives defined in the objectives module.
    """
    if isinstance(loss, dict):
        name = loss['name']
        params = loss.get('params', {})
        func = objectives[name]
        orig_name = func.__name__
        func = partial(func, **params)
        func.__name__ = orig_name
        # keras requests and uses the __name__ of the func (for serialization), this
        # is why I do this.
        return func
    # assume loss is a str
    else:
        func = objectives[loss]
        return func
