"""
This module contains some common functions used in models
"""
from __future__ import division
import numpy as np

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Layer
from keras.layers import Convolution2D
from keras.layers import GaussianNoise
from keras.engine.training import Model
from keras import optimizers

from .objectives import custom_objectives
from .layers import custom_layers

__all__ = [
    "activation_function",
    "fully_connected_layers",
    "build_optimizer",
    "WrongModelFamilyException",
    "check_family_or_exception"
]

custom_objects = {}
custom_objects.update(custom_objectives)
custom_objects.update(custom_layers)

def activation_function(name):
    if isinstance(name, dict):
        act = name
        name, params = act['name'], act['params']
        if name in custom_layers:
            return custom_layers[name](**params)
        else:
            raise ValueError('Unknown activation function : {}'.format(name))
    else:
        return Activation(name)

def noise(x, name, params):
    if name == 'gaussian':
        std = params['std']
        return GaussianNoise(std)(x)
    elif name == 'none':
        return x
    else:
        raise ValueError('Unknown noise function')

def fully_connected_layers(x, nb_hidden_units, activations, init='glorot_uniform'):
    """
    Apply a stack of fully connected layers to a layer `x`

    Parameters
    ----------

    x : keras layer
    nb_hidden_units : list of int
        number of hidden units
    activations : str
        list of activation functions for each layer
        (should be the same size than nb_hidden_units)

    Returns
    -------

    keras layer
    """
    assert len(activations) == len(nb_hidden_units)
    for nb_hidden, act in zip(nb_hidden_units, activations):
        x = Dense(nb_hidden, init=init)(x)
        x = activation_function(act)(x)
    return x

def conv2d_layers(x, nb_filters, filter_sizes, activations,
                  init='glorot_uniform', border_mode='valid',
                  stride=1, conv_layer=Convolution2D):
    """
    Apply a stack of 2D convolutions to a layer `x`

    Parameters
    ----------

    x : keras layer
    nb_filters : list of int
        nb of filters/feature_maps per layer
    filter_sizes : list of int
        size of (square) filters per layer
    activations : str
        list of activation functions for each layer
        (should be the same size than nb_hidden_units)
    init : str
        init method used in all layers
    border_mode : str
        padding type to use in all layers
    stride : int
        stride to use
    conv_layer : keras layer class
        keras layer to use from convolution

    Returns
    -------

    keras layer
    """
    assert len(nb_filters) == len(filter_sizes) == len(activations)
    for nb_filter, filter_size, act in zip(nb_filters, filter_sizes, activations):
        x = conv_layer(nb_filter, filter_size, filter_size, init=init, border_mode=border_mode, subsample=(stride, stride))(x)
        x = activation_function(act)(x)
    return x



def build_optimizer(algo_name, algo_params):
    """
    build a keras optimizer instance from its name and params

    Parameters
    ----------
        algo_name: str
            name of the optimizer
        algo_params: dict
            parameters of the optimizer
    """
    optimizer = _get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)
    return optimizer

def _get_optimizer(name):
    """Get a keras optimizer class from its name"""
    if hasattr(optimizers, name):
        return getattr(optimizers, name)
    else:
        raise Exception('unknown optimizer : {}'.format(name))

class WrongModelFamilyException(ValueError):
    """
    raised when the model family is not the expected one
    model families are kinds of models different enough in
    their training pipeline that they need to be separated:
    e.g GAN and autoencoders are distinct families.
    """
    pass

def check_family_or_exception(family, expected):
    """if family is not equal to expected, raise WrongModelFamilyException"""
    if family != expected:
        raise WrongModelFamilyException("expected family to be '{}', got {}".format(expected, family))

def show_model_info(model, print_func=print):
    print_func('Input shape : {}'.format(model.input_shape))
    print_func('Output shape : {}'.format(model.output_shape))
    print_func('Number of parameters : {}'.format(model.count_params()))

    layers = list(_get_layers(model))
    nb = sum(1 for layer in layers if hasattr(layer, 'W') and layer.trainable)
    nb_W_params = sum(np.prod(layer.W.get_value().shape) for layer in layers if hasattr(layer, 'W') and layer.trainable)
    print_func('Number of weight parameters : {}'.format(nb_W_params))
    print_func('Number of learnable layers : {}'.format(nb))

def _get_layers(model):
    for layer in model.layers:
        if isinstance(layer, Model):
            for l in _get_layers(layer):
                yield l
        elif isinstance(layer, Layer):
            yield layer

def check_model_shape_or_exception(model, shape):
    if model.output_shape[1:] != shape:
        msg = """Wrong output shape of the model, expected : {}, got : {}.
                 Please fix the parameters""".format(shape, model.output_shape[1:])
        raise ValueError(msg)
