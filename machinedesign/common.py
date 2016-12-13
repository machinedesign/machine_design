"""
This module contains some common functions used in models
"""
import os

from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers

__all__ =[
    "activation_function",
    "fully_connected_layers",
    "get_optimizer",
    "build_optimizer",
    "object_to_dict",
    "mkdir_path"
]

def activation_function(name):
    return Activation(name)

def fully_connected_layers(x, nb_hidden_units, activation, init='glorot_uniform'):
    """
    Apply a stack of fully connected layers to a layer `x`

    Parameters
    ----------

    x : layer
        keras layer
    nb_hidden_units : list of int
        number of hidden units
    activation : str
        activation function, obtained using `activation_function`
    """
    for nb_hidden in nb_hidden_units:
        x = Dense(nb_hidden, init=init)(x)
        x = activation_function(activation)(x)
    return x

def get_optimizer(name):
    """Get a keras optimizer class from its name"""
    if hasattr(optimizers, name):
        return getattr(optimizers, name)
    else:
        raise Exception('unknown optimizer : {}'.format(name))

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
    optimizer = get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)
    return optimizer

def object_to_dict(obj):
    """return the attributes of an object"""
    return obj.__dict__

def mkdir_path(path):
    """
    Create folder in `path` silently: if it exists, ignore, if not
    create all necessary folders reaching `path`
    """
    if not os.access(path, os.F_OK):
        os.makedirs(path)
