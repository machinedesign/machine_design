"""
This module contains some common functions used in models
"""
import os

import numpy as np

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


def minibatcher(func, batch_size=1000):
  """
  Decorator to apply a function minibatch wise to avoid memory
  problems.

  Paramters
  ---------
  func : a function that takes an input and returns an output
  batch_size : int
    size of each minibatch

  iterate through all the minibatches, call func, get the results,
  then concatenate all the results.
  """
  def f(X):
      results = []
      for sl in iterate_minibatches(len(X), batch_size):
          results.append(func(X[sl]))
      if len(results) == 0:
          return []
      else:
          return np.concatenate(results, axis=0)
  return f

def iterate_minibatches(nb_inputs, batch_size):
  """
  Get slices pointing to indices of example forming minibatches

  Paramaters
  ----------
  nb_inputs : int
    size of the data
  batch_size : int
    minibatch size

  Yields
  ------

  slice
  """
  for start_idx in range(0, nb_inputs, batch_size):
      end_idx = min(start_idx + batch_size, nb_inputs)
      excerpt = slice(start_idx, end_idx)
      yield excerpt

class WrongModelFamilyException(ValueError):
    pass
