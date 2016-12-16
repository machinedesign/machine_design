"""
This module contains some common functions used in models
"""
import os
import numpy as np

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Layer
from keras import optimizers

__all__ = [
    "ksparse",
    "custom_objects",
    "activation_function",
    "fully_connected_layers",
    "get_optimizer",
    "build_optimizer",
    "object_to_dict",
    "mkdir_path",
    "minibatcher",
    "iterate_minibatches",
    "WrongModelFamilyException",
    "check_family_or_exception"
]


class ksparse(Layer):

    #TODO make it compatible with tensorflow (only works with theano)
    """
    For each example, sort activations, keep only a proportion of 1-zero_ratio from the biggest activations,
    that rest is zeroed out (a proportion of zero_ratio is zeroed out).
    Only for fully connected layers.
    Corresponds to ksparse autoencoders in [1].

    References
    ----------

    [1] Makhzani, A., & Frey, B. (2013). k-Sparse Autoencoders. arXiv preprint arXiv:1312.5663.

    """
    def __init__(self, zero_ratio=0,  **kwargs):
        super(ksparse, self).__init__(**kwargs)
        self.zero_ratio = zero_ratio

    def call(self, X, mask=None):
        import theano.tensor as T
        idx = T.cast((1 - self.zero_ratio) * X.shape[1], 'int32')
        theta = X[T.arange(X.shape[0]), T.argsort(X, axis=1)[:, idx]]
        mask = X > theta[:, None]
        return X * mask

    def get_config(self):
        config = {'zero_ratio': self.zero_ratio}
        base_config = super(ksparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# use this whenever you use load_model of keras load_model(..., custom_objects=custom_objects)
# to take into account the new defined layers when loading
custom_objects = {
    'ksparse': ksparse
}

def activation_function(name):
    if isinstance(name, dict):
        act = name
        name, params = act['name'], act['params']
        if name == 'ksparse':
            return ksparse(**params)
        else:
            raise ValueError('Unknown activation function : {}'.format(name))
    else:
        return Activation(name)

def fully_connected_layers(x, nb_hidden_units, activations, init='glorot_uniform'):
    """
    Apply a stack of fully connected layers to a layer `x`

    Parameters
    ----------

    x : layer
        keras layer
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
