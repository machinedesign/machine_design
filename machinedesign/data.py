"""
This module contains common functions for data processing.
"""
from __future__ import division
import os
from functools import partial
from itertools import cycle
from six.moves import map
from six.moves import range

import numpy as np
import h5py

from datakit.pipeline import pipeline_load
from datakit.image import operators as image_operators
from datakit.helpers import minibatch
from datakit.helpers import expand_dict
from datakit.helpers import dict_apply

from keras.models import Model

def _pipeline_load_numpy(iterator, filename,
                         cols=['X', 'y'],
                         start=0, nb=None, shuffle=False,
                         random_state=None):
    """
    Operator to load npy or npz files

    Parameters
    ----------

    filename : str
        filename to load
    cols : list of str
        columns to retrieve from the npy file
    start : int(default=0)
        starting index of the data
    nb : int(default=None)
        the size of the data to read.
        if None, take everything starting
        from start.
    shuffle : bool(default=False)
        whether to shuffle the data
    random_state : int(default=None)
    """
    rng = np.random.RandomState(random_state)
    filename = os.path.join(os.getenv('DATA_PATH'), filename)
    data = np.load(filename)
    if shuffle:
        indices = np.arange(len(data[cols[0]]))
        rng.shuffle(indices)
        data_shuffled = {}
        for c in cols:
            data_shuffled[c] = data[c][indices]
        data = data_shuffled
    return _iterate(data, start=start, nb=nb, cols=cols)

def _pipeline_load_hdf5(iterator, filename,
                        cols=['X', 'y'],
                        start=0, nb=None, buffer_size=128):
    """
    Operator to load hdf5 files

    Paramters
    ---------

    filename : str
        filename to load
    cols : list of str
        columns to retrieve from the npy file
    start : int(default=0)
        starting index of the data
    nb : int(default=None)
        the size of the data to read.
        if None, take everything starting
        from start.
    buffer_size : int(default=128)
        read buffer_size rows each time from the file
    random_state : int(default=None)

    """
    filename = os.path.join(os.getenv('DATA_PATH'), filename)
    hf = h5py.File(filename)

    def iter_func():
        for i in range(start, start + nb, buffer_size):
            d = {}
            for c in cols:
                d[c] = hf[c][i:i+buffer_size]
            for n in range(len(d[cols[0]])):
                p = {}
                for c in cols:
                    p[c] = d[c][n]
                yield p
    return iter_func()

def _iterate(data, start=0, nb=None, cols=['X', 'y']):
    it = {}
    for c in cols:
        d = data[c]
        if nb:
            d = d[start:start+nb]
        else:
            d = d[start:]
        it[c] = iter(d)
    def iter_func():
        while True:
            d = {}
            for c in cols:
                d[c] = next(it[c])
            yield d
    return iter_func()

_pretrained = {}
def _pipeline_pretrained_transform(iterator, model_name='inceptionv3',
                                   layer=None, include_top=False,
                                   input_col='X', output_col='h'):
    # This function assumes the pixels are between 0 and 1
    def _transform(data):
        assert layer is not None, 'expected layer to not be None, please specify it'
        X = data[input_col]
        if model_name in _pretrained:
            model = _pretrained[model_name]
        else:
            if model_name == 'inceptionv3':
                from keras.applications import InceptionV3
                model = InceptionV3(input_shape=X.shape, weights='imagenet', include_top=include_top)
                X = X * 2 - 1
            elif model_name == 'alexnet':
                from convnetskeras.convnets import convnet
                assert X.shape == (3, 227, 227), 'for "alexnet" shape should be (3, 227, 227), got : {}'.format(X.shape)
                weights_path = "{}/.keras/models/alexnet_weights.h5".format(os.getenv('HOME'))
                assert os.path.exists(weights_path), ('weights path of alexnet {} does not exist, please download manually'
                                                     'from http://files.heuritech.com/weights/alexnet_weights.h5 and put it there '
                                                     '(see https://github.com/heuritech/convnets-keras)'.format(weights_path))
                model = convnet('alexnet',weights_path=weights_path, heatmap=False)
                X *= 255.
                X[0, :, :] -= 123.68
                X[1, :, :] -= 116.779
                X[2, :, :] -= 103.939
            else:
                raise ValueError('expected name to be "inceptionv3" or "alexnet", got : {}'.format(model_name))
            names = [layer.name for layer in model.layers]
            assert layer in names, 'layer "{}" does not exist, available : {}'.format(layer, names)
            model = Model(input=model.layers[0].input, output=model.get_layer(layer).output)
            _pretrained[model_name] = model
        X = X[np.newaxis, :, :, :]
        h = model.predict(X)
        h = h[0]
        data[output_col] = h
        return data
    iterator = map(_transform, iterator)
    return iterator

load_operators = {
    'load_numpy': _pipeline_load_numpy,
    'load_hdf5': _pipeline_load_hdf5
}

transform_operators = {
    'pretrained_transform': _pipeline_pretrained_transform
}
operators = {}
operators.update(image_operators)
operators.update(load_operators)
operators.update(transform_operators)
pipeline_load = partial(pipeline_load, operators=operators)

def get_nb_samples(pipeline):
    """
    get nb of samples of a pipeline

    Parameters
    ----------

    pipeline : list of dict

    Returns
    -------

    int
    """
    if len(pipeline) == 0:
        return 0
    p = pipeline[0:1]
    p = pipeline_load(p)
    p = list(p)
    return len(p)

def get_shapes(sample):
    """
    get the shapes of a sample with modalities
    it returns a dict where the keys are the modalities (e.g `X`, `y`)
    and the values are the shapes (exluding the nb_examples dimension).

    Parameters
    ----------

    sample : dict

    Returns
    -------

    dict
    """
    return {k: v.shape[1:] for k, v in sample.items()}

def get_nb_minibatches(nb_samples, batch_size):
    """
    get nb of minibatches corresponding to a dataset of size `nb_samples`
    and a minibatch of size `batch_size`.
    """
    if batch_size == 0:
        return 0
    return (nb_samples // batch_size) + (nb_samples % batch_size > 0)

def batch_iterator(iterator, batch_size=128, repeat=True, cols=['X', 'y']):
    """
    returns a version of iterator with minibatches

    Parameters
    ----------

    iterator : iterable of dict
        dataset
    batch_size : int(default=128)
        size of minibatches
    repeat: bool
        if True, `itertools.cycle` is applied to the resulting `iterator` so that
        it repeats.
    cols: list of str or str
        if it is list of str, columns to use from the dicts.
        if it is str and cols=='all', use all columns.
        np.array is applied to those columns to convert
        them into a numpy array.
    Returns
    -------

    iterator of dicts.
    The keys of the dict are `cols` (e.g `X`, `y`).
    The number of examples for the values in the dict are at max `batch_size` (can be less).

    """
    if not isinstance(cols, list):
        if cols == 'all':
            cols = None
        else:
            raise ValueError('Expected cols to be either a list or the str "all", got : {}'.format(cols))

    iterator = minibatch(iterator, batch_size=batch_size)
    iterator = expand_dict(iterator)
    iterator = map(partial(dict_apply, fn=floatX, cols=cols), iterator)
    if cols:
        iterator = map(lambda data: {c: data[c] for c in cols}, iterator)
    if repeat:
        iterator = cycle(iterator)
    return iterator

def floatX(X):
    #TODO should depend on theano.config.floatX
    # for tensorflow, what is the equivalent thing?
    return np.array(X).astype('float32')

def intX(X):
    return X.astype(np.int32)

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
