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

from datakit.pipeline import pipeline_load
from datakit.image import operators as image_operators
from datakit.loaders import operators as load_operators
from datakit.helpers import minibatch_from_chunks

_pretrained = {}

transform_operators = {
}
operators = {}
operators.update(image_operators)
operators.update(load_operators)
operators.update(transform_operators)

# just pipeline_load but with custom operators defined here
pipeline_load = partial(pipeline_load, operators=operators)


def get_nb_samples(data_iter):
    """
    get nb of samples of a pipeline

    Parameters
    ----------

    data_iter : iterator of numpy array like objects or iterators

    Returns
    -------

    int
    """
    return sum(map(lambda data: len(data), data_iter))


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
        if True, `itertools.cycle` is applied to the resulting `iterator` so
        that it repeats.
    cols: list of str or str
        if it is list of str, columns to use from the dicts.
        if it is str and cols=='all', use all columns.
    Returns
    -------

    iterator of dicts.
    The keys of the dict are `cols` (e.g `X`, `y`).
    The number of examples for the values in the dict are at max `batch_size`
    (can be less).

    """
    if not isinstance(cols, list):
        if cols == 'all':
            cols = None
        else:
            raise ValueError(
                'Expected cols to be either a list or the str "all", got : {}'.format(cols))

    iterator = minibatch_from_chunks(iterator, batch_size=batch_size)
    if cols:
        iterator = map(lambda data: {c: data[c] for c in cols}, iterator)
    if repeat:
        iterator = cycle(iterator)
    return iterator


def floatX(X):
    # TODO should depend on theano.config.floatX
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
