"""
This module contains common functions for data processing
"""
from __future__ import division
from functools import partial
from itertools import cycle
try:
    from itertools import imap
except ImportError:
    imap = map

import numpy as np

from datakit.pipeline import pipeline_load
from datakit.image import operators as image_operators
from datakit.helpers import minibatch
from datakit.helpers import expand_dict
from datakit.helpers import dict_apply

__all__ = [
    "get_nb_samples",
    "get_shapes",
    "get_nb_minibatches",
    "batch_iterator",
    "floatX",
    "intX"
]

operators = image_operators
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
    cols: list of str
        columns to use from the dicts.
        np.array is applied to those columns to convert
        them into a numpy array.
    Returns
    -------

    iterator of dicts.
    The keys of the dict are `cols` (e.g `X`, `y`).
    The number of examples for the values in the dict are at max `batch_size` (can be less).

    """
    iterator = minibatch(iterator, batch_size=batch_size)
    iterator = expand_dict(iterator)
    iterator = imap(partial(dict_apply, fn=floatX, cols=cols), iterator)
    iterator = imap(lambda data: {c: data[c] for c in cols}, iterator)
    if repeat:
        iterator = cycle(iterator)
    return iterator

def floatX(X):
    return np.array(X).astype('float32')

def intX(X):
    return X.astype(np.int32)
