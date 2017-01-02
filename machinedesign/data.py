"""
This module contains common functions for data processing
"""
from __future__ import division
import os
from functools import partial
from itertools import cycle
try:
    from itertools import imap
except ImportError:
    imap = map
try:
    xrange
except NameError:
    irange = range
else:
    irange = xrange

import numpy as np
import h5py

from datakit.pipeline import pipeline_load
from datakit.image import operators as image_operators
from datakit.helpers import minibatch
from datakit.helpers import expand_dict
from datakit.helpers import dict_apply

__all__ = [
    "pipeline_load",
    "get_nb_samples",
    "get_shapes",
    "get_nb_minibatches",
    "batch_iterator",
    "floatX",
    "intX"
]

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
        for i in irange(start, start + nb, buffer_size):
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

load_operators = {
    'load_numpy': _pipeline_load_numpy,
    'load_hdf5': _pipeline_load_hdf5
}
operators = {}
operators.update(image_operators)
operators.update(load_operators)
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
    iterator = imap(partial(dict_apply, fn=floatX, cols=cols), iterator)
    if cols:
        iterator = imap(lambda data: {c: data[c] for c in cols}, iterator)
    if repeat:
        iterator = cycle(iterator)
    return iterator

def floatX(X):
    #TODO should depend on theano.config.floatX
    # for tensorflow, what is the equivalent thing?
    return np.array(X).astype('float32')

def intX(X):
    return X.astype(np.int32)
