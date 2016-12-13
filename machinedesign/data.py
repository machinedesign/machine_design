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

operators = image_operators
pipeline_load = partial(pipeline_load, operators=operators)

def get_nb_samples(pipeline):
    p = pipeline[0:1]
    p = pipeline_load(p)
    p = list(p)
    return len(p)

def get_shapes(iterator):
    sample = next(iterator)
    #v.shape[1:] because we only return the dimensions coming
    # after the 'nb of examples' dimennsion
    return {k: v.shape for k, v in sample.items()}

def get_nb_minibatches(nb_samples, batch_size):
    return (nb_samples // batch_size) + (nb_samples % batch_size > 0)

class BatchIterator(object):

    def __init__(self, iterator_func, cols=['X', 'y']):
        self.iterator_func = iterator_func
        self.cols = cols

    def flow(self, batch_size=128, repeat=True):
        iterator = self.iterator_func()
        iterator = minibatch(iterator, batch_size=batch_size)
        iterator = expand_dict(iterator)
        iterator = imap(partial(dict_apply, fn=floatX, cols=self.cols), iterator)
        iterator = imap(lambda data: {c: data[c] for c in self.cols}, iterator)
        if repeat:
            iterator = cycle(iterator)
        return iterator

def floatX(X):
    return np.array(X).astype('float32')

class ArrayBatchIterator(object):

    def __init__(self, inputs, targets=None,
                 shuffle=False, random_state=None):
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        if targets is not None:
            assert len(inputs) == len(targets)

    def flow(self, batch_size=128, repeat=True):
        while True:
            if self.shuffle:
                indices = np.arange(len(self.inputs))
                self.rng.shuffle(indices)
            for start_idx in range(0, len(self.inputs), batch_size):
                if self.shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                if self.targets is not None:
                    yield self.inputs[excerpt], self.targets[excerpt]
                else:
                    yield self.inputs[excerpt]
            if repeat is False:
                break
