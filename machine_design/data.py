from __future__ import division
from functools import partial
from datakit.pipeline import pipeline_load
from datakit.image import operators as image_operators

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
    return {k: v.shape[1:] for k, v in sample.items()}

def get_nb_minibatches(nb_samples, batch_size):
    return (nb_samples // batch_size) + (nb_samples % batch_size > 0)
