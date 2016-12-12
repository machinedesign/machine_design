from functools import partial
from datakit.pipeline import pipeline_load

operators = {
}

pipeline_load = partial(pipeline_load, operators=operators)

def get_nb_samples(pipeline):
    return len(list(pipeline_load(pipeline[0:1])))

def get_nb_minibatches(nb_samples, batch_size):
    return (nb_samples / batch_size) + (nb_samples % batch_size > 0)
