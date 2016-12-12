from itertools import cycle

from ..common import build_optimizer
from ..common import get_loss
from ..data import pipeline_load
from ..data import get_nb_samples
from ..data import get_nb_minibatches

import model_builders

def train(params):
    # Get relevant variables from params
    model_name = params['model']['name']
    model_params = params['model']['params']
    data = params['data']
    report = params['report']

    optim = params['optim']
    max_nb_epochs = optim['max_nb_epochs']
    batch_size = optim['batch_size']
    algo_name = optim['algo_name']
    algo_params = optim['algo_params']
    loss = optim['loss']

    # Build and compile model
    model = _build_model(model_name, model_params)
    optimizer = build_optimizer(algo_name, algo_params)
    loss = get_loss(loss)
    model.compile(loss=loss, optimizer=optimizer)

    # Load data iterators
    train_iterator = cycle(pipeline_load(data['train']['pipeline']))
    nb_train_samples = get_nb_samples(data['train'])
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)

    for epoch in range(max_nb_epochs):
        for minibatch in range(nb_minibatches):
            train = next(train_iterator)
            X, Y = train['X'], train['X']
            model.fit(X, Y)

def _build_model(name, params):
    model_builder = getattr(model_builders, name)
    model = model_builder(params) # keras model
    return model

def _start_epoch_report(model):
    pass

def load(filename):
    pass

def generate(params):
    pass
