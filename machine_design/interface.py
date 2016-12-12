import os
from itertools import cycle
import numpy as np

from .common import build_optimizer
from .common import get_loss
from .data import pipeline_load
from .data import get_nb_samples
from .data import get_nb_minibatches
from .data import get_shapes

from .callbacks import CallbackList
from .callbacks import BudgetFinishedException
from .callbacks import TimeBudget
from .callbacks import build_early_stopping_callback
from .callbacks import build_model_checkpoint_callback
from .callbacks import build_lr_schedule_callback

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(params, builders={}, inputs='X', outputs='y', logger=logger):
    """
    Generic training procedure to train a mapping from some inputs
    to some outputs. You can use this for most kind of models (e.g autoencoders,
    variational autoencoders, etc.) but not with GANs. GANs have their own module
    with the same interface.
    """
    # Get relevant variables from params
    model_name = params['model']['name']
    model_params = params['model']['params']
    data = params['data']
    report = params['report']
    outdir = report['outdir']

    optim = params['optim']
    max_nb_epochs = optim['max_nb_epochs']
    batch_size = optim['batch_size']
    algo_name = optim['algo']['name']
    algo_params = optim['algo']['params']
    loss_name = optim['loss']
    lr_schedule = optim['lr_schedule']
    budget_secs = float(optim['budget_secs'])
    lr_schedule_name = lr_schedule['name']
    lr_schedule_params = lr_schedule['params']

    early_stopping = optim['early_stopping']
    early_stopping_name = early_stopping['name']
    early_stopping_params = early_stopping['params']

    checkpoint = report['checkpoint']

    # Build and compile model
    shapes = get_shapes(pipeline_load(data['train']['pipeline']))
    model = build_model(
        name=model_name, params=model_params,
        shapes=shapes, builders=builders)

    optimizer = build_optimizer(algo_name, algo_params)
    loss = get_loss(loss_name)
    model.compile(loss=loss, optimizer=optimizer)

    logger.info('Number of parameters : {}'.format(model.count_params()))
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    nb_W_params = sum(np.prod(layer.W.get_value().shape) for layer in model.layers if hasattr(layer, 'W'))
    logger.info('Number of weight parameters : {}'.format(nb_W_params))
    logger.info('Number of learnable layers : {}'.format(nb))

    # Load data iterators
    train_iterator = cycle(pipeline_load(data['train']['pipeline']))
    nb_train_samples = get_nb_samples(data['train']['pipeline'])
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)

    # Build callbacks
    learning_rate_scheduler = build_lr_schedule_callback(
        name=lr_schedule_name,
        params=lr_schedule_params,
        print=logger.info,
        model=model)

    early_stopping = build_early_stopping_callback(
        name=early_stopping_name,
        params=early_stopping_params,
        model=model)

    model_filename = os.path.join(outdir, 'model.pkl')
    checkpoint = build_model_checkpoint_callback(
        model_filename=model_filename,
        params=checkpoint,
        model=model)

    time_budget = TimeBudget(budget_secs=budget_secs)
    callbacks = [
        learning_rate_scheduler,
        early_stopping,
        checkpoint,
        time_budget
    ]
    callbacks = CallbackList(callbacks)

    for epoch in range(max_nb_epochs):
        logger.info('Epoch {:05d}...'.format(epoch))
        stats = {}
        callbacks.on_epoch_begin(epoch, logs=stats)
        for minibatch in range(nb_minibatches):
            train = next(train_iterator)
            X, Y = train[inputs], train[outputs]
            model.fit(X, Y)
        try:
            callbacks.on_epoch_end(epoch, logs=stats)
        except BudgetFinishedException:
            break
        for k, v in stats.items():
            logger.info('{}={:.4f}'.format(k, v))

def build_model(name, params, shapes, builders={}):
    model_builder = builders[name]
    model = model_builder(params, shapes) # keras model
    return model

def load(folder):
    pass

def generate(params):
    pass
