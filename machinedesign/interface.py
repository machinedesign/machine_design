from __future__ import print_function
import os
import numpy as np
import time
from functools import partial
try:
    from itertools import imap
except ImportError:
    imap = map

from .common import build_optimizer
from .common import mkdir_path

from .objectives import get_loss
from .data import pipeline_load
from .data import get_nb_samples
from .data import get_nb_minibatches
from .data import get_shapes
from .data import batch_iterator
from .data import dict_apply

from .callbacks import CallbackContainer
from .callbacks import BudgetFinishedException
from .callbacks import TimeBudget
from .callbacks import RecordEachEpoch
from .callbacks import build_early_stopping_callback
from .callbacks import build_model_checkpoint_callback
from .callbacks import build_lr_schedule_callback

from .transformers import make_transformers_pipeline
from .transformers import transform_one
from .transformers import fit_transformers

from . import metrics as metric_functions
from .metrics import compute_metric

import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "train",
    "show_model_info"
]

def train(params, builders={}, inputs='X', outputs='y', logger=logger, callbacks=[]):
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
    metrics = report['metrics']

    optim = params['optim']
    max_nb_epochs = optim['max_nb_epochs']
    batch_size = optim['batch_size']
    pred_batch_size = optim['pred_batch_size']
    algo_name = optim['algo']['name']
    algo_params = optim['algo']['params']
    loss_name = optim['loss']
    lr_schedule = optim['lr_schedule']
    budget_secs = float(optim['budget_secs'])
    seed = optim['seed']

    lr_schedule_name = lr_schedule['name']
    lr_schedule_params = lr_schedule['params']

    early_stopping = optim['early_stopping']
    early_stopping_name = early_stopping['name']
    early_stopping_params = early_stopping['params']

    checkpoint = report['checkpoint']

    # set the seed there
    np.random.seed(seed)

    # build and fit transformers
    train_pipeline = data['train']['pipeline']

    logger.info('Fitting transformers on training data...')
    transformers = make_transformers_pipeline(data['transformers'])
    def transformers_data_generator():
        it = pipeline_load(train_pipeline)
        it = batch_iterator(it, batch_size=batch_size, repeat=False, cols=[inputs])
        it = imap(lambda d:d[inputs], it)
        return it
    fit_transformers(
        transformers,
        transformers_data_generator
    )
    # save transformers
    mkdir_path(outdir)
    with open(os.path.join(outdir, 'transformers.pkl'), 'wb') as fd:
        pickle.dump(transformers, fd)

    # Load data iterators
    iterators = {}
    nb_train_samples = get_nb_samples(train_pipeline)
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)
    apply_transformers = partial(transform_one, transformers=transformers)
    def train_data_generator(batch_size=batch_size, repeat=False):
        it = pipeline_load(train_pipeline)
        it = batch_iterator(it, batch_size=batch_size, repeat=repeat, cols=[inputs, outputs])
        it = imap(partial(dict_apply, fn=apply_transformers, cols=[inputs]), it)
        return it
    iterators['train'] = train_data_generator

    # Build and compile model
    shapes = get_shapes(next(train_data_generator(batch_size=batch_size, repeat=False)))
    model = _build_model(
        name=model_name,
        params=model_params,
        shapes=shapes,
        builders=builders)

    optimizer = build_optimizer(algo_name, algo_params)
    loss = get_loss(loss_name)
    model.compile(loss=loss, optimizer=optimizer)

    show_model_info(model, print_func=logger.info)

    # Build callbacks
    learning_rate_scheduler = build_lr_schedule_callback(
        name=lr_schedule_name,
        params=lr_schedule_params,
        print_func=logger.debug)

    early_stopping = build_early_stopping_callback(
        name=early_stopping_name,
        params=early_stopping_params)

    model_filename = os.path.join(outdir, 'model.h5')
    checkpoint = build_model_checkpoint_callback(
        model_filename=model_filename,
        params=checkpoint)

    metric_callbacks = []
    for metric in metrics:
        metric_func = getattr(metric_functions, metric)
        for which in ('train',):
            compute_func = _build_compute_func(
                predict=model.predict,
                data_generator=lambda: iterators[which](batch_size=pred_batch_size, repeat=False),
                metric=metric_func,
                inputs=inputs,
                outputs=outputs,
                aggregate=np.mean)
            callback = RecordEachEpoch(which + '_' + metric, compute_func)
            metric_callbacks.append(callback)

    time_budget = TimeBudget(budget_secs=budget_secs)
    basic_callbacks = [
        learning_rate_scheduler,
        early_stopping,
        checkpoint
    ]
    callbacks = metric_callbacks + basic_callbacks + callbacks + [time_budget]
    for cb in callbacks:
        cb.model = model
        cb.data_iterators = iterators
        cb.params = params
        cb.transformers = transformers
    callbacks = CallbackContainer(callbacks)

    # Training loop
    callbacks.on_train_begin()
    train_iterator = train_data_generator(batch_size=batch_size, repeat=True)
    for epoch in range(max_nb_epochs):
        logger.info('Epoch {:05d}...'.format(epoch))
        dt = time.time()
        stats = {}
        callbacks.on_epoch_begin(epoch, logs=stats)
        for _ in range(nb_minibatches):
            train_batch = next(train_iterator)
            X, Y = train_batch[inputs], train_batch[outputs]
            model.fit(X, Y, verbose=0)
        try:
            callbacks.on_epoch_end(epoch, logs=stats)
        except BudgetFinishedException:
            logger.info('Budget finished. Stop.')
            model.stop_training = True
        for k, v in stats.items():
            logger.info('{}={:.4f}'.format(k, v))
        logger.info('elapsed time : {:.3f}s'.format(time.time() - dt))
        _update_history(model, logs=stats)
        # the following happens
        # when early stopping or budget finished
        if model.stop_training:
            logger.info('Stopping training.')
            break
    return model

def load(folder):
    pass

def generate(params):
    pass

def show_model_info(model, print_func=print):
    print_func('Number of parameters : {}'.format(model.count_params()))
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    nb_W_params = sum(np.prod(layer.W.get_value().shape) for layer in model.layers if hasattr(layer, 'W'))
    print_func('Number of weight parameters : {}'.format(nb_W_params))
    print_func('Number of learnable layers : {}'.format(nb))

def _update_history(model, logs):
    for k, v in logs.items():
        if k not in model.history.history:
            model.history.history[k] = []
        model.history.history[k].append(v)

def _build_compute_func(predict, data_generator, metric,
                        inputs='X', outputs='y',
                        aggregate=np.mean):
    get_real_and_pred = lambda: imap(lambda data: (data[inputs], predict(data[inputs])), data_generator())
    compute_func = lambda: aggregate(compute_metric(get_real_and_pred, metric))
    return compute_func

def _build_model(name, params, shapes, builders={}):
    model_builder = builders[name]
    model = model_builder(params, shapes) # keras model
    return model
