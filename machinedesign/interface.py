from __future__ import print_function
import os
import numpy as np
import time
from functools import partial
from six.moves import map
from collections import namedtuple

from .common import build_optimizer
from .common import show_model_info
from .common import callback_trigger

from .utils import mkdir_path
from .utils import write_csv

from .objectives import get_loss
from .data import pipeline_load
from .data import get_nb_samples
from .data import get_nb_minibatches
from .data import get_shapes
from .data import batch_iterator
from .data import floatX
from datakit.helpers import dict_apply

from .callbacks import BudgetFinishedException
from .callbacks import StopTrainingException
from .callbacks import TimeBudget
from .callbacks import RecordEachEpoch
from .callbacks import build_early_stopping_callback
from .callbacks import build_models_checkpoint_callback
from .callbacks import build_lr_schedule_callback

from .transformers import make_transformers_pipeline
from .transformers import transform_one
from .transformers import fit_transformers

from .metrics import compute_metric
from .metrics import get_metric

# elements of Config
from .model_builders import builders as model_builders
from .layers import layers
from .data import operators as data_operators
from .transformers import transformers
from .metrics import metrics
from .objectives import objectives

import pickle
import logging

logging.basicConfig(
    format='%(asctime)s ## %(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y,%I:%M:%S')
logger = logging.getLogger(__name__)

Config = namedtuple(
    'Config',
    ['model_builders',
     'layers',
     'data_operators',
     'transformers',
     'metrics',
     'objectives'
     ])

default_config = Config(
    model_builders=model_builders,
    layers=layers,
    data_operators=data_operators,
    transformers=transformers,
    metrics=metrics,
    objectives=objectives
)


def train(params,
          config=default_config,
          custom_callbacks=[],
          logger=logger):
    """
    Generic training procedure to train a mapping from some inputs
    to some outputs. You can use this for most kind of models (e.g autoencoders),
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
    input_col = params['input_col']
    output_col = params['output_col']

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

    transformers = make_transformers_pipeline(
        data['transformers'],
        transformers=config.transformers)

    def transformers_data_generator():
        it = pipeline_load(train_pipeline, operators=config.data_operators)
        it = batch_iterator(it, batch_size=batch_size, repeat=False, cols=[input_col])
        it = map(lambda d: d[input_col], it)
        return it

    fit_transformers(
        transformers,
        transformers_data_generator
    )
    logger.info('Saving transformers into {}'.format(outdir))
    # save transformers
    mkdir_path(outdir)
    with open(os.path.join(outdir, 'transformers.pkl'), 'wb') as fd:
        pickle.dump(transformers, fd)

    # Load data iterators
    iterators = {}
    # the number of samples may not be the actual number of samples
    # because get_nb_samples is only based on the first operator
    # of the datakit pipeline
    # (maybe force it to pass through everything?)
    # So instead we can provide the number of sample explicitly
    # to know how many minibatches we have per epoch
    logger.debug('Getting number of samples on training data...')
    nb_train_samples = data['train'].get(
        'nb_samples', get_nb_samples(transformers_data_generator()))
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)
    logger.info('Number of training examples : {}'.format(nb_train_samples))
    logger.info('Number of training minibatches : {}'.format(nb_minibatches))
    apply_transformers = partial(transform_one, transformer_list=transformers)

    def data_generator(batch_size=batch_size, repeat=False, pipeline=train_pipeline):
        it = pipeline_load(pipeline)
        it = batch_iterator(it, batch_size=batch_size, repeat=repeat, cols=[input_col, output_col])
        it = map(partial(dict_apply, fn=apply_transformers, cols=[input_col]), it)
        it = map(partial(dict_apply, fn=floatX, cols=[input_col, output_col]), it)
        return it
    iterators['train'] = partial(data_generator, pipeline=train_pipeline)
    # add other data splits than train
    for name, params in data.items():
        if name in ('train', 'transformers'):
            continue
        pipeline = params['pipeline']
        iterators[name] = partial(data_generator, pipeline=pipeline)
    # Build and compile model
    logger.debug('Getting input and output shapes...')
    shapes = get_shapes(next(data_generator(batch_size=batch_size,
                                            repeat=False, pipeline=train_pipeline)))
    model = _build_model(
        name=model_name,
        params=model_params,
        input_shape=shapes[input_col],
        output_shape=shapes[output_col],
        builders=config.model_builders)
    # TODO: understand more this
    # I did this to avoid missinginputerror of K.learning_phase when
    # using a model in the loss such as objectness. If the model does
    # not have any layer that differ in train/test phase (such as dropout)
    # then uses_learning_phase is False, but if the behavior of the model
    # used in the loss needs uses_learning_phase and it's false, then it
    # throws this missinginputerror, so I force uses_learning_phase to
    # True.
    for lay in model.layers:
        lay.uses_learning_phase = True

    optimizer = build_optimizer(algo_name, algo_params)
    loss = get_loss(loss_name, objectives=config.objectives)
    logger.debug('Compiling the model...')
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
    checkpoint = build_models_checkpoint_callback(
        params=checkpoint,
        models=[model],
        filepaths=[model_filename])

    metric_callbacks = []
    for metric in metrics:
        # metric can either be a str and in that case, the metric will be computed for
        # all data splits (e.g train, valid) defined in data.
        # or it can be a dict, and in that case it should have the keys:
        # - "name" : name of the metric
        # - "params"(optional) : the params of the metric
        # - "data" (optional): data where to compute the metric
        # - "each" (optional) : compute the metric each 'each' epochs
        # (if not given, it will be computed for alll data splits)
        metric_func = get_metric(metric, metrics=config.metrics)
        if isinstance(metric, dict):
            metric_name = metric['name']
            data = metric.get('data', iterators.keys())
            each = metric.get('each', 1)
        else:
            metric_name = metric
            data = iterators.keys()
            each = 1
        for which in data:
            compute_func_ = _build_compute_func(
                predict=model.predict,
                data_generator=lambda which=which: iterators[which](
                    batch_size=pred_batch_size, repeat=False),
                metric=metric_func,
                input_col=input_col,
                output_col=output_col,
                aggregate=np.mean)
            callback = RecordEachEpoch(
                which + '_' + metric_name,
                partial(verbose_compute_func, which=which, metric=metric_name,
                        func=compute_func_, logger=logger),
                each=each)
            metric_callbacks.append(callback)

    time_budget = TimeBudget(budget_secs=budget_secs)
    basic_callbacks = [
        learning_rate_scheduler,
        early_stopping,
        checkpoint
    ]
    callbacks = metric_callbacks + basic_callbacks + custom_callbacks + [time_budget]
    for cb in callbacks:
        cb.model = model
        cb.data_iterators = iterators
        cb.params = params
        cb.transformers = transformers

    # Training loop
    callback_trigger(callbacks, 'on_train_begin')

    train_iterator = data_generator(batch_size=batch_size, repeat=True, pipeline=train_pipeline)

    history_stats = []
    model.history_stats = history_stats
    for epoch in range(max_nb_epochs):
        logger.info('Starting epoch {:05d}...'.format(epoch))
        t0 = time.time()
        stats = {}
        callback_trigger(callbacks, 'on_epoch_begin', epoch, logs=stats)
        t0 = time.time()
        for _ in range(nb_minibatches):
            train_batch = next(train_iterator)
            X, Y = train_batch[input_col], train_batch[output_col]
            model.train_on_batch(X, Y)

        training_time = time.time() - t0

        t0_callbacks = time.time()
        try:
            callback_trigger(callbacks, 'on_epoch_end', epoch, logs=stats)
        except BudgetFinishedException:
            logger.info('Budget finished. Stop training.')
            stop_training = True
        except StopTrainingException:
            logger.info('Early stopping. Stop training.')
            stop_training = True
        else:
            stop_training = False
        callback_time = time.time() - t0_callbacks
        total_time = time.time() - t0
        logger.info('Finished training epoch {:05d}...'.format(epoch))
        history_stats.append(stats)
        for k, v in stats.items():
            logger.info('{}={:.8f}'.format(k, v))
        write_csv(history_stats, os.path.join(outdir, 'stats.csv'))

        logger.info('Training time : {:.3f}s'.format(training_time))
        logger.info('Callbacks time : {:.3f}s'.format(callback_time))
        logger.info('Total elapsed time in the epoch {:05d} : {:.3f}s'.format(epoch, total_time))

        # the following happens
        # when early stopping or budget finished
        if stop_training:
            break

    return model


def load(folder):
    pass


def generate(params):
    pass


def _update_history(model, logs):
    for k, v in logs.items():
        if k not in model.history.history:
            model.history.history[k] = []
        model.history.history[k].append(v)


def verbose_compute_func(which, metric, func, logger):
    logger.debug('Computing the metric "{}" on "{}"...'.format(metric, which))
    t0 = time.time()
    val = func()
    delta_t = time.time() - t0
    logger.debug('Done computing the metric "{}" on "{}" in {:.3f} seconds, the value is : {}'.format(
        metric, which, delta_t, val))
    return val


def _build_compute_func(predict, data_generator, metric,
                        input_col='X', output_col='y',
                        aggregate=np.mean):

    def _get_real_and_pred_batch(data):
        yreal = data[output_col]
        ypred = predict(data[input_col])
        return yreal, ypred

    def _get_real_and_pred():
        data = data_generator()
        data = map(_get_real_and_pred_batch, data)
        return data

    def _compute_func():
        return aggregate(compute_metric(_get_real_and_pred, metric))
    return _compute_func


def _build_model(name, params, input_shape, output_shape, builders={}):
    model_builder = builders[name]
    model = model_builder(params, input_shape, output_shape)  # keras model
    return model
