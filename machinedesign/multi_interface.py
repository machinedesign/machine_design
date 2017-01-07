from __future__ import print_function
import numpy as np
from functools import partial
import os
import time
from six.moves import map

from .common import build_optimizer
from .common import show_model_info
from .common import callback_trigger

from .utils import write_csv

from .objectives import get_loss
from .data import pipeline_load
from .data import get_nb_samples
from .data import get_nb_minibatches
from .data import get_shapes
from .data import batch_iterator

from .callbacks import BudgetFinishedException
from .callbacks import StopTrainingException
from .callbacks import TimeBudget
from .callbacks import RecordEachEpoch
from .callbacks import build_early_stopping_callback
from .callbacks import build_models_checkpoint_callback
from .callbacks import build_lr_schedule_callback

from . import metrics as metric_functions
from .metrics import compute_metric

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_data_generator(pipeline, cols='all'):
    """
    take a `datakit` pipeline and returns a function that
    takes batch_size and repeat, and returns an iterator.

    Parameters
    ----------

    pipeline : list of dict
        `datakit` pipeline
    cols : list of str or str
        if list, list of columns to retrieve from the `datakit` pipeline
        if str, should be 'all', this will include all the available columns.

    Returns
    -------

    a function that takes (batch_size, repeat=False) and returns an iterator
    of dict, where the keys are modalities (e.g `X`, `y`) and the values
    are data.
    """
    def _gen(batch_size, repeat=False):
        it = pipeline_load(pipeline)
        it = batch_iterator(it, batch_size=batch_size, repeat=repeat, cols=cols)
        return it
    return _gen

def train(params, model_builders={}, logger=logger, callbacks=[],
          build_data_generator=build_data_generator):
    """
    Train a set of predictors and their corresponding evaluators jointly.
    This training interface is a generalization of the GANs.

    Parameters
    ----------

    params : dict
        parameters necessary for training
    model_builders : dict
        keys are model type names
        values are functions that build models based on params, input shape and output shape
        (e.g check model_builders module to know the expected signature)
    logger : logger to use for logging
    callbacks : list of Callback
        additional callbacks to use
    build_data_generator : function
        function that builds an iterator that gives data, by default it uses
        the one already present in multi_interface module.
        it can be modified to allow more flexibility.
    """

    # get relevant entries from the params
    models = params['models']
    data = params['data']
    optim = params['optim']
    batch_size = optim['batch_size']
    max_nb_epochs = optim['max_nb_epochs']
    seed = optim['seed']

    callbacks_spec = params['callbacks']
    outdir = callbacks_spec['outdir']
    budget_secs = callbacks_spec['budget_secs']
    early_stopping = callbacks_spec['early_stopping']
    early_stopping_name = early_stopping['name']
    early_stopping_params = early_stopping['params']
    checkpoint = callbacks_spec['checkpoint']

    data_train_pipeline = data['train']['pipeline']

    np.random.seed(seed)

    # Build data generators
    data_train_generator = build_data_generator(data_train_pipeline)
    data_generators = {
        'train': data_train_generator
    }
    # this is a dict of column names as keys (e.g X, y) and shapes as values
    # e.g (3, 32, 32) for X.
    col_shapes = get_shapes(next(data_train_generator(batch_size=batch_size)))

    # Build and compile models
    build_model_func = partial(
        build_model_and_evaluators_from_spec,
        col_shapes=col_shapes,
        builders=model_builders)
    # models are a list of models (predictors), each model
    # has a reference to its own evaluators, via model.evalutors.
    # evaluators are also models. It could have been also recursive
    # by having evaluators of evaluators, but I choose to stop the
    # recursion so far. So the way to parse this 'graph' is through:
    # 'for model in models for evaluator in model.evaluators'.
    models = list(map(build_model_func, models))
    # compiling models and evaluators, respecting the order of
    # first compiling evaluators of a model before the model itself
    models = compile_models_and_evaluators(models)

    # get a full list of models, whether it is a predictor or an evaluator
    # then show useful infos for each model
    models_and_evaluators = list(get_models_and_evaluators(models))
    for model in models_and_evaluators:
        logger.info('Model "{}"'.format(model.name))
        show_model_info(model)

    # build callbacks for all the models (predictors + evaluators)

    #callbacks specific to each model (predictors + evaluators)
    callbacks_list = build_models_callbacks(models_and_evaluators, data_generators=data_generators)
    # general callbacks
    time_budget = TimeBudget(budget_secs=budget_secs)
    early_stopping = build_early_stopping_callback(early_stopping_name, early_stopping_params)
    filepaths = [os.path.join(outdir, model.name + '.h5') for model in models_and_evaluators]
    checkpoint = build_models_checkpoint_callback(checkpoint, models_and_evaluators, filepaths)
    callbacks_list += [early_stopping, checkpoint]
    callbacks_list += callbacks
    callbacks_list += [time_budget]
    # compat with interface.py
    for cb in callbacks_list:
        cb.model = models[0]
        cb.data_iterators = data_generators
        cb.params = cb.model.spec
        cb.transformers = []
        cb.outdir = outdir

    nb_train_samples = data['train'].get('nb_samples', get_nb_samples(data_train_pipeline))
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)
    logger.info('Number of training examples : {}'.format(nb_train_samples))
    logger.info('Number of training minibatches : {}'.format(nb_minibatches))

    # Training loop!
    train_iterator = data_train_generator(batch_size=batch_size, repeat=True)
    callback_trigger(callbacks_list, 'on_train_begin')
    history_stats = []
    for epoch in range(max_nb_epochs):
        logger.info('Epoch {}...'.format(epoch))
        t0 = time.time()
        stats = {}
        callback_trigger(callbacks_list, 'on_epoch_begin', epoch, logs=stats)
        for _ in range(nb_minibatches):
            train_batch = next(train_iterator)
            train_models_and_evaluators_on_batch(models, train_batch)
        try:
            callback_trigger(callbacks_list, 'on_epoch_end', epoch, logs=stats)
        except BudgetFinishedException:
            logger.info('Budget finished.')
            stop_training = True
        except StopTrainingException:
            logger.info('Early stopping because patience exceeded.')
            stop_training = True
        else:
            stop_training = False

        for k, v in stats.items():
            logger.info('{}={:.4f}'.format(k, v))
        history_stats.append(stats)
        write_csv(history_stats, os.path.join(outdir, 'stats.csv'))
        logger.info('elapsed time : {:.3f}s'.format(time.time() - t0))
        if stop_training:
            logger.info('Stop training.')
            break
    return models

def build_model_and_evaluators_from_spec(spec, col_shapes, builders={},
                                         override_input_shape=None,
                                         override_output_shape=None):
    """
    build models and their corresponding evaluators from a configuration
    described in `spec`.
    Use a library of model builders from `builders` to build models.
    Building models require to know input and output shapes.
    input and output columns are provided in the `spec` as `input_col` and `output_col`
    sot that col_shapes[input_col] and col_shapes[output_col] can  be used to know the shapes.

    if input_col is not provided, use `override_input_shape`, which in that case should be not None,
    otherwise an exception will be thrown. Same for output_col.

    For each model, we build it, then build its evaluators.

    The models are keras Model augmented with the following attributes:
        name : str
            model name, used to identify models, the name is used to prefix
            the metrics and to specify for a model which evaluator to use in
            the loss function.
        get_input_col : callable
            takes a dict of data (keys are modalities, e.g X and y, values are data)
            and return data corresponding to the input col used by the model.
        get_output_col : callable
            takes a dict of data (keys are modalities, e.g X and y, values are data)
            and return data corresponding to the output col used by the model.
            For models, it is just returning the modality provided by output_col.
            For evaluators, it returns evaluator.get_real_output(input_col).
        optimizer : keras Optimizer of the model
        evaluators : list of keras model
            evaluators of the model
        spec : dict
            the spec of the model

    Parameters
    ----------
    spec : dict
        config of the models
    col_shapes : dict
        shapes of modalities, keys are modalities (e.g X or y), values are data.
    builders : dict
        keys are str, corresponding to type of models, and values are callable
        that construct keras Model.
    override_input_shape : tuple or None
        if input_col is not provided, use `override_input_shape`.
    override_output_shape : tuple or None
        if output_col is not provided, use `override_output_shape`.

    Returns
    -------
    list of keras Model where each model is augmented with a set
    of attributes (see above)
    """
    name = spec['name']
    input_col = spec.get('input_col')
    output_col = spec.get('output_col')
    model_type = spec['architecture']['name']
    model_type_params = spec['architecture']['params']
    algo_name = spec['optimizer']['name']
    algo_params = spec['optimizer']['params']
    losses = spec['losses']
    evaluators_spec = spec.get('evaluators', [])

    input_shape = col_shapes[input_col] if input_col else override_input_shape
    output_shape = col_shapes[output_col] if output_col else override_output_shape
    assert input_shape is not None, 'expected input_shape to be not None'
    assert output_shape is not None, 'expected output_shape to be not None'

    model = build_model(
        type=model_type,
        params=model_type_params,
        input_shape=input_shape,
        output_shape=output_shape,
        builders=builders)
    model.name = name

    evaluators = []
    for evaluator_spec in evaluators_spec:
        type_ = evaluator_spec['type']
        get_real_output, get_fake_output, output_shape = get_evaluator_output_funcs_and_shape(type_)
        evaluator = build_model_and_evaluators_from_spec(
                evaluator_spec,
                col_shapes=col_shapes,
                builders=builders,
                override_output_shape=output_shape)
        evaluator.get_real_output = get_real_output
        evaluator.get_fake_output = get_fake_output
        evaluator.get_output_col = lambda d:get_real_output(evaluator.get_input_col(d), backend=np)
        evaluators.append(evaluator)

    evaluator_by_name = {evaluator.name: evaluator for evaluator in evaluators}

    loss_names = (loss['name'] for loss in losses)
    loss_funcs = list(map(lambda name: build_loss_func_from_evaluator(evaluator_by_name[name])
                          if name in evaluator_by_name else get_loss(name), loss_names))
    loss_coefs = [loss['coef'] for loss in losses]

    model.loss_func = loss_aggregate(loss_coefs, loss_funcs)
    if input_col:
        model.get_input_col = lambda d:d[input_col]
    if output_col:
        model.get_output_col = lambda d:d[output_col]
    model.optimizer = build_optimizer(algo_name, algo_params)
    model.evaluators = evaluators
    model.spec = spec
    return model

def build_model(type, params, input_shape, output_shape, builders={}):
    """
    build a keras Model.
    Building a keras model requires specifying its `type` and the
    parameters `params` corresponding to the `type`. The `type` is a
    str that should exist in the keys of `builders`.`builders` is a
    mapping from type names to functions that build a keras Model.
    It is also required to specify `input_shape` and `output_shape`
    of the model.

    Parameters
    ----------

    type : str
        name of the type of the model
    params : dict
        parameters corresponding to the type of model
    input_shape : tuple
        shape of the input, starting from the second axis
        (the first axis is the number of examples and is not specified)
    output_shape : tuple
        shape of the output, starting from the second axis
        (the first axis is the number of examples and is not specified)
    builders : dict
        keys are type names, values are functions that build keras Model.
        (see e.g `autoencoder.model_builders` for examples of builders)
    Returns
    -------

    """
    model_builder = builders[type]
    model = model_builder(params, input_shape, output_shape) # keras Model
    return model

def get_evaluator_output_funcs_and_shape(type_):
    """

    get evaluator necessary functions to be trained.
    evaluators need two functions, `get_real_output`, `get_fake_output`, and also the shape
    of the output to be trained in e.g `train_evaluator_on_batch`.
    Different types of evaluators could be desired, but for now only one type
    is supported, one that is equivalent to GANs, `type_`=="discriminator"
    In that case, `get_real_output` returns a set of ones and `get_fake_output`
    returns a set of zeros.

    Parameters
    ----------

    type_ : 'discriminator'
        type of evaluator

    Returns
    -------

    (get_real_output, get_fake_output, output_shape)

    where get_real_output is a function, get_fake_output is a function
    and output_shape is a tuple.

    """
    if type_ == 'discriminator':
        get_real_output = lambda Y, backend:backend.ones((Y.shape[0], 1))
        get_fake_output = lambda Y, backend:backend.zeros((Y.shape[0], 1))
        return get_real_output, get_fake_output, (1,)
    else:
        raise ValueError('Expected type_ to be discriminator, got : {}'.format(type_))

def build_loss_func_from_evaluator(evaluator):
    """

    build a loss function to be used by model (e.g a generator) from an evaluator.
    the loss should express what the model seeks, instead of what the evaluator seeks,
    that is, in GAN formulation, if the model is a generator, the loss from the evaluator
    (which is representing the discriminator) should be we want to maximize
    the fake data to be predicted as real.
    Parameters
    ----------

    evaluator : keras Model

    Returns
    -------

    loss function
    """
    def loss(y_real, y_fake):
        import theano.tensor as T
        # TODO support tensorflow
        z_fake = evaluator(y_fake)
        z_fake_expected = evaluator.get_real_output(y_fake, backend=T)
        return evaluator.loss_func(z_fake, z_fake_expected)
    loss.__name__ = 'loss_evaluator'
    return loss

def loss_aggregate(coefs, funcs):
    """

    Return a function which consists in a weighted average of a set
    of functions with weights coming from coefs.

    Parameters
    ----------

    coefs : list of float
        coeficient of the loss terms
    funcs : list of loss functions

    Returns
    -------

    loss function
    """
    def loss(y_true, y_pred):
        total = 0
        for coef, func in zip(coefs, funcs):
            total += coef * func(y_true, y_pred).mean()
        return total
    loss.__name__ = 'loss_aggregate'
    return loss

def build_models_callbacks(models, data_generators):
    """

    Build all the callbacks necessary for all the models and return all
    the callbacks concatenated into one single list.

    Parameters
    ----------

    models : list of keras Model
    data_generators : dict
        keys are data split names (e.g train, test).
        values are functions that create an iterator.

    Returns
    -------

    list of Callback
    """
    callbacks = []
    for model in models:
        callbacks += build_callbacks(spec=model.spec.get('callbacks', []), model=model, data=data_generators)
    return callbacks

def build_callbacks(spec, model, data):
    """

    Build callbacks from a callback specification for a given model and data generator

    Parameters
    ----------

    spec : dict
        specification of callbacks
    model : keras Model
    data : dict
        keys are data split names (e.g train, test).
        values are functions that create an iterator.

    Returns
    -------

    list of Callback
    """
    lr_schedule = spec['lr_schedule']
    lr_schedule_name = lr_schedule['name']
    lr_schedule_params = lr_schedule['params']

    metrics = spec['metrics']
    metric_names = metrics['names']
    pred_batch_size = metrics['pred_batch_size']

    learning_rate_scheduler = build_lr_schedule_callback(
        name=lr_schedule_name,
        params=lr_schedule_params,
        print_func=logger.debug)

    metric_callbacks = build_metric_callbacks(
        metric_names,
        model,
        data,
        pred_batch_size=pred_batch_size,
        name_prefix=model.name)
    callbacks = [learning_rate_scheduler] + metric_callbacks
    for cb in callbacks:
        cb.model = model
        cb.data_iterators = data
    return callbacks

def build_metric_callbacks(metrics, model, data_generators, pred_batch_size=128, name_prefix=''):
    """

    Build metric callbacks from metrics callback specification for a given model and data generator

    Parameters
    ----------

    metrics : list of str
        each str correspond to a metric name (see `metric` module)
    model : keras Model
    data_generators : dict
        keys are data split names (e.g train, test).
        values are functions that create an iterator.
    pred_batch_size : int
        prediction batch size for computing the metrics
    name_prefix : str
        prefix of the names of the metrics that will appear in the log stats, that is
        if the name of the metric is 'mean_squared_error' and the prefix
        is 'model', and the data split is 'train', it will appear as
        'model_train_mean_squared_error' in the log stats.
    Returns
    -------

    list of Callback
    """
    metric_callbacks = []
    for metric in metrics:
        metric_func = getattr(metric_functions, metric)
        for which, iterator in data_generators.items():
            compute_func = build_compute_func(
                predict=model.predict,
                data_generator=partial(iterator, batch_size=pred_batch_size, repeat=False),
                metric=metric_func,
                get_input_col=model.get_input_col,
                get_output_col=model.get_output_col,
                aggregate=np.mean)
            callback = RecordEachEpoch(name_prefix + '_' + which + '_' + metric, compute_func)
            metric_callbacks.append(callback)
    return metric_callbacks

def build_compute_func(predict, data_generator, metric,
                       get_input_col, get_output_col,
                       aggregate=np.mean):
   """
   Parameters
   ----------


   Returns
   -------
   """
   get_real_and_pred = lambda: map(lambda data: (get_output_col(data), predict(get_input_col(data))), data_generator())
   compute_func = lambda: aggregate(compute_metric(get_real_and_pred, metric))
   return compute_func

def compile_models_and_evaluators(models):
    """
    Parameters
    ----------

    Returns
    -------
    """
    for model in models:
        for evaluator in model.evaluators:
            logger.info('Compiling "{}"...'.format(evaluator.name))
            evaluator.compile(optimizer=evaluator.optimizer, loss=evaluator.loss_func)
        logger.info('Compiling ""{}"...'.format(model.name))
        model.compile(optimizer=model.optimizer, loss=model.loss_func)
    return models

def get_models_and_evaluators(models):
    """
    Parameters
    ----------

    Returns
    -------
    """
    for model in models:
        for evaluator in model.evaluators:
            yield evaluator
        yield model

def train_models_and_evaluators_on_batch(models, train_batch):
    """
    Parameters
    ----------

    Returns
    -------
    """
    for model in models:
        X = model.get_input_col(train_batch)
        Y = model.get_output_col(train_batch)
        model.train_on_batch(X, Y)
        Y_real = Y
        Y_fake = model.predict(X)
        for evaluator in model.evaluators:
            train_evaluator_on_batch(evaluator, Y_real, Y_fake)

def train_evaluator_on_batch(evaluator, Y_real, Y_fake):
    """
    Parameters
    ----------

    Returns
    -------
    """

    # in GAN it is [1 1 1 1....]
    Z_real = evaluator.get_real_output(Y_real, backend=np)
    # in GAN it is [0 0 0 0....]
    Z_fake = evaluator.get_fake_output(Y_fake, backend=np)
    # merge them
    Y_concat = np.concatenate((Y_real, Y_fake), axis=0)
    Z_concat = np.concatenate((Z_real, Z_fake), axis=0)
    # In GAN, this trains the evaluator to separate between real and fake data
    return evaluator.train_on_batch(Y_concat, Z_concat)
