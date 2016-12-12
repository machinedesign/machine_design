import os
from itertools import cycle

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

def train(params, builders={}):
    # Get relevant variables from params
    model_name = params['model']['name']
    model_params = params['model']['params']
    data = params['data']
    report = params['report']
    outdir = report['outdir']

    optim = params['optim']
    max_nb_epochs = optim['max_nb_epochs']
    batch_size = optim['batch_size']
    algo_name = optim['algo_name']
    algo_params = optim['algo_params']
    loss_name = optim['loss']
    lr_schedule = optim['lr_schedule']
    budget_secs = float(optim['budget_secs'])
    lr_schedule_name = lr_schedule['name']
    lr_schedule_params = lr_schedule['params']

    early_stopping = optim['early_stopping']
    early_stopping_name = early_stopping['name']
    early_stopping_params = early_stopping['params']

    checkpoint = optim['checkpoint']

    # Build and compile model
    shapes = get_shapes(pipeline_load(data['train']['pipeline']))
    model = build_model(
        name=model_name, params=model_params,
        shapes=shapes, builders=builders)

    optimizer = build_optimizer(algo_name, algo_params)
    loss = get_loss(loss_name)
    model.compile(loss=loss, optimizer=optimizer)

    # Load data iterators
    train_iterator = cycle(pipeline_load(data['train']['pipeline']))
    nb_train_samples = get_nb_samples(data['train'])
    nb_minibatches = get_nb_minibatches(nb_train_samples, batch_size)

    # Build callbacks
    learning_rate_scheduler = build_lr_schedule_callback(
        name=lr_schedule_name,
        params=lr_schedule_params,
        print=print)

    early_stopping = build_early_stopping_callback(
        name=early_stopping_name,
        params=early_stopping_params,
        print=print)

    model_filename = os.path.join(outdir, 'model.pkl')
    checkpoint = build_model_checkpoint_callback(
        model_filename=model_filename,
        params=checkpoint)

    time_budget = TimeBudget(budget_secs=budget_secs)
    callbacks = [
        learning_rate_scheduler,
        early_stopping,
        checkpoint,
        time_budget
    ]
    callbacks = CallbackList(callbacks)

    for epoch in range(max_nb_epochs):
        stats = {}
        callbacks.on_epoch_begin(epoch, stats=stats)
        for minibatch in range(nb_minibatches):
            train = next(train_iterator)
            X, Y = train['X'], train['Y']
            model.fit(X, Y)
        try:
            callbacks.on_epoch_end(epoch, stats=stats)
        except BudgetFinishedException:
            break

def build_model(name, params, shapes, builders={}):
    model_builder = builders[name]
    model = model_builder(params, shapes) # keras model
    return model

def load(filename):
    pass

def generate(params):
    pass
