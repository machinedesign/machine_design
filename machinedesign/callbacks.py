"""
This module contains a list of callback classes and some
helpers used commonly in models.
"""
import numpy as np
import time

import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

__all__ = [
    "CallbackContainer",
    "LearningRateScheduler",
    "TimeBudget",
    "RecordEachEpoch",
    "DoEachEpoch",
    "build_early_stopping_callback",
    "build_model_checkpoint_callback",
    "build_lr_schedule_callback",
    "BudgetFinishedException"
]

class Dummy(Callback):
    pass

class CallbackContainer(Callback):
    """
    a callback class that can contain a list of callback instances.

    Parameters
    ----------
        callbacks : list of Callback
    """
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_begin(self, epoch, logs={}):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs={}):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)


class LearningRateScheduler(Callback):
    """
    Callback for learning rate scheduling

    Parameters:
    -----------

    name: str
        type of learning rate schedule
        available are :
            - 'constant'
            - 'decrease_when_stop_improving'
            - 'decrease_every'
            - 'manual'
    params: dict
        parameters of learning rate schedule.

    print: callable(default=print)
        function to report changes in learning rate

    the following are the definitions and
    parameters needed for each learning rate
    schedule type.

    `constant`:
        a constant learning rate schedule.
        no parameters are needed

    `decrease_when_stop_improving`:
        divide the learning rate by `shrink_factor` if the `loss`
        has not improved since `patience` epochs.

        Parameters:
            shrink_factor: float
                divide the `learning_rate` by it in case of no improvements.
            patience: int
                number of epochs without improvements of `loss` to wait
            mode: str
                'auto' or 'max' or 'min'
                used to know whether we want
            loss: str
                the stat to use for checking improvements

    `decrease_every`:
        divide the  learning rate by `shrink_factor` periodically

        Parameters
            shrink_factor: float
                divide the `learning_rate` by.
            every: int
                the size of the period
    `manual`:
        manual schedule

        Parameters
            schedule : list of dicts
                 each dict has two keys, `range` and `lr`.
                `range` is a tuple (start, end) definign an interval.
                `lr` is the learning rate used in the interval defined
                 by `range`.
    """
    def __init__(self, name='decrease_when_stop_improving',
                 params=None, print=print):
        ""
        self.schedule_params = params if params else {}
        self.name = name
        self.print = print

    def on_epoch_end(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'

        params = self.schedule_params
        model = self.model
        old_lr = float(model.optimizer.lr.get_value())
        if epoch == 0:
            new_lr = old_lr
        elif self.name == 'constant':
            new_lr = old_lr
            new_lr = lr_schedule_constant(old_lr)
        elif self.name == 'decrease_when_stop_improving':
            patience = params['patience']
            mode = params.get('mode', 'auto')
            loss = params['loss']
            shrink_factor = params['shrink_factor']
            if mode == 'auto':
                mode = 'max' if 'acc' in loss else 'min'
            new_lr = lr_schedule_decrease_when_stop_improving(
                old_lr,
                patience=patience,
                mode=mode,
                shrink_factor=shrink_factor,
                loss_history=model.history.history[loss] + [logs[loss]])
        elif self.name == 'decrease_every':
            every = params['every']
            shrink_factor = params['shrink_factor']
            new_lr = lr_schedule_decrease_every(
                old_lr,
                every=every,
                shrink_factor=shrink_factor,
                epoch=epoch)
        elif self.name == 'manual':
            schedule = params['schedule']
            new_lr = lr_schedule_manual(old_lr, schedule=schedule, epoch=epoch)
        else:
            raise ValueError('Unknown lr schedule : {}'.format(self.name))
        min_lr = params.get('min_lr', 0)
        new_lr = max(new_lr, min_lr)
        if not np.isclose(new_lr, old_lr):
            self.print('Learning rate changed.')
            self.print('prev learning rate : {}, new learning rate : {}'.format(old_lr, new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr

def lr_schedule_constant(old_lr):
    return old_lr

def lr_schedule_decrease_when_stop_improving(old_lr,
                                             patience,
                                             mode,
                                             shrink_factor,
                                             loss_history):
    """
    divide the learning rate by `shrink_factor` if the `loss`
    has not improved since `patience` epochs.

    Parameters
    ----------
        old_lr : float
            the old learning rate
        patience: int
            number of epochs without improvements of `loss` to wait
        mode: str
            'auto' or 'max' or 'min'
            used to know whether we want maximize or minimize
        shrink_factor: float
            divide the `learning_rate` by it in case of no improvements.
        loss_history: list of scalar
            losses until the current iteration
    """
    epoch = len(loss_history)
    if epoch < patience:
        new_lr = old_lr
    elif len(loss_history) == 0:
        new_lr = old_lr
    else:
        cur_value = loss_history[-1]
        max_or_min = {'max': max, 'min': min}[mode]
        arg_best = np.argmax if max_or_min == max else np.argmin
        best_index = arg_best(loss_history[0:-1])
        best_value = loss_history[best_index]
        not_improved = max_or_min(cur_value, best_value) == best_value
        out_of_patience = (epoch - best_index + 1) >= patience
        if (not_improved and out_of_patience):
            new_lr = old_lr / shrink_factor
        else:
            new_lr = old_lr
    return new_lr

def lr_schedule_decrease_every(old_lr, every, shrink_factor, epoch):
    """
    divide the  learning rate by `shrink_factor` periodically

    Parameters
    ----------
        shrink_factor: float
            divide the `learning_rate` by it.
        every: int
            the length of the period
    """
    if every == 0:
        new_lr = old_lr
    elif epoch % (every) == 0:
        new_lr = old_lr / shrink_factor
    else:
        new_lr = old_lr
    return new_lr

def lr_schedule_manual(old_lr, schedule, epoch):
    """
    manual schedule

    Parameters
    ----------
        schedule : list of dicts
             each dict has two keys, `range` and `lr`.
            `range` is a tuple (start, end) defining an interval,
            start and end are included in the interval.
            `lr` is the learning rate used in the interval defined
             by `range`.

        epoch : int
            Epoch number
    """
    new_lr = old_lr
    for s in schedule:
        first, last = s['range']
        lr = s['lr']
        if epoch >= first and epoch <= last:
            new_lr = lr
            break
    return new_lr

class TimeBudget(Callback):
    """
    a time budget callback that raises BudgetFinishedException() when
    the time budget is reached.

    Parameters
    ----------

    budget_secs: int
        budget in secs
    """
    def __init__(self, budget_secs=float('inf'), time=time.time):
        self.start = time()
        self.time = time
        self.budget_secs = budget_secs

    def on_epoch_end(self, epoch, logs={}):
        t = self.time()
        if t - self.start >= self.budget_secs:
            raise BudgetFinishedException()

class RecordEachEpoch(Callback):
    """
    record a stat each epoch

    Parameters
    ----------

    name : str
        name of the stat to record
    compute_fn: callable
        called as `compute_fn()` each epoch to get a value.
    """
    def __init__(self, name, compute_fn, on_logs=True):
        self.name = name
        self.compute_fn = compute_fn
        self.values = []
        self.on_logs = on_logs

    def on_epoch_end(self, batch, logs={}):
        val = self.compute_fn()
        if self.on_logs:
            logs[self.name] = val
        self.values.append(val)

class DoEachEpoch(Callback):

    """
    do something each epoch
    Parameters
    ----------

    func: callable
        called as `func(self)` each epoch.
    """
    def __init__(self, func):
        self.func = func

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        self.func(self)

def build_early_stopping_callback(name, params, outdir='out'):
    """
    Helper to build EarlyStopping callback

    Parameters
    ----------

    name: str
        'basic' or 'none'.

    params: dict
        If name='basic':
            'patience_loss': str
                loss to use for early stopping
            'patience': int
                number of epochs to wait without improvements
        If name='none':
            no params are neededs

    Returns
    -------

    EarlyStopping instance

    """
    if name == 'basic':
        patience_loss = params['patience_loss']
        patience = params['patience']
        callback = EarlyStopping(monitor=patience_loss,
                                 patience=patience,
                                 verbose=1,
                                 mode='auto')
        return callback
    elif name == 'none':
        return Dummy()

def build_model_checkpoint_callback(params, model_filename='model.pkl'):
    """
    Helper to build ModelCheckpoint callback

    Parameters
    ----------

    params : dict
        parameters of ModelCheckpoint
        'loss': str
            loss to use in case `save_best_only` is True for saving
            the model at the best epoch
        'save_best_only': bool
            if True, save the model with the best `loss`, otherwise
            otherwise always save the model.
    model_filename: str
        filename where to save the model
    model: keras model
        the model

    Returns
    -------

    ModelCheckpoint instance

    """
    loss = params['loss']
    save_best_only = params['save_best_only']
    callback = ModelCheckpoint(
        model_filename,
        monitor=loss,
        verbose=1,
        save_best_only=save_best_only,
        mode='auto' if loss else 'min')
    return callback

def build_lr_schedule_callback(name, params, print=print):
    """
    Helper to build LearningRateSchedule callback

    Parameters
    ----------
    name : str
        type of lr schedule (refer exactly to `name` in `LearningRateScheduler`)
    params : dict
        params of lr schedule (refer exactly to `params` in `LearningRateScheduler`)
    """
    callback = LearningRateScheduler(name=name, params=params, print=print)
    return callback

class BudgetFinishedException(Exception):
    """raised when the time budget is reached"""
    pass
