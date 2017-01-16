"""
This module contains a list of callback classes and some
helpers used commonly in models.
"""
from __future__ import print_function
import numpy as np
import time
import warnings

import keras.backend as K
from keras.callbacks import Callback


class Dummy(Callback):
    pass


class EarlyStopping(Callback):
    '''

    This is pasted from keras code to launch a StopTrainingException
    when early stopping is detected instead of modifying self.model.training
    to True in keras.

    Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    '''

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available! available stats are : %s.Skipping.' %
                          (self.monitor, logs.keys()), RuntimeWarning)
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                raise StopTrainingException()
            self.wait += 1

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


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

    print_func: callable(default=print)
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
                 params=None, print_func=print):
        ""
        self.schedule_params = params if params else {}
        self.name = name
        self.print_func = print

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
            hist = [stat[loss] for stat in model.history_stats] + [logs[loss]]
            new_lr = lr_schedule_decrease_when_stop_improving(
                old_lr,
                patience=patience,
                mode=mode,
                shrink_factor=shrink_factor,
                loss_history=hist)
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
            self.print_func('Learning rate changed.')
            self.print_func(
                'prev learning rate : {}, new learning rate : {}'.format(old_lr, new_lr))
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
    each : int
        record each 'each' epochs
    """

    def __init__(self, name, compute_fn, each=1, on_logs=True):
        self.name = name
        self.compute_fn = compute_fn
        self.values = []
        self.each = each
        self.on_logs = on_logs

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.each) == 0:
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


class ModelsCheckpoint(Callback):
    '''
    this is pasted from keras code with few modifications to enable
    saving several models at the checkpoint instead of one.

    Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        models : list of keras Model
        filepaths: string, paths where to save the models.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).

    '''

    def __init__(self, models, filepaths, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto'):
        super(ModelsCheckpoint, self).__init__()
        self.models = models
        self.filepaths = filepaths
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping. available stats are : %s' % (self.monitor, logs.keys()), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model(s).'
                              % (epoch, self.monitor, self.best,
                                 current))
                    self.best = current
                    if self.save_weights_only:
                        for model, filepath in zip(self.models, self.filepaths):
                            model.save_weights(filepath, overwrite=True)
                    else:
                        for model, filepath in zip(self.models, self.filepaths):
                            model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model(s)' % (epoch,))
            if self.save_weights_only:
                for model, filepath in zip(self.models, self.filepaths):
                    model.save_weights(filepath, overwrite=True)
            else:
                for model, filepath in zip(self.models, self.filepaths):
                    model.save(filepath, overwrite=True)


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


def build_models_checkpoint_callback(params, models, filepaths):
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
    models: list of keras Model
        models to consider
    filepaths : list of str of filenames
        filenames where to save the models
    Returns
    -------

    ModelsCheckpoint instance

    """
    loss = params['loss']
    save_best_only = params['save_best_only']
    callback = ModelsCheckpoint(
        models,
        filepaths,
        monitor=loss,
        verbose=1,
        save_best_only=save_best_only,
        mode='auto' if loss else 'min')
    return callback


def build_lr_schedule_callback(name, params, print_func=print):
    """
    Helper to build LearningRateSchedule callback

    Parameters
    ----------
    name : str
        type of lr schedule (refer exactly to `name` in `LearningRateScheduler`)
    params : dict
        params of lr schedule (refer exactly to `params` in `LearningRateScheduler`)
    """
    callback = LearningRateScheduler(name=name, params=params, print_func=print)
    return callback


class BudgetFinishedException(Exception):
    """raised when the time budget is reached"""
    pass


class StopTrainingException(Exception):
    """raised when early stopping asks to stop training"""
    pass
