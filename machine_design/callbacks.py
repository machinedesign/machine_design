import numpy as np
import time

import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

eps = 1e-8

class Dummy(Callback):
    pass

class CallbackContainer(Callback):

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
    # TODO factorize this
    # TODO document this
    def __init__(self, name='decrease_when_stop_improving',
                 params=None, print=print, eps=eps):
        self.schedule_params = params if params else {}
        self.name = name
        self.print = print
        self.eps = eps

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
        elif self.name == 'decrease_when_stop_improving':
            patience = params['patience']
            mode = params.get('mode', 'auto')
            loss = params['loss']
            shrink_factor = params['shrink_factor']
            if epoch < patience:
                new_lr = old_lr
            else:
                hist = model.history.history
                value_epoch = logs[loss]
                if mode == 'auto':
                    best = max if 'acc' in loss else min
                else:
                    best = {'max': max, 'min': min}[mode]
                arg_best = np.argmax if best == max else np.argmin
                best_index = arg_best(hist[loss])
                best_value = hist[loss][best_index]
                if ( best(value_epoch, best_value) == best_value and
                     epoch - best_index + 1 >= patience):
                    self.print('shrinking learning rate, loss : {},'
                          'prev best epoch : {}, prev best value : {},'
                          'current value: {}'.format(loss,
                                                     best_index + 1, best_value,
                                                     value_epoch))
                    new_lr = old_lr / shrink_factor
                else:
                    new_lr = old_lr
        elif self.name == 'decrease_every':
            every = params['every']
            shrink_factor = params['shrink_factor']
            if epoch % (every) == 0:
                new_lr = old_lr / shrink_factor
            else:
                new_lr = old_lr
        elif self.name == 'cifar':
            # source : https://github.com/gcr/torch-residual-networks/blob/master/train-cifar.lua#L181-L187
            if epoch == 80:
                new_lr = old_lr / 10.
            elif epoch == 120:
                new_lr = old_lr / 10.
            else:
                new_lr = old_lr
        elif self.name == 'manual':
            schedule = params['schedule']
            new_lr = old_lr
            for s in schedule:
                first, last = s['range']
                lr = s['lr']
                if epoch >= first and epoch <= last:
                    new_lr = lr
                    break
        else:
            raise Exception('Unknown lr schedule : {}'.format(self.name))
        min_lr = params.get('min_lr', 0)
        new_lr = max(new_lr, min_lr)
        if abs(new_lr - old_lr) > eps:
            self.print('prev learning rate : {}, '
                  'new learning rate : {}'.format(old_lr, new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr


class BudgetFinishedException(Exception):
    pass


class TimeBudget(Callback):

    def __init__(self, budget_secs=float('inf')):
        self.start = time.time()
        self.budget_secs = budget_secs

    def on_epoch_end(self, epoch, logs={}):
        t = time.time()
        if t - self.start >= self.budget_secs:
            raise BudgetFinishedException()

class RecordEachEpoch(Callback):

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

    def __init__(self, func):
        self.func = func

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        self.func(self)

def build_early_stopping_callback(name, params, outdir='out', model=None):
    if name == 'basic':
        patience_loss = params['patience_loss']
        patience = params['patience']
        callback = EarlyStopping(monitor=patience_loss,
                                 patience=patience,
                                 verbose=1,
                                 mode='auto')
        callback.model = model
        return callback
    elif name == 'none':
        return Dummy()

def build_model_checkpoint_callback(params, model_filename='model.pkl', model=None):
    loss = params['loss']
    save_best_only = params['save_best_only']
    callback = ModelCheckpoint(
        model_filename,
        monitor=loss,
        verbose=1,
        save_best_only=save_best_only,
        mode='auto' if loss else 'min')
    callback.model = model
    return callback

def build_lr_schedule_callback(name, params, print=print, model=None):
    callback = LearningRateScheduler(name=name, params=params, print=print)
    callback.model = model
    return callback
