import pytest

import numpy as np
from keras import backend as K

from machinedesign.callbacks import Callback
from machinedesign.callbacks import LearningRateScheduler
from machinedesign.callbacks import lr_schedule_constant
from machinedesign.callbacks import lr_schedule_decrease_when_stop_improving
from machinedesign.callbacks import lr_schedule_decrease_every
from machinedesign.callbacks import lr_schedule_manual
from machinedesign.callbacks import TimeBudget
from machinedesign.callbacks import BudgetFinishedException


class Store(Callback):

    def __init__(self, caption):
        self.caption = caption
        self.actions = []

    def on_epoch_begin(self, epoch, logs={}):
        self.actions.append('{}_epoch_begin'.format(self.caption))

    def on_epoch_end(self, epoch, logs={}):
        self.actions.append('{}_epoch_end'.format(self.caption))

    def on_batch_begin(self, batch, logs={}):
        self.actions.append('{}_batch_begin'.format(self.caption))

    def on_batch_end(self, batch, logs={}):
        self.actions.append('{}_batch_end'.format(self.caption))


class DummyModel:

    def __init__(self, optimizer):
        self.optimizer = optimizer


class DummyOptimizer:

    def __init__(self, lr=0.1):
        self.lr = K.variable(lr)


def test_learning_rate_callback():
    # decrease when stop improving along with LearningRateScheduler
    name = 'decrease_when_stop_improving'
    params = {
        'patience': 3,
        'loss': 'mse',
        'shrink_factor': 2.
    }
    cb = LearningRateScheduler(name=name, params=params)
    optimizer = DummyOptimizer(lr=0.01)
    model = DummyModel(optimizer=optimizer)
    cb.model = model

    model.history_stats = []
    logs = {'mse': 1}
    cb.on_epoch_end(1, logs=logs)
    assert 'lr' in logs
    assert np.allclose(logs['lr'],  0.01)

    model.history_stats = [{'mse': 1}]
    logs = {'mse': 10}
    cb.on_epoch_end(1, logs=logs)
    assert 'lr' in logs
    assert np.allclose(logs['lr'],  0.01)

    model.history_stats = [{'mse': 1}, {'mse': 2}]
    logs = {'mse': 10}
    cb.on_epoch_end(1, logs=logs)
    assert 'lr' in logs
    assert np.allclose(logs['lr'],  0.01 / 2.)


def test_learning_rate_scheduler_constant():
    assert lr_schedule_constant(0.5) == 0.5
    assert lr_schedule_constant(0) == 0


def test_learning_rate_scheduler_decrease_when_stop_improving():

    for mode in ('min', 'max'):
        assert lr_schedule_decrease_when_stop_improving(
            old_lr=1, patience=5, mode='min', shrink_factor=2.,
            loss_history=[]) == 1
        assert lr_schedule_decrease_when_stop_improving(
            old_lr=1, patience=0, mode='min', shrink_factor=2.,
            loss_history=[]) == 1
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='min', shrink_factor=2.,
        loss_history=[10, 20, 30]) == 0.5
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='min', shrink_factor=2.,
        loss_history=[10, 20, 5]) == 1
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='max', shrink_factor=2.,
        loss_history=[10, 20, 30]) == 1
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='max', shrink_factor=2.,
        loss_history=[10, 20, 5]) == 0.5
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='min', shrink_factor=2.,
        loss_history=[0, 2, 4, 1, 50, 60]) == 0.5
    assert lr_schedule_decrease_when_stop_improving(
        old_lr=1, patience=2, mode='max', shrink_factor=2.,
        loss_history=[0, 2, 4, 1, 50, 60]) == 1


def test_learning_rate_scheduler_decrease_every():
    assert lr_schedule_decrease_every(1, every=5, shrink_factor=2., epoch=0) == 0.5
    assert lr_schedule_decrease_every(1, every=5, shrink_factor=2., epoch=1) == 1
    assert lr_schedule_decrease_every(1, every=5, shrink_factor=2., epoch=5) == 0.5
    assert lr_schedule_decrease_every(1, every=5, shrink_factor=2., epoch=10) == 0.5
    assert lr_schedule_decrease_every(1, every=1, shrink_factor=2., epoch=0) == 0.5
    assert lr_schedule_decrease_every(1, every=1, shrink_factor=2., epoch=1) == 0.5
    assert lr_schedule_decrease_every(1, every=1, shrink_factor=2., epoch=100) == 0.5
    assert lr_schedule_decrease_every(1, every=0, shrink_factor=2., epoch=0) == 1
    assert lr_schedule_decrease_every(1, every=0, shrink_factor=2., epoch=1) == 1
    assert lr_schedule_decrease_every(1, every=0, shrink_factor=2., epoch=100) == 1


def test_learning_rate_scheduler_manual():
    assert lr_schedule_manual(1, schedule=[], epoch=0) == 1
    assert lr_schedule_manual(1, schedule=[{'range': (0, 1), 'lr': 0.5}], epoch=0) == 0.5
    assert lr_schedule_manual(0, schedule=[{'range': (0, 1), 'lr': 0.5}], epoch=0) == 0.5
    assert lr_schedule_manual(0, schedule=[{'range': (0, 1), 'lr': 0.5}], epoch=1) == 0.5
    assert lr_schedule_manual(0, schedule=[{'range': (0, 1), 'lr': 0.5}], epoch=2) == 0

    s = [
        {'range': (0, 10), 'lr': 0.5},
        {'range': (11, 20), 'lr': 0.1},
    ]
    for e in range(0, 11):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.5
    for e in range(11, 21):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.1
    s = [
        {'range': (5, 8), 'lr': 0.01},
        {'range': (0, 10), 'lr': 0.5},
        {'range': (11, 20), 'lr': 0.1},
    ]
    for e in (0, 1, 2, 3, 4, 9, 10):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.5
    for e in (5, 6, 7, 8):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.01
    for e in range(11, 21):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.1

    s = [
        {'range': (0, 10), 'lr': 0.5},
        {'range': (5, 8), 'lr': 0.01},
        {'range': (11, 20), 'lr': 0.1},
    ]
    for e in (0, 1, 2, 3, 4, 9, 10):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.5
    for e in (5, 6, 7, 8):
        assert lr_schedule_manual(0, schedule=s, epoch=e) == 0.5


def test_time_budget():
    clock = 0

    def fake_time():
        return clock
    with pytest.raises(BudgetFinishedException):
        TimeBudget(budget_secs=0, time=fake_time).on_epoch_end(epoch=1)

    clock = 0
    cb = TimeBudget(budget_secs=100, time=fake_time)
    clock = 99
    cb.on_epoch_end(epoch=1)

    TimeBudget(budget_secs=100, time=fake_time).on_epoch_end(epoch=1)

    with pytest.raises(BudgetFinishedException):
        clock = 0
        cb = TimeBudget(budget_secs=3600, time=fake_time)
        clock = 3600
        cb.on_epoch_end(epoch=1)

    with pytest.raises(BudgetFinishedException):
        clock = 0
        cb = TimeBudget(budget_secs=3600, time=fake_time)
        clock = 3601
        cb.on_epoch_end(epoch=1)
