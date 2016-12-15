import pytest

from machinedesign.callbacks import Callback
from machinedesign.callbacks import CallbackContainer
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

def test_callback_container():
    cb1, cb2 = Store(caption=1), Store(caption=2)
    callbacks = [cb1, cb2]
    callbacks = CallbackContainer(callbacks)

    callbacks.on_epoch_begin(1, logs={'a': 5})
    assert cb1.actions == ['1_epoch_begin']
    assert cb2.actions == ['2_epoch_begin']

    callbacks.on_epoch_end(1, logs={'a': 5})
    assert cb1.actions == ['1_epoch_begin', '1_epoch_end']
    assert cb2.actions == ['2_epoch_begin', '2_epoch_end']

    callbacks.on_batch_begin(1, logs={'a': 5})
    assert cb1.actions == ['1_epoch_begin', '1_epoch_end', '1_batch_begin']
    assert cb2.actions == ['2_epoch_begin', '2_epoch_end', '2_batch_begin']

    callbacks.on_batch_end(1, logs={'a': 5})
    assert cb1.actions == ['1_epoch_begin', '1_epoch_end', '1_batch_begin', '1_batch_end']
    assert cb2.actions == ['2_epoch_begin', '2_epoch_end', '2_batch_begin', '2_batch_end']

    callbacks = CallbackContainer([])
    callbacks.on_epoch_begin(1, logs={'a': 5})
    callbacks.on_epoch_end(1, logs={'a': 5})
    callbacks.on_batch_begin(1, logs={'a': 5})
    callbacks.on_batch_end(1, logs={'a': 5})

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
    fake_time = lambda: clock
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
