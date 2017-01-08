import pytest
import shutil
import numpy as np

from machinedesign.common import WrongModelFamilyException
from machinedesign.autoencoder.interface import train
from machinedesign.autoencoder.interface import _apply_noise
from machinedesign.autoencoder.interface import _apply_binarization

params_test = {
    'family': 'autoencoder',
    'input_col': 'X',
    'output_col': 'X',
    'model': {
        'name': 'fully_connected',
        'params': {
            'nb_hidden_units': [10],
            'activations': ['relu'],
            'output_activation': 'sigmoid',
            'input_noise': {
                'name': 'gaussian',
                'params': {
                    'std': 1
                }
            },
        }
    },
    'data': {
        'train': {
            'pipeline': [
                {"name": "toy", "params": {"nb": 128, "w": 8, "h": 8,
                                           "pw": 2, "ph": 2, "nb_patches": 2, "random_state": 42}},
                {"name": "shuffle", "params": {"random_state": 42}},
                {"name": "normalize_shape", "params": {}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "order", "params": {"order": "th"}}
            ]
        },
        'transformers': [
        ]
    },
    'report': {
        'outdir': 'out',
        'checkpoint': {
            'loss': 'train_mean_squared_error',
            'save_best_only': True
        },
        'metrics': ['mean_squared_error'],
        'callbacks': ['image_reconstruction']
    },
    'optim': {
        'algo': {
            'name': 'adam',
            'params': {'lr': 1e-3}
        },
        'lr_schedule': {
            'name': 'constant',
            'params': {}
        },
        'early_stopping': {
            'name': 'none',
            'params': {}
        },
        'max_nb_epochs': 2,
        'batch_size': 128,
        'pred_batch_size': 128,
        'loss': 'mean_squared_error',
        'budget_secs': 3600,
        'seed': 42
    },
}


def test_family():
    params = {'family': 'none'}
    with pytest.raises(WrongModelFamilyException):
        train(params)


def test_seed():
    p = params_test.copy()
    p['report']['outdir'] = 'tmp'
    p['optim']['seed'] = 1
    model1 = train(params_test)
    shutil.rmtree('tmp')

    p = params_test.copy()
    p['report']['outdir'] = 'tmp'
    p['optim']['seed'] = 1
    model2 = train(params_test)
    shutil.rmtree('tmp')
    assert model1.history_stats == model2.history_stats


def test_apply_noise():
    X = np.ones((1000, 10))
    rng = np.random.RandomState(42)
    Y = _apply_noise('masking', {'proba': 0.5}, X, rng=rng)
    assert np.isclose(Y.mean(axis=0), 0.5, atol=1e-3)


def test_binarization():
    pass
