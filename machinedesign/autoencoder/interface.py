import numpy as np
import os
import logging
from functools import partial

from skimage.io import imsave

from ..interface import train as train_basic
from ..common import object_to_dict
from ..common import mkdir_path
from ..viz import grid_of_images
from ..viz import horiz_merge
from ..callbacks import DoEachEpoch
from . import model_builders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_builders = object_to_dict(model_builders)

def train(params):
    report_callbacks = [DoEachEpoch(_report_reconstruction)]
    return train_basic(
        params,
        builders=model_builders,
        inputs='X', outputs='X',
        logger=logger,
        callbacks=report_callbacks)

def load(folder):
    pass

def generate(params):
    pass

def _report_reconstruction(cb):
    model = cb.model
    data_iterators = cb.data_iterators
    params = cb.params
    epoch = cb.epoch
    data = next(data_iterators['train'].flow(batch_size=128))
    X = data['X']
    X_rec = model.predict(X)
    X = np.clip(X, 0, 1)
    X_rec = np.clip(X_rec, 0, 1)
    grid_of_images_ = partial(grid_of_images, border=1, bordercolor=(0.3, 0, 0))
    img = _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images_)
    folder = os.path.join(params['report']['outdir'], 'recons')
    mkdir_path(folder)
    filename = os.path.join(folder, '{:05d}.png'.format(epoch))
    imsave(filename, img)


def _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images):
    X = grid_of_images(X)
    X_rec = grid_of_images(X_rec)
    img = horiz_merge(X, X_rec)
    return img

def main():
    params = {
        'model': {
            'name': 'fully_connected',
            'params':{
                'fully_connected_nb_hidden_units_list': [100],
                'fully_connected_activation': 'relu',
                'output_activation': 'sigmoid'
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "toy", "params": {"nb": 256, "w": 8, "h": 8, "pw": 2, "ph": 2, "nb_patches": 2, "random_state": 42}},
                    {"name": "shuffle", "params": {}},
                    {"name": "normalize_shape", "params": {}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "order", "params": {"order": "th"}}
                ]
            }
        },
        'report':{
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_mean_squared_error',
                'save_best_only': True
            },
            'metrics': ['mean_squared_error']
        },
        'optim':{
            'algo': {
                'name': 'adam',
                'params': {'lr': 1e-3}
            },
            'lr_schedule':{
                'name': 'constant',
                'params': {}
            },
            'early_stopping':{
                'name': 'none',
                'params': {}
            },
            'max_nb_epochs': 40,
            'batch_size': 128,
            'loss': 'mean_squared_error',
            'budget_secs': 3600,
            'seed': 42
        },
    }
    train(params)

if __name__ == '__main__':
    main()
