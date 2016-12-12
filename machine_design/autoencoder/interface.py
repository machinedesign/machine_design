import os
import logging
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
    report_callbacks = [DoEachEpoch(report_reconstruction)]
    return train_basic(
        params,
        builders=model_builders,
        inputs='X', outputs='X',
        logger=logger,
        callbacks=report_callbacks)

def report_reconstruction(cb):
    model = cb.model
    data_iterators = cb.data_iterators
    params = cb.params
    epoch = cb.epoch
    data = next(data_iterators['train'].flow(batch_size=128))
    X = data['X']
    X_rec = model.predict(X)
    img = _get_input_reconstruction_grid(X, X_rec)
    folder = os.path.join(params['report']['outdir'], 'recons')
    mkdir_path(folder)
    filename = os.path.join(folder, '{:05d}.png'.format(epoch))
    imsave(filename, img)

def _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images):
    X = grid_of_images(X)
    X_rec = grid_of_images(X_rec)
    img = horiz_merge(X, X_rec)
    return img

def load(folder):
    pass

def generate(params):
    pass

def main():
    params = {
        'model': {
            'name': 'fullyconnected',
            'params':{
                'fully_connected_nb_hidden_units_list': [500],
                'fully_connected_activation': 'relu',
                'output_activation': 'sigmoid'
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "toy", "params": {"nb": 256, "w": 16, "h": 16, "pw": 4, "ph": 4, "nb_patches": 2}},
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
                'loss': 'train_mse',
                'save_best_only': True
            },
            'metrics': ['mean_squared_error']
        },
        'optim':{
            'algo': {
                'name': 'adadelta',
                'params': {'lr': 0.1}
            },
            'lr_schedule':{
                'name': 'constant',
                'params': {}
            },
            'early_stopping':{
                'name': 'none',
                'params': {}
            },
            'max_nb_epochs': 100,
            'batch_size': 128,
            'loss': 'mean_squared_error',
            'budget_secs': 3600,
            'seed': 42
        },
    }
    train(params)

if __name__ == '__main__':
    main()
