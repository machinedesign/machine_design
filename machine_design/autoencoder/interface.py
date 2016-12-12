import logging

from ..interface import train as train_basic
from ..common import object_to_dict
from . import model_builders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_builders = object_to_dict(model_builders)

def train(params):
    return train_basic(params, builders=model_builders, inputs='X', outputs='X', logger=logger)

def load(folder):
    pass

def generate(params):
    pass

def main():
    params = {
        'model': {
            'name': 'fullyconnected',
            'params':{
                'fully_connected_nb_hidden_units_list': [100],
                'fully_connected_activation': 'relu',
                'output_activation': 'sigmoid'
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "toy", "params": {"nb": 1000, "w": 16, "h": 16, "pw": 4, "ph": 4, "nb_patches": 2}},
                    {"name": "shuffle", "params": {}},
                    {"name": "normalize_shape", "params": {}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "order", "params": {"order": "th"}}            ]
            }
        },
        'report':{
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_mse',
                'save_best_only': True
            }
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
            'max_nb_epochs': 10,
            'batch_size': 128,
            'loss': 'mse',
            'budget_secs': 3600
        },
    }
    train(params)

if __name__ == '__main__':
    main()
