import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

from machinedesign.autoencoder.interface import train
from machinedesign.autoencoder.interface import generate

def main():
    params = {
        'family': 'autoencoder',
        'model': {
            'name': 'fully_connected',
            'params':{
                'fully_connected_nb_hidden_units_list': [1000, 1000],
                'fully_connected_activations': [
                    {'name': 'ksparse', 'params':{'zero_ratio': 0.1}},
                    {'name': 'ksparse', 'params':{'zero_ratio': 0.1}}
                ],
                'output_activation': 'sigmoid'
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "imagefilelist", "params": {"pattern": "{gametiles}"}},
                    {"name": "shuffle", "params": {}},
                    {"name": "imageread", "params": {}},
                    {"name": "normalize_shape", "params": {}},
                    {"name": "force_rgb", "params": {}},
                    {"name": "resize", "params": {"shape": [16, 16]}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "order", "params": {"order": "th"}}
                ]
            },
            'transformers':[
            ]
        },
        'report':{
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_mean_squared_error',
                'save_best_only': True
            },
            'metrics': ['mean_squared_error'],
            'domain_specific': ['image_reconstruction', 'image_features']
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
                'name': 'basic',
                'params': {
                    'patience_loss': 'train_mean_squared_error',
                    'patience': 5
                }
            },
            'max_nb_epochs': 20,
            'batch_size': 128,
            'pred_batch_size': 128,
            'loss': 'mean_squared_error',
            'budget_secs': 3600,
            'seed': 42
        },
    }
    train(params)
    params = {
        'model':{
            'folder': 'out'
        },
        'method':{
            'name': 'iterative_refinement',
            'params': {
                'batch_size': 128,
                'nb_samples': 10,
                'nb_iter': 10,
                'binarize':{
                    'name': 'none',
                    'params': {
                    }
                },
                'noise':{
                    'name': 'none',
                    'params': {}
                }
            },
            'save_folder': 'gen'
        }
    }
    generate(params)
    data = np.load('gen/generated.npz')
    X = data['generated']
    img = grid_of_images_default(X)
    imsave('samples.png', img)

if __name__ == '__main__':
    main()
