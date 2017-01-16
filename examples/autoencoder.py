import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

from machinedesign.autoencoder.interface import train
from machinedesign.autoencoder.interface import generate


def main():
    params = {
        'family': 'autoencoder',
        'input_col': 'X',
        'output_col': 'X',
        'model': {
            'name': 'fully_connected',
            'params': {
                'nb_hidden_units': [256],
                'activations': ['relu'],
                'output_activation': 'sigmoid',
            }
        },
        'data': {
            'train': {
                'pipeline': [
                    {"name": "toy",
                     "params": {"nb": 512, "w": 16, "h": 16,
                                "pw": 4, "ph": 4,
                                "nb_patches": 2, "random_state": None}},
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
            'callbacks': ['image_reconstruction', 'image_features']
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
                'name': 'basic',
                'params': {
                    'patience_loss': 'train_mean_squared_error',
                    'patience': 5
                }
            },
            'max_nb_epochs': 200,
            'batch_size': 128,
            'pred_batch_size': 128,
            'loss': 'binary_crossentropy',
            'budget_secs': 3600,
            'seed': 42
        }
    }
    train(params)
    params = {
        'model': {
            'folder': 'out'
        },
        'method': {
            'name': 'iterative_refinement',
            'params': {
                'seed': 42,
                'batch_size': 128,
                'nb_samples': 256,
                'nb_iter': 100,
                'binarize': {
                    'name': 'none',
                    'params': {
                    }
                },
                'noise': {
                    'name': 'gaussian',
                    'params': {'std': 0.1}
                },
                'stop_if_unchanged': True,
                'seed': 42
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
