import os
import numpy as np
from six.moves import map

from keras.layers import Input
from keras.models import Model

from machinedesign.multi_interface import train
from machinedesign import model_builders
from machinedesign.common import object_to_dict
from machinedesign.common import Normalize

from machinedesign.callbacks import DoEachEpoch
from machinedesign.autoencoder.interface import _report_image_reconstruction
from machinedesign.autoencoder.interface import _report_image_features
from machinedesign.data import pipeline_load
from machinedesign.data import batch_iterator

from convnetskeras.convnets import convnet

model_builders = object_to_dict(model_builders)

def build_alexnet_model(layer, input_shape=(3, 227, 227)):
    """
    WARNING : assumes pixel values are between 0 and 1
    """
    assert input_shape == (3, 227, 227), 'for "alexnet" shape should be (3, 227, 227), got : {}'.format(input_shape)
    weights_path = "{}/.keras/models/alexnet_weights.h5".format(os.getenv('HOME'))
    assert os.path.exists(weights_path), ('weights path of alexnet {} does not exist, please download manually'
                                         'from http://files.heuritech.com/weights/alexnet_weights.h5 and put it there '
                                         '(see https://github.com/heuritech/convnets-keras)'.format(weights_path))
    full_model = convnet('alexnet',weights_path=weights_path, heatmap=False)
    names = [layer.name for layer in full_model.layers]
    assert layer in names, 'layer "{}" does not exist, available : {}'.format(layer, names)
    encoder = Model(input=full_model.layers[1].input, output=full_model.get_layer(layer).output)
    x = Input(input_shape)
    inp = x
    pixel_mean = np.array([123.68, 116.779, 103.939], dtype='float32')[np.newaxis, :, np.newaxis, np.newaxis]
    x = Normalize(scale=255., bias=-pixel_mean)(x)
    x = encoder(x)
    out = x
    model = Model(input=inp, output=out)
    return model

def build_data_generator(pipeline, cols='all'):
    model = build_alexnet_model('dense_1')
    def _apply(data):
        data['h'] = model.predict(data['X'])
        return data
    def _gen(batch_size, repeat=False):
        it = pipeline_load(pipeline)
        it = batch_iterator(it, batch_size=batch_size, repeat=repeat, cols=cols)
        it = map(_apply, it)
        return it
    return _gen

if __name__ == '__main__':
    leaky_relu = {'name': 'leaky_relu', 'params':{'alpha': 0.3}}
    models = [
        {
            'name': 'autoencoder',
            'input_col': 'h',
            'output_col': 'X',
            'architecture': {
                'name': 'fc_upconvolutional',
                'params':{
                    'fully_connected': {
                        'nb_hidden_units': [512],
                        'activations': [leaky_relu]
                    },
                    'reshape': [8, 8, 8],
                    'upconvolutional':{
                        'nb_filters': [16, 16, 32, 32, 64],
                        'filter_sizes': [5, 5, 5, 5, 5],
                        'activations': [leaky_relu] * 5,
                        'stride': 2
                    },
                    'output_activation': 'sigmoid',
                },
            },
            'evaluators':[
                {
                    'name': 'discriminator',
                    'input_col': 'X',
                    'type': 'discriminator',
                    'architecture': {
                        'name': 'convolutional',
                        'params':{
                            'nb_filters': [16, 16, 32, 32, 64],
                            'filter_sizes': [5, 5, 5, 5, 5],
                            'activations': [leaky_relu] * 5,
                            'output_activation': 'sigmoid',
                            'output_filter_size': 1,
                            'stride': 2
                        },
                    },
                    'optimizer':{
                        'name': 'adam',
                        'params':{
                            'lr': 0.0002
                        }
                    },
                    'losses': [
                        {'name': 'binary_crossentropy', 'coef': 1, 'params': {}}
                    ],
                    'callbacks':{
                        'lr_schedule':{
                            'name': 'constant',
                            'params': {}
                        },
                        'metrics': {'names': ['binary_crossentropy'], 'pred_batch_size': 128},
                    }
                }
            ],
            'optimizer':{
                'name': 'adam',
                'params':{
                    'lr': 0.0002
                }
            },
            'losses':[
                {'name': 'binary_crossentropy', 'coef': 1, 'params': {}},
                {'name': 'discriminator', 'coef': 0.01, 'params': {}}
            ],
            'callbacks':{
                'lr_schedule':{
                    'name': 'constant',
                    'params': {}
                },
                'metrics': {'names': ['binary_crossentropy'], 'pred_batch_size': 128},
            }
        },

    ]
    data = {
        'train':{
            'pipeline':[
                {"name": "imagefilelist", "params": {"pattern": "{gametiles}"}},
                {"name": "shuffle", "params": {}},
                {"name": "imageread", "params": {}},
                {"name": "normalize_shape", "params": {}},
                {"name": "force_rgb", "params": {}},
                {"name": "resize", "params": {"shape": [227, 227]}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "order", "params": {"order": "th"}},
            ],

        }
    }
    callbacks ={
        'checkpoint':{
            'loss': 'autoencoder_train_binary_crossentropy',
            'save_best_only': True
        },
        'budget_secs': 1600,
        'early_stopping':{
            'name': 'basic',
            'params': {
                'patience_loss': 'autoencoder_train_binary_crossentropy',
                'patience': 10
            }
        },
        'outdir': 'out'
    }
    params = {'models': models, 'data': data, 'optim': {'batch_size': 128, 'max_nb_epochs': 100, 'seed': 42}, 'callbacks': callbacks}
    train(params, model_builders,
          callbacks=[DoEachEpoch(_report_image_reconstruction), DoEachEpoch(_report_image_features)],
          build_data_generator=build_data_generator)
