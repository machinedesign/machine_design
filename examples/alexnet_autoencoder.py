import os
import numpy as np

from keras.layers import Input
from keras.models import Model

from machinedesign.multi_interface import train
from machinedesign import model_builders
from machinedesign.common import object_to_dict
from machinedesign.common import Normalize
from machinedesign.callbacks import DoEachEpoch
from machinedesign.autoencoder.interface import _report_image_reconstruction
from machinedesign.autoencoder.interface import _report_image_features

from convnetskeras.convnets import convnet

model_builders = object_to_dict(model_builders)

def alexnet_autoencoder(params, input_shape, output_shape):
    """
    WARNING : assumes pixel values are between 0 and 1
    """
    assert input_shape == output_shape
    layer = params['layer_name']
    decoder_name = params['decoder']['name']
    decoder_params = params['decoder']['params']
    freeze_encoder = params['freeze_encoder']

    assert input_shape == (3, 227, 227), 'for "alexnet" shape should be (3, 227, 227), got : {}'.format(input_shape)
    weights_path = "{}/.keras/models/alexnet_weights.h5".format(os.getenv('HOME'))
    assert os.path.exists(weights_path), ('weights path of alexnet {} does not exist, please download manually'
                                         'from http://files.heuritech.com/weights/alexnet_weights.h5 and put it there '
                                         '(see https://github.com/heuritech/convnets-keras)'.format(weights_path))
    full_model = convnet('alexnet',weights_path=weights_path, heatmap=False)
    names = [layer.name for layer in full_model.layers]
    assert layer in names, 'layer "{}" does not exist, available : {}'.format(layer, names)
    encoder = Model(input=full_model.layers[1].input, output=full_model.get_layer(layer).output)
    if freeze_encoder:
        for lay in encoder.layers:
            lay.trainable = False
    decoder = model_builders[decoder_name](decoder_params, encoder.output_shape[1:], output_shape)
    decoder = Model(input=decoder.layers[1].input, output=decoder.layers[-1].output)
    x = Input(input_shape)
    inp = x
    pixel_mean = np.array([123.68, 116.779, 103.939], dtype='float32')[np.newaxis, :, np.newaxis, np.newaxis]
    x = Normalize(scale=255., bias=-pixel_mean)(x)
    x = encoder(x)
    x = decoder(x)
    out = x
    model = Model(input=inp, output=out)
    return model

model_builders['alexnet_autoencoder'] = alexnet_autoencoder

if __name__ == '__main__':

    models = [
        {
            'name': 'autoencoder',
            'input_col': 'X',
            'output_col': 'X',
            'architecture': {
                'name': 'alexnet_autoencoder',
                'params': {
                    'layer_name': 'dense_1',
                    'freeze_encoder': True,
                    'decoder':{
                        'name': 'fully_connected',
                        'params':{
                            'fully_connected_nb_hidden_units_list': [1000, 1000],
                            'fully_connected_activations': [{'name': 'leaky_relu', 'params':{'alpha': 0.3}}] * 2,
                            'output_activation': 'sigmoid',
                            'input_noise':{
                                'name': 'none',
                                'params': {
                                }
                            }
                        }

                    }
                },
            },
            'evaluators':[
                {
                    'name': 'discriminator',
                    'input_col': 'X',
                    'type': 'discriminator',
                    'architecture': {
                        'name': 'fully_connected',
                        'params': {
                            'fully_connected_nb_hidden_units_list': [100],
                            'fully_connected_activations': [{'name': 'leaky_relu', 'params':{'alpha': 0.3}}],
                            'output_activation': 'sigmoid',
                            'input_noise':{
                                'name': 'none',
                                'params': {
                                }
                            }
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
    train(params, model_builders, callbacks=[DoEachEpoch(_report_image_reconstruction), DoEachEpoch(_report_image_features)])
