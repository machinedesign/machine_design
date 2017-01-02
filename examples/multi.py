if __name__ == '__main__':
    from machinedesign.multi_interface import train
    from machinedesign import model_builders
    from machinedesign.common import object_to_dict
    from machinedesign.callbacks import DoEachEpoch
    from machinedesign.autoencoder.interface import _report_image_reconstruction, _report_image_features
    model_builders = object_to_dict(model_builders)
    models = [
        {
            'name': 'autoencoder',
            'input_col': 'X',
            'output_col': 'X',
            'architecture': {
                'name': 'fully_connected',
                'params': {
                    'fully_connected_nb_hidden_units_list': [500],
                    'fully_connected_activations': [{'name': 'leaky_relu', 'params':{'alpha': 0.3}}],
                    'output_activation': 'sigmoid',
                    'input_noise':{
                        'name': 'none',
                        'params': {
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
                            'fully_connected_nb_hidden_units_list': [50],
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
                            'lr': 1e-3
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
                        'metrics': {'names': ['mean_squared_error'], 'pred_batch_size': 128},
                    }
                }
            ],
            'optimizer':{
                'name': 'adam',
                'params':{
                    'lr': 1e-2
                }
            },
            'losses':[
                {'name': 'binary_crossentropy', 'coef': 1, 'params': {}},
                {'name': 'discriminator', 'coef': 1, 'params': {}}
            ],
            'callbacks':{
                'lr_schedule':{
                    'name': 'constant',
                    'params': {}
                },
                'metrics': {'names': ['mean_squared_error'], 'pred_batch_size': 128},
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
                {"name": "resize", "params": {"shape": [16, 16]}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "order", "params": {"order": "th"}}
            ],
        }
    }
    callbacks ={
        'checkpoint':{
            'loss': 'autoencoder_train_mean_squared_error',
            'save_best_only': True
        },
        'budget_secs': 1600,
        'early_stopping':{
            'name': 'basic',
            'params': {
                'patience_loss': 'autoencoder_train_mean_squared_error',
                'patience': 10
            }
        },
        'outdir': 'out'
    }
    params = {'models': models, 'data': data, 'optim': {'batch_size': 128, 'max_nb_epochs': 100, 'seed': 42}, 'callbacks': callbacks}
    train(params, model_builders, callbacks=[DoEachEpoch(_report_image_reconstruction), DoEachEpoch(_report_image_features)])
