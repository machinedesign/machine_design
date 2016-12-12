from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers
from keras import objectives

def activation_function(name):
    return Activation(name)


def fully_connected_layers(x, nb_hidden_units, activation):
    for nb_hidden in nb_hidden_units:
        x = Dense(nb_hidden)(x)
        x = activation_function(activation)(x)
    return x

def get_optimizer(name):
    if hasattr(optimizers, name):
        return getattr(optimizers, name)
    else:
        raise Exception('unknown optimizer : {}'.format(name))

def build_optimizer(algo_name, algo_params, optimizers=optimizers):
    optimizer = get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)
    return optimizer

def get_loss(name, objectives=objectives):
    return getattr(objectives, name)
