from functools import partial

import numpy as np

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from .common import activation_function
from .common import fully_connected_layers
from .common import noise
from .common import conv2d_layers
from .common import Convolution2D

def fully_connected(params, input_shape, output_shape):
    output_shape_flat = np.prod(output_shape)
    nb_hidden_units = params['fully_connected_nb_hidden_units_list']
    hidden_activations = params['fully_connected_activations']
    output_activation = params['output_activation']

    noise_name = params['input_noise']['name']
    noise_params = params['input_noise']['params']
    apply_noise = partial(noise, name=noise_name, params=noise_params)

    x = Input(input_shape)
    inp = x
    x = Flatten()(x)
    x = apply_noise(x)
    x = fully_connected_layers(x, nb_hidden_units, hidden_activations)
    x = Dense(output_shape_flat, init='glorot_uniform')(x)
    x = Reshape(output_shape)(x)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(input=inp, output=out)
    return model

def convolutional(params, input_shape, output_shape):
    nb_filters = params['nb_filters']
    filter_sizes = params['filter_sizes']
    activations = params['activations']
    stride = params['stride']

    inp = Input(input_shape)
    x = inp
    # Encode
    x = conv2d_layers(
        x,
        nb_filters=nb_filters,
        filter_sizes=filter_sizes,
        activations=activations,
        border_mode='valid',
        stride=stride,
        conv_layer=Convolution2D)
    if len(output_shape) == 1:
        x = GlobalAveragePooling2D()(x)
    out = x
    model = Model(input=inp, output=out)
    if model.output_shape[1:] != output_shape:
        msg = """Wrong final output shape, expected : {}, got : {}.
                 Please fix the parameters""".format(output_shape, model.output_shape[1:])
        raise ValueError(msg)
    return model
