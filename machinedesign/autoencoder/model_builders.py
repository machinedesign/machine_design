from functools import partial

import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from ..common import activation_function
from ..common import fully_connected_layers
from ..common import conv2d_layers
from ..common import Convolution2D
from ..common import UpConv2D
from ..common import noise

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

def convolutional_bottleneck(params, input_shape, output_shape):
    assert input_shape == output_shape
    nb_channels = input_shape[0]

    stride = params['stride']

    encode_nb_filters = params['conv_encode_nb_filters']
    encode_filter_sizes = params['conv_encode_filter_sizes']
    encode_activations = params['conv_encode_activations']

    code_activations = params['code_activations']

    decode_nb_filters = params['conv_decode_nb_filters']
    decode_filter_sizes = params['conv_decode_filter_sizes']
    decode_activations = params['conv_decode_activations']

    output_filter_size = params['conv_output_filter_size']
    output_activation = params['output_activation']

    inp = Input(input_shape)
    x = inp
    # Encode
    x = conv2d_layers(
        x,
        nb_filters=encode_nb_filters,
        filter_sizes=encode_filter_sizes,
        activations=encode_activations,
        border_mode='valid' if stride == 1 else 'same',
        stride=stride,
        conv_layer=Convolution2D)

    # Apply code activations (e.g sparsity)
    for act in code_activations:
        x = activation_function(act)(x)

    # Decode back
    x = conv2d_layers(
        x,
        nb_filters=decode_nb_filters + [nb_channels],
        filter_sizes=decode_filter_sizes + [output_filter_size],
        activations=decode_activations + [output_activation],
        border_mode='full' if stride == 1 else 'same',
        stride=stride,
        conv_layer=UpConv2D)

    out = x
    model = Model(input=inp, output=out)
    if model.output_shape[1:] != input_shape:
        msg = """Wrong final output shape, expected : {}, got : {}.
                 Please fix the parameters of encoder/decoder/both""".format(input_shape, model.output_shape[1:])
        raise ValueError(msg)
    return model
