import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from ..common import activation_function
from ..common import fully_connected_layers
from ..common import conv2d_layers

def fully_connected(params, shapes):
    input_shape = shapes['X']
    output_shape = shapes['X']
    output_shape_flat = np.prod(output_shape)
    nb_hidden_units = params['fully_connected_nb_hidden_units_list']
    hidden_activations = params['fully_connected_activations']
    output_activation = params['output_activation']
    x = Input(input_shape)
    inp = x
    x = Flatten()(x)
    x = fully_connected_layers(x, nb_hidden_units, hidden_activations)
    x = Dense(output_shape_flat, init='glorot_uniform')(x)
    x = Reshape(output_shape)(x)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(input=inp, output=out)
    return model

def convolutional_bottleneck(params, shapes):
    input_shape = shapes['X']
    nb_channels = input_shape[0]

    encode_nb_filters = params['conv_encode_nb_filters']
    encode_filter_sizes = params['conv_encode_filter_sizes']
    encode_activations = params['conv_encode_activations']

    decode_nb_filters = params['conv_decode_nb_filters']
    decode_filter_sizes = params['conv_decode_filter_sizes']
    decode_activations = params['conv_decode_activations']

    output_filter_size = params['conv_output_filter_size']
    output_activation = params['output_activation']

    inp = Input(input_shape)
    x = inp
    x = conv2d_layers(
        x,
        nb_filters=encode_nb_filters,
        filter_sizes=encode_filter_sizes,
        activations=encode_activations,
        border_mode='valid')
    x = conv2d_layers(
        x,
        nb_filters=decode_nb_filters,
        filter_sizes=decode_filter_sizes,
        activations=decode_activations,
        border_mode='full')
    x = conv2d_layers(
        x,
        nb_filters=[nb_channels],
        filter_sizes=[output_filter_size],
        activations=[output_activation],
        border_mode='full')
    out = x
    model = Model(input=inp, output=out)
    if model.output_shape[1:] != input_shape:
        msg = """Wrong final output shape, expected : {}, got : {}.
                 Please fix the parameters of encoder/decoder/both""".format(input_shape, model.output_shape[1:])
        raise ValueError(msg)
    return model
