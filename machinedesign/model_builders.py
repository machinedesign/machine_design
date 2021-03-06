"""
Basic model builders.
model builders are functions that take specifications of architectures
as a dict, input and output shape, and it return a keras Model.
The model is a function that takes an input tensor and returns an output tensor.
Model builders are specifically designed for hyper-parameter optimization,
hence the use of dict to parametrize its architecture.
We note that the architecture should be independent of input and output shapes,
this is the reason why I put them outside the specification of the architecture.
"""
from functools import partial

import numpy as np

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from .layers import Convolution2D
from .layers import UpConv2D

from .common import activation_function
from .common import fully_connected_layers
from .common import noise as _noise
from .common import conv2d_layers
from .common import check_model_shape_or_exception


def noise(params, input_shape, output_shape):
    noise_name = params['type']
    noise_params = params['params']
    apply_noise = partial(_noise, name=noise_name, params=noise_params)
    inp = Input(input_shape)
    out = apply_noise(inp)
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def fully_connected(params, input_shape, output_shape):
    """
    stack of fully connected layers

    input -> fc1 -> fc2 -> ... -> output
    params
    ------
    nb_hidden_units : list of int
        it is not including the final output layer
    activations : list of str
    output_activation : str
    """
    output_shape_flat = np.prod(output_shape)
    output_activation = params['output_activation']

    x = Input(input_shape)
    inp = x
    if len(input_shape) > 1:
        x = Flatten()(x)
    x = _fully_connected_stack(x, params)
    x = Dense(output_shape_flat, kernel_initializer='glorot_uniform')(x)
    x = Reshape(output_shape)(x)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def _fully_connected_stack(x, params):
    nb_hidden_units = params['nb_hidden_units']
    hidden_activations = params['activations']
    x = fully_connected_layers(x, nb_hidden_units, hidden_activations)
    return x


def convolutional(params, input_shape, output_shape):
    """

    input -> conv1 -> conv2 -> ... -> global average pooling (if output is a matrix) -> output

    params
    ------

    nb_filters : list of int
    filter_sizes : list of int
    activations : list of str
    stride : int
    """
    output_activation = params['output_activation']
    output_filter_size = params['output_filter_size']
    nb_channels = np.prod(output_shape)
    inp = Input(input_shape)
    x = inp
    x = _convolutional_stack(x, params)
    x = conv2d_layers(
        x,
        nb_filters=[nb_channels],
        filter_sizes=[output_filter_size],
        activations=[output_activation],
        border_mode='valid',
        stride=1,
        conv_layer=Convolution2D)
    out = x
    if len(output_shape) == 1:
        out = GlobalAveragePooling2D()(out)
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def _convolutional_stack(x, params):
    nb_filters = params['nb_filters']
    filter_sizes = params['filter_sizes']
    activations = params['activations']
    stride = params['stride']
    # Encode
    x = conv2d_layers(
        x,
        nb_filters=nb_filters,
        filter_sizes=filter_sizes,
        activations=activations,
        border_mode='valid',
        stride=stride,
        conv_layer=Convolution2D)
    return x


def _upconvolutional_stack(x, params):
    nb_filters = params['nb_filters']
    filter_sizes = params['filter_sizes']
    activations = params['activations']
    stride = params['stride']
    # Encode
    x = conv2d_layers(
        x,
        nb_filters=nb_filters,
        filter_sizes=filter_sizes,
        activations=activations,
        border_mode='full' if stride == 1 else 'same',
        stride=stride,
        conv_layer=UpConv2D if stride > 1 else Convolution2D)
    return x


def fc_upconvolutional(params, input_shape, output_shape):
    """

    input -> fc1 -> fc2 -> ... -> fc_n -> reshape -> conv1 -> conv2 -> ... -> output

    params
    ------

    fully_connected : dict
        params from _fully_connected_stack
    reshape : list of int
    upconvolutional : dict
        params from _upconvolutional_stack
    output_filter_size : int
    output_activation : str
    """
    params = params.copy()
    reshape = params['reshape']
    output_activation = params['output_activation']
    nb_output_channels = output_shape[0]
    x = Input(input_shape)
    inp = x
    x = _fully_connected_stack(x, params['fully_connected'])
    x = Reshape(reshape)(x)
    x = _upconvolutional_stack(x, params['upconvolutional'])
    shape = Model(inputs=inp, outputs=x).output_shape[1:]
    remaining = shape[1] - output_shape[1] + 1
    x = conv2d_layers(
        x,
        nb_filters=[nb_output_channels],
        filter_sizes=[remaining],
        activations=[output_activation],
        init='glorot_uniform',
        border_mode='valid',
        stride=1)
    out = x
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


builders = {
    'fully_connected': fully_connected,
    'convolutional': convolutional,
    'fc_upconvolutional': fc_upconvolutional,
    'noise': noise,
}
