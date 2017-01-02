from keras.layers import Input
from keras.models import Model

from ..common import activation_function
from ..common import conv2d_layers
from ..common import Convolution2D
from ..common import UpConv2D


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
