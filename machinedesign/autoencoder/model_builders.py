from keras.layers import Input
from keras.models import Model

from ..layers import Convolution2D
from ..layers import UpConv2D

from ..common import activation_function
from ..common import conv2d_layers
from ..common import check_model_shape_or_exception

def convolutional_bottleneck(params, input_shape, output_shape):
    assert input_shape == output_shape
    nb_channels = input_shape[0]

    stride = params['stride']

    encode_nb_filters = params['encode_nb_filters']
    encode_filter_sizes = params['encode_filter_sizes']
    encode_activations = params['encode_activations']

    code_activations = params['code_activations']

    decode_nb_filters = params['decode_nb_filters']
    decode_filter_sizes = params['decode_filter_sizes']
    decode_activations = params['decode_activations']

    output_filter_size = params['output_filter_size']
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
    check_model_shape_or_exception(model, output_shape)
    return model
