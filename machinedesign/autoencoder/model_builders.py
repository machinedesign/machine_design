from keras.layers import Input
from keras.layers import RepeatVector
from keras.models import Model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import merge

from ..layers import Convolution2D
from ..layers import UpConv2D
from ..layers import CategoricalMasking

from ..common import rnn_stack
from ..common import activation_function
from ..common import conv2d_layers
from ..common import conv1d_layers
from ..common import check_model_shape_or_exception
from ..common import fully_connected_layers


def convolutional_bottleneck(params, input_shape, output_shape):
    """
    conv1->conv2->...conv_h -> code_activations(conv_h)...-> conv_n

    form 1 to h : pad='valid' or pad='same' if stride > 1
    from h + 1 to n : pad='full' or pad='same' if stride > 1
    params
    ------

    stride : int
        stride to use in all layers
        in encode layers, it behaves as a downscaler.
        in decode layers, it behaves as upscaler.
    encode_nb_filters : list of int
        nb filters of encoder
    encode_filter_sizes : list of int
        filter sizes of encoder
    encode_activations : list of str

    code_activations : list of str
        list of activations to apply to the bottleneck layer

    decode_nb_filters : list of int
        nb filters of decoder
    decode_filter_sizes : list of int
        filter sizes of decoder
    decode_activations : list of str
        activations of decoder

    """
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
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def convolutional_rnn_autoencoder(params, input_shape, output_shape):
    assert input_shape == output_shape
    max_length = input_shape[0]
    encode_nb_filters = params['encode_nb_filters']
    encode_filter_sizes = params['encode_filter_sizes']
    encode_activations = params['encode_activations']
    latent_nb_hidden_units = params['latent_nb_hidden_units']
    latent_activations = params['latent_activations']
    decode_nb_hidden_units = params['decode_nb_hidden_units']
    output_activation = params['output_activation']
    rnn_type = params['rnn_type']

    inp = Input(input_shape)
    x = inp
    x = conv1d_layers(
        x,
        encode_nb_filters,
        encode_filter_sizes,
        encode_activations)
    x = Flatten()(x)
    x = fully_connected_layers(x, latent_nb_hidden_units, latent_activations)
    x = RepeatVector(max_length)(x)
    x = rnn_stack(x, decode_nb_hidden_units, rnn_type=rnn_type)
    x = TimeDistributed(Dense(output_shape[1]))(x)
    out = activation_function(output_activation)(x)
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def rnn_rnn_autoencoder(params, input_shape, output_shape):
    assert input_shape == output_shape
    max_length = input_shape[0]
    rnn_type = params['rnn_type']
    encode_nb_hidden_units = params['encode_nb_hidden_units']
    latent_nb_hidden_units = params['latent_nb_hidden_units']
    latent_activations = params['latent_activations']
    decode_nb_hidden_units = params['decode_nb_hidden_units']
    output_activation = params['output_activation']

    decoder_include_input = params.get('decoder_include_input', False)

    inp = Input(input_shape)
    x = inp
    x = CategoricalMasking(mask_char=0)(x)
    inp_masked = x
    if decoder_include_input:
        x = CategoricalMasking(mask_char=2)(x)
    x = rnn_stack(
        x,
        encode_nb_hidden_units,
        rnn_type=rnn_type,
        return_sequences=False)
    x = fully_connected_layers(x, latent_nb_hidden_units, latent_activations)
    x = RepeatVector(max_length)(x)
    if decoder_include_input:
        x = merge((inp_masked, x), mode='concat', concat_axis=-1, name='concat')
    x = rnn_stack(x, decode_nb_hidden_units, rnn_type=rnn_type)
    x = TimeDistributed(Dense(output_shape[1]))(x)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


def rnn(params, input_shape, output_shape):
    """
    rnm1 -> rnn2-> ... -> rnn_n
    """
    assert input_shape == output_shape
    input_shape = (None,) + input_shape[1:]
    output_shape = input_shape

    rnn_type = params['rnn_type']
    nb_hidden_units = params['nb_hidden_units']
    output_activation = params['output_activation']
    dropout = params['dropout']
    inp = Input(input_shape)
    x = inp
    x = rnn_stack(x, nb_hidden_units, rnn_type=rnn_type, dropout=dropout)
    x = TimeDistributed(Dense(output_shape[1]))(x)
    out = activation_function(output_activation)(x)
    model = Model(inputs=inp, outputs=out)
    check_model_shape_or_exception(model, output_shape)
    return model


builders = {
    'convolutional_bottleneck': convolutional_bottleneck,
    'convolutional_rnn_autoencoder': convolutional_rnn_autoencoder,
    'rnn_rnn_autoencoder': rnn_rnn_autoencoder,
    'rnn': rnn
}
