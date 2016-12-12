import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from ..common import activation_function
from ..common import fully_connected_layers

def fullyconnected(params, shapes):
    input_shape = shapes['X']
    output_shape = shapes['X']
    output_shape_flat = np.prod(output_shape)
    nb_hidden_units = params['fully_connected_nb_hidden_units_list']
    hidden_activation = params['fully_connected_activation']
    output_activation = params['output_activation']
    x = Input(input_shape)
    inp = x
    x = Flatten()(x)
    x = fully_connected_layers(x, nb_hidden_units, hidden_activation)
    x = Dense(output_shape_flat, init='glorot_uniform')(x)
    x = Reshape(output_shape)(x)
    x = activation_function(output_activation)(x)
    out = x
    model = Model(input=inp, output=out)
    return model
