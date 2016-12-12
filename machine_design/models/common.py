from keras.layers import Activation
from keras.layers import Dense


def activation_function(name):
    return Activation(name)


def fully_connected_layers(x, nb_hidden_units, activation):
    for nb_hidden in nb_hidden_units:
        x = Dense(nb_hidden)(x)
        x = activation_function(activation)(x)
    return x
