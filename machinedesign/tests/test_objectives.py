import numpy as np
import keras.backend as K

from machinedesign.objectives import axis_categorical_crossentropy

def test_axis_categorical_crossentropy():
    X = K.placeholder(shape=(None, 2, 10, 10))
    Y = K.placeholder(shape=(None, 2, 10, 10))
    pred = K.function([X, Y], axis_categorical_crossentropy(X, Y, axis=1))
    np.random.seed(42)
    nb = 100

    label = np.random.randint(0, 1 , size=(nb * 100))
    y = np.zeros((nb * 100, 2))
    y[label==0, 0] = 1
    y[label==1, 1] = 1

    x = np.random.uniform(0, 1, size=(nb * 100, 2))
    x[label==0, 0] = 0.9
    x[label==1, 0] = 0.1
    x[:, 1] = 1 - x[:, 0]

    x = x.reshape((nb, 10, 10, 2))
    x = x.transpose((0, 3, 1, 2))

    y = y.reshape((nb, 10, 10, 2))
    y = y.transpose((0, 3, 1, 2))

    l = pred([y, x])
    assert np.allclose(l, -np.log(0.9))
