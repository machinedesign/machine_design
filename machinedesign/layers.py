"""
In this module I provide common keras layers used in the framework.
"""
import keras.backend as K
from keras.layers import Layer
from keras.layers import LeakyReLU
from keras.layers import Convolution2D

from .utils import get_axis
from .data import floatX


class ksparse(Layer):
    # TODO make it compatible with tensorflow (only works with theano)
    """
    For each example, sort activations, then zerout a proportion of zero_ratio from the smallest activations,
    that rest (1 - zero_ratio) is kept as it is.
    Works inly for fully connected layers.
    Corresponds to k-sparse autoencoders in [1].

    References
    ----------

    [1] Makhzani, A., & Frey, B. (2013). k-Sparse Autoencoders. arXiv preprint arXiv:1312.5663.

    """

    def __init__(self, zero_ratio=0,  **kwargs):
        super(ksparse, self).__init__(**kwargs)
        self.zero_ratio = zero_ratio

    def call(self, X, mask=None):
        import theano.tensor as T
        idx = T.cast(self.zero_ratio * T.cast(X.shape[1], 'float32'), 'int32')
        theta = X[T.arange(X.shape[0]), T.argsort(X, axis=1)[:, idx]]
        mask = X >= theta[:, None]
        return X * mask

    def get_config(self):
        config = {'zero_ratio': self.zero_ratio}
        base_config = super(ksparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class winner_take_all_spatial(Layer):
    # TODO make it compatible with tensorflow (only works with theano)

    """
    Winner take all spatial sparsity defined in [1].
    it takes a convolutional layer, then for each feature map,
    keep only nb_active positions with biggets value
    and zero-out the rest. nb_active=1 corresponds to [1],
    but it can be bigger.
    assumes input of shape (nb_examples, nb_features_maps, h, w).

    Parameters
    ----------

    nb_active : int
        number of active positions in each feature map

    References
    ----------
    [1] Makhzani, A., & Frey, B. J. (2015). Winner-take-all autoencoders.
    In Advances in Neural Information Processing Systems (pp. 2791-2799).

    """

    def __init__(self, nb_active=1, **kwargs):
        super(winner_take_all_spatial, self).__init__(**kwargs)
        self.nb_active = nb_active

    def call(self, X, mask=None):
        if self.nb_active == 0:
            return X * 0
        elif self.nb_active == 1:
            return _winner_take_all_spatial_one_active(X)
        else:
            import theano.tensor as T
            shape = X.shape
            X_ = X.reshape((X.shape[0] * X.shape[1], X.shape[2] * X.shape[3]))
            idx = T.argsort(X_, axis=1)[:, X_.shape[1] - T.minimum(self.nb_active, X_.shape[1])]
            val = X_[T.arange(X_.shape[0]), idx]
            mask = X_ >= val.dimshuffle(0, 'x')
            X_ = X_ * mask
            X_ = X_.reshape(shape)
            return X_

    def get_config(self):
        config = {'nb_active': self.nb_active}
        base_config = super(winner_take_all_spatial, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _winner_take_all_spatial_one_active(X):
    mask = K.cast(_equals(X, K.max(X, axis=(2, 3), keepdims=True)), K.floatx()).reshape(X.shape)
    return mask * X


class winner_take_all_channel(Layer):
    """
    divide each channel into a grid of sizes stride x stride.
    for each grid, across all channels, only one value (the max value) will be active.
    assumes input of shape (nb_examples, nb_features_maps, h, w).

    Parameters
    ----------

    stride : int
        size of the stride

    """

    def __init__(self, stride=1, **kwargs):
        super(winner_take_all_channel, self).__init__(**kwargs)
        self.stride = stride

    def call(self, X, mask=None):
        B, F = X.shape[0:2]
        w, h = X.shape[2:]
        X_ = X.reshape((B, F, w // self.stride, self.stride, h // self.stride, self.stride))
        mask = _equals(X_, X_.max(axis=(1, 3, 5), keepdims=True)) * 1
        mask = mask.reshape(X.shape)
        return X * mask

    def get_config(self):
        config = {'stride': self.stride}
        base_config = super(winner_take_all_channel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _equals(x, y, eps=1e-8):
    return K.abs(x - y) <= eps


class axis_softmax(Layer):
    """
    softmax on a given axis
    keras default softmax only works for matrices and applies to axis=1.
    this works for any tensor and any axis.

    Parameters
    ----------

    axis: int(default=1)
        axis where to do softmax
    """

    def __init__(self, axis=1, **kwargs):
        super(axis_softmax, self).__init__(**kwargs)
        self.axis = get_axis(axis)
        # avoid errors like :
        # ValueError: Layer axis_softmax_1 does not support masking, but was passed an input_mask: All{2}.0
        # when using RNNs with masking
        self.supports_masking = True

    def call(self, X, mask=None):
        e_X = K.exp(X - X.max(axis=self.axis, keepdims=True))
        e_X = e_X / e_X.sum(axis=self.axis, keepdims=True)
        return e_X

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(axis_softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpConv2D(Convolution2D):
    """
    This is a simple up convolution layer that rescales the dimension of a
    convolutional layer using nearest neighbor interpolation (check distill.pub/2016/deconv-checkerboard/).
    It only works with border_mode='same', if it is not the case (border_mode != 'same'), an
    exception will be thrown.
    """

    def compute_output_shape(self, input_shape):
        self._check_stride()
        N, c, h, w = input_shape
        if self.padding == 'same':
            h = h * self.strides[0]
            w = w * self.strides[1]
        elif self.padding == 'full':
            return super().compute_output_shape(input_shape)
        input_shape = N, self.filters, h, w
        return input_shape

    def call(self, x, mask=None):
        self._check_stride()
        sh, sw = self.strides
        # inspired by : <http://distill.pub/2016/deconv-checkerboard/>
        # Upsample by just copying pixel values in grids of size subsamplexsubsample
        s = sh
        # don't do anything if there is any subsampling
        if s > 1:
            # TODO make this comptabile with tensorflow
            import theano.tensor as T
            shape = x.shape
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3], 1))
            x = T.ones((shape[0], shape[1], shape[2], s, shape[3], s)) * x
            x = x.reshape((shape[0], shape[1], shape[2] * s, shape[3] * s))
        # the following is equivalent to keras code except strides=(1, 1) instead
        # of being equal to self.strides
        W, b = self.weights
        output = K.conv2d(
            x, W,
            strides=(1, 1),
            padding=self.padding,
            data_format=self.data_format)
        if self.bias:
            if self.data_format == 'channels_first':
                output += K.reshape(b, (1, self.filters, 1, 1))
            elif self.data_format == 'channels_last':
                output += K.reshape(b, (1, 1, 1, self.filters))
            else:
                raise ValueError('Invalid data format: ' + self.data_format)
        output = self.activation(output)
        return output

    def _check_stride(self):
        sh, sw = self.strides
        assert sh == sw, 'stride should be the same in height and width'
        if sh > 1:
            assert self.padding == 'same', 'Only padding="same" is supported if stride > 1'


class Normalize(Layer):
    """
    a simple layer that takes an input X , multiply it by a scaler and
    add  bias X * scaler + bias. the scaler and bias are fixed and not
    learned. I could have used a Lambda for this, but Lambda sucks
    for serialization, so I needed a specific Layer which I also
    add to custom_objects when calling keras load_model.
    """

    def __init__(self, bias, scale, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.bias = floatX(bias)
        self.scale = floatX(scale)

    def call(self, X, mask=None):
        return (X * self.scale) + self.bias

    def get_config(self):
        config = {'bias': self.bias.tolist(), 'scale': self.scale.tolist()}
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CategoricalMasking(Layer):

    def __init__(self, mask_char=0., **kwargs):
        self.supports_masking = True
        self.mask_char = mask_char
        super(CategoricalMasking, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask=None):
        return K.not_equal(x.argmax(axis=-1, keepdims=True), self.mask_char)

    def call(self, x, mask=None):
        boolean_mask = self.compute_mask(x)
        return x * K.cast(boolean_mask, K.floatx())

    def get_config(self):
        config = {'mask_char': self.mask_char}
        base_config = super(CategoricalMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WordDropout(Layer):

    def __init__(self, proba=0.5, seed=None, char=0, **kwargs):
        self.proba = proba
        self.seed = seed
        self.char = char
        if 0. < self.proba < 1.:
            self.uses_learning_phase = True
        super(WordDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.proba < 1.:
            import theano.tensor as T
            # TODO make it compatible with tensorflow
            char = self.char
            axis = 2
            noise_pr = self.proba
            noise = T.zeros(x.shape)
            noise = T.set_subtensor(noise[:, :, char], 1)
            u = K.random_uniform((x.shape[0], x.shape[1], 1), 0, 1) <= (1 - noise_pr)
            m = K.equal(x.argmax(axis=axis, keepdims=True), char)
            u = u | m
            u = T.addbroadcast(u, axis)
            xn = x * u + noise * (1 - u)
            return K.in_train_phase(xn, x)
        else:
            return K.in_train_phase(x, x)

        def get_config(self):
            config = {'proba': self.proba}
            base_config = super(WordDropout, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class CategoricalNoise(Layer):

    def __init__(self, proba=0.5, seed=None, axis=2, mask_char=None, **kwargs):
        self.proba = proba
        self.seed = seed
        self.axis = axis
        self.mask_char = mask_char
        if 0. < self.proba < 1.:
            self.uses_learning_phase = True
        super(CategoricalNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.proba < 1.:
            import theano.tensor as T
            # TODO make it compatible with tensorflow
            axis = get_axis(self.axis)
            noise_pr = self.proba
            noise = K.random_uniform(x.shape, 0, 1)
            noise = (K.equal(noise, noise.max(axis=axis, keepdims=True)))
            y = K.cast(x.max(axis=axis, keepdims=True), K.floatx())
            u = K.random_uniform(y.shape, 0, 1) <= (1 - noise_pr)
            if self.mask_char is not None:
                m = K.equal(x.argmax(axis=axis, keepdims=True), self.mask_char)
                u = u | m
            u = T.addbroadcast(u, axis)
            xn = x * u + noise * (1 - u)
            return K.in_train_phase(xn, x)
        else:
            return K.in_train_phase(x, x)

        def get_config(self):
            config = {'proba': self.proba, 'axis': self.axis, 'mask_char': self.mask_char}
            base_config = super(CategoricalNoise, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class ZeroMasking(Layer):
    """
    proba : float
        probability of corruption (that is, of zeroing out a unit)
    """
    def __init__(self, proba=0, **kwargs):
        super(ZeroMasking, self).__init__(**kwargs)
        self.proba = proba
    
    def call(self, X, training=None):
        def noised():
            mask = K.random_uniform(X.shape, 0, 1) <= (1 - self.proba) # a == 1  means keep the value
            return X * mask
        return K.in_train_phase(noised, X, training=training)

    def get_config(self):
        config = {'proba': self.proba}
        base_config = super(ZeroMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SaltAndPepper(Layer):
    """
    proba : float
        probability of corruption
    """
    def __init__(self, proba=0, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.proba = proba
    
    def call(self, X, training=None):
        def noised():
            a = K.random_uniform(X.shape, 0, 1) <= (1 - self.proba) # a == 1  means keep the value
            b = K.random_uniform(X.shape, 0, 1) <= 0.5 # b == 1 means set 1, b == 0 means set 0
            c = K.equal(a, 0) * b
            return X * a + c
        return K.in_train_phase(noised, X, training=training)

    def get_config(self):
        config = {'proba': self.proba}
        base_config = super(SaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


layers = {
    'ksparse': ksparse,
    'winner_take_all_spatial': winner_take_all_spatial,
    'winner_take_all_channel': winner_take_all_channel,
    'axis_softmax': axis_softmax,
    'UpConv2D': UpConv2D,
    'leaky_relu': LeakyReLU,
    'Normalize': Normalize,
    'CategoricalNoise': CategoricalNoise,
    'WordDropout': WordDropout,
    'CategoricalMasking': CategoricalMasking,
    'SaltAndPepper': SaltAndPepper,
    'ZeroMasking': ZeroMasking,
}
