"""
Module containng transformers.
transformers are used to preprocess data before feeding
it into models, it is mostly used when the preprocessing
needs to fit some values from training (e.g mean and std for Standardize),
transformers allows to save these parameters and re-use them when
loading the model for the generation phase for instance.
The Transformer instances follow a scikit-learn like API.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .data import floatX
from .data import intX

EPS = 1e-10

class Standardize:

    """
    Standardize transformer.
    Estimate mean and std of each feature then
    transforms by substracting the mean and dividing
    by std.

    Parameters
    ----------

    axis: int or tuple of int
        axis or axes where to compute mean and std

    Attributes
    ----------

    mean_: numpy array
        current estimate of mean of features
    std_: numpy array
        current estimate of std of features
    n_ : int
        number of calls to partial_fit used to compute the current estimates
    input_shape_ : tuple
        expected shape of inputs to `transform`.
        the number of examples first axis is excluded from
        the shape.
    output_shape_ : tuple
        expected shape of inputs to `inverse_transform`.
        the number of examples first axis is excluded from
        the shape.
    """

    def __init__(self, axis=0, eps=EPS):
        self.mean_ = None
        self.std_ = None
        self.input_shape_ = None
        self.output_shape_ = None
        self.n_ = 0
        self._sum = 0
        self._sum_sqr = 0
        self.axis = axis
        self.eps = eps

    def transform(self, X):
        self._check_if_fitted()
        X = (X - self.mean_) / (self.std_ + self.eps)
        return X

    def inverse_transform(self, X):
        self._check_if_fitted()
        return (X * self.std_) + self.mean_

    def _check_if_fitted(self):
        assert self.mean_ is not None, 'the instance has not been fitted yet'
        assert self.std_ is not None, 'the instance has not been fitted yet'

    def partial_fit(self, X):
        self.n_ += len(X)
        self._sum += X.sum(axis=0)
        self._sum_sqr += (X**2).sum(axis=0)
        self.mean_ = self._sum / self.n_
        self.std_ = np.sqrt(self._sum_sqr / self.n_ - self.mean_ ** 2)
        if not self.input_shape_ and not self.output_shape_:
            self.input_shape_ = X.shape[1:]
            self.output_shape_ = X.shape[1:]

class ColorDiscretizer:
    """
    Color discretizer transformer.
    Used to discretize the colors of a dataset of images with k-means.
    The expected shape of inputs is (nb_examples, nb_colors, h, w)
    where nb_colors is 1 or 3.
    The shape after transformation is (nb_examples, nb_centers, h, w).
    which is a one-hot representation of the data. Only one of the centers
    is one, the others are zero.

    Parameters
    ----------

    nb_centers: int
        Size of the 'palette' after discretization
        (corresponding to the number of centers of k-means applied to colors)
    batch_size: int
        size of the batch when using transform
        (not sure if this is needed anymore)

    Attributes
    ----------

    input_shape_ : tuple
        expected shape of inputs to `transform`.
        the number of examples first axis is excluded from
        the shape.
    output_shape_ : tuple
        expected shape of inputs to `inverse_transform`.
        the number of examples first axis is excluded from
        the shape.

    """
    def __init__(self, nb_centers=5, batch_size=1000):
        # assume centers has shape (nb_centers, nb_channels)
        self.batch_size = batch_size
        self.nb_centers = nb_centers
        self._kmeans = MiniBatchKMeans(n_clusters=nb_centers)
        self.input_shape_ = None
        self.output_shape_ = None

    def _check_if_fitted(self):
        assert self.input_shape_ is not None, 'the instance has not been fitted yet'
        assert self.output_shape_ is not None, 'the instance has not been fitted yet'

    def partial_fit(self, X):
        # assume X has shape (nb_examples, nb_colors, h, w)
        input_shape = X.shape
        X = X.transpose((0, 2, 3, 1))
        nb, h, w, nb_colors = X.shape
        X = X.reshape((nb * h * w, nb_colors))
        self._kmeans.partial_fit(X)
        self.centers = self._kmeans.cluster_centers_# (nb_centers, nb_channels)
        if not self.input_shape_ and not self.output_shape_:
            self.input_shape_ = input_shape[1:]
            self.output_shape_ = (self.nb_centers,) + X.shape[2:]
        return self

    def transform(self, X):
        self._check_if_fitted()
        # assume X has shape (nb_examples, nb_channels, h, w)
        X = X[:, :, :, :, np.newaxis] #(nb_examples, nb_channels, h, w, 1)
        centers = self.centers.T # (nb_channels, nb_centers)
        nb_centers = centers.shape[1]
        centers = centers[np.newaxis, :, np.newaxis, np.newaxis, :]#(1, nb_channels, 1, 1, nb_centers)
        outputs = []
        for i in range(0, len(X), self.batch_size):
            dist = np.abs(X[i:i + self.batch_size] - centers) # (nb_examples, nb_channels, h, w, nb_centers)
            dist = dist.sum(axis=1) # (nb_examples, h, w, nb_centers)
            out = dist.argmin(axis=3) # (nb_examples, h, w)
            out = onehot(out, D=nb_centers) # (nb_examples, h, w, nb_centers)
            out = out.transpose((0, 3, 1, 2))# (nb_examples, nb_centers, h, w)
            outputs.append(out)
        return np.concatenate(outputs, axis=0)

    def inverse_transform(self, X):
        self._check_if_fitted()
        # assume X has shape (nb_examples, nb_centers, h, w)
        X = X.argmax(axis=1)
        nb, h, w = X.shape
        X = X.flatten()
        X = self.centers[X]
        nb_channels = X.shape[1]
        X = X.reshape((nb, h, w, nb_channels))
        X = X.transpose((0, 3, 1, 2))
        return X # (nb_examples, nb_channels, h, w)

def onehot(X, D=10):
    """
    Converts a numpy array of integers to a one-hot
    representation.
    `X` can have arbitrary number of dimesions, it just
    needs to be integers.
    A new tensor dim is added to `X` with `D` dimensions :
    the shape of the transformed array is X.shape + (D,)
    Parameters
    ----------

    X : numpy array of integers

    D : total number of elements in the one-hot
        representation.

    Returns
    -------

    array of shape : X.shape + (D,)
    """
    X = intX(X)
    nb = np.prod(X.shape)
    x = X.flatten()
    m = np.zeros((nb, D))
    m[np.arange(nb), x] = 1.
    m = m.reshape(X.shape + (D,))
    m = floatX(m)
    return m

transformer = {
    'Standardize': Standardize,
    'ColorDiscretizer': ColorDiscretizer
}

def make_transformers_pipeline(transformers):
    """
    helpers create a list of instances of Transformer.

    Parameters
    ----------

    transformers : list of dict
        each dict has two keys, `name` and `params`.
        `name` is the name of the Transformer.
        `params` are the parameters of the __init__ of the Transformer.
        available transformers :
            - 'Standardize'.

    Returns
    -------

    list of Transformer

    """
    return [transformer[t['name']](**t['params']) for t in transformers]

def fit_transformers(transformers, iter_generator):
    """
    fit a list of Transformers

    Parameters
    ----------

    transformers: list of Transformer

    iter_generator : callable
        function that returns an iterator (fresh one)

    WARNING:
        make sure that the iterators generated by the call
        are deterministic so that we don't end up each time
        with a different sample
    """
    for i, t in enumerate(transformers):
        tprev = transformers[0:i]
        for X in iter_generator():
            for tp in tprev:
                X = tp.transform(X)
            t.partial_fit(X)

def transform(iterator, transformers):
    """
    transform an iterator using a list of Transformer

    Parameters
    ----------

    iterator: iterable of array like
        data to transform

    transformers: list of Transformer

    Yields
    ------

    transformed iterator

    """
    for d in iterator:
        d = transform_one(d, transformers)
        yield d

def transform_one(d, transformers):
    for t in transformers:
        d = t.transform(d)
    return d

def inverse_transform_one(d, transformers):
    for t in transformers[::-1]:
        d = t.inverse_transform(d)
    return d
