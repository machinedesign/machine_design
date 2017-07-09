"""
Module containing transformers.
transformers are used to preprocess data before feeding
it into models, it is mostly used when the preprocessing
needs to fit some values from training (e.g mean and std for Standardize),
transformers allows to save these parameters and re-use them when
loading the model for the generation phase for instance.
The Transformer instances follow a scikit-learn like API.
"""

import six
from six.moves import map

import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .data import floatX
from .data import intX

EPS = 1e-10

ZERO_CHARACTER = 0
BEGIN_CHARACTER = 1
END_CHARACTER = 2


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
        self.centers = self._kmeans.cluster_centers_  # (nb_centers, nb_channels)
        if not self.input_shape_ and not self.output_shape_:
            self.input_shape_ = input_shape[1:]
            self.output_shape_ = (self.nb_centers,) + X.shape[2:]
        return self

    def transform(self, X):
        self._check_if_fitted()
        # assume X has shape (nb_examples, nb_channels, h, w)
        X = X[:, :, :, :, np.newaxis]  # (nb_examples, nb_channels, h, w, 1)
        centers = self.centers.T  # (nb_channels, nb_centers)
        nb_centers = centers.shape[1]
        # (1, nb_channels, 1, 1, nb_centers)
        centers = centers[np.newaxis, :, np.newaxis, np.newaxis, :]
        outputs = []
        for i in range(0, len(X), self.batch_size):
            # (nb_examples, nb_channels, h, w, nb_centers)
            dist = np.abs(X[i:i + self.batch_size] - centers)
            dist = dist.sum(axis=1)  # (nb_examples, h, w, nb_centers)
            out = dist.argmin(axis=3)  # (nb_examples, h, w)
            out = onehot(out, D=nb_centers)  # (nb_examples, h, w, nb_centers)
            out = out.transpose((0, 3, 1, 2))  # (nb_examples, nb_centers, h, w)
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
        return X  # (nb_examples, nb_channels, h, w)


class FileLoader:

    def __init__(self, filename, pos=0):
        self.filename = filename
        self.pos = pos
        self.transformer_ = None

    def partial_fit(self, X):
        pass

    def _load(self):
        if self.transformer_ is None:
            with open(self.filename, 'rb') as fd:
                self.transformer_ = pickle.load(fd)

    def transform(self, X):
        self._load()
        return self.transformer_[self.pos].transform(X)

    def inverse_transform(self, X):
        self._load()
        self.transformer_[self.pos].inverse_transform(X)


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


class DocumentVectorizer:
    """
    a document transformer.
    it takes as input a list of documents and returns a vectorized
    representation of the documents.

    documents are either list of str or list of list of str.
    - if the documents are a list of str, then each document is a str so
      the tokens are the individual characters
    - if the documents is a list of list of str, then the tokens are
      the elements of the list (words)

    Parameters
    ----------

    length : int or None
        if int, maximum length of the documents.
        if None, no maximum length is assumed.
        the behavior of length is that if the length of a document
        is greater than length then it is truncated to fit length.
        if pad is True, then all documents will have exactly length size
        (counting the beging, the end character and the zero characters),
        the remaining characters are filled with the zero character.

    begin_character : bool
        whether to add a begin character in the beginning of each document

    end_character : bool
        whether to add an end character in the end of each document

    onehot : bool
        whether to convert the documents into onehot representation when
        calling the method transform. this also means that when calling
        inverse_transform, it expects the the documents to be represented
        as onehot, to give back the strings as a result.

    """

    def __init__(self, length=None,
                 begin_character=True, end_character=True,
                 pad=True, onehot=False):
        self.length = length
        self.begin_character = begin_character
        self.end_character = end_character
        self.pad = pad
        self.onehot = onehot

        # input_dtype_ is needed by machinedesign.autoencoder iterative_refinement
        # because the input array must be initialized there and the type of the
        # data must be known, by default it is a float, whereas here we have strs.
        if length:
            self.input_dtype_ = '<U{}'.format(length)
        else:
            # if length is not specified, make a 1000 limit to strs
            self.input_dtype_ = '<U1000'
        # input shape is a scalar (str scalar)
        self.input_shape_ = tuple([])
        # to know whether the documents is a list of str (tokens_are_chars is True)
        # or list of list of str (tokens_are_chars is False)
        self._tokens_are_chars = None
        self._ind = 0
        self.words_ = set()
        self.word2int_ = {}
        self.int2word_ = {}
        self.nb_words_ = 0
        # ensure that the ZERO_CHARACTER takes the index 0
        # in the vectorized representation by appending it
        # first to the vocabulary.
        self._update(set([ZERO_CHARACTER]))
        self._update(set([BEGIN_CHARACTER]))
        self._update(set([END_CHARACTER]))

    def partial_fit(self, docs):
        # if not set, set _tokens_are_chars
        # TODO in principle I should also verify the coherence
        # of it for all documents
        if self._tokens_are_chars is None:
            if isinstance(docs[0], six.string_types):
                self._tokens_are_chars = True
            else:
                self._tokens_are_chars = False
        words = set(word for doc in docs for word in doc)
        self._update(words)
        return self

    def _update(self, words):
        """this functions adds new words to the vocabulary"""
        new_words = words - self.words_
        for word in new_words:
            self.word2int_[word] = self._ind
            self.int2word_[self._ind] = word
            self._ind += 1
        self.words_ |= new_words
        self.nb_words_ = len(self.words_)

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _doc_transform(self, doc):
        doc = list(map(self._word_transform, doc))
        if self.length:
            len_doc = min(
                len(doc) + self.begin_character + self.end_character,
                self.length)  # max possible length is self.length
            if self.begin_character:
                len_doc -= 1
            if self.end_character:
                len_doc -= 1
            doc_new = []
            if self.begin_character:
                doc_new.append(self._word_transform(BEGIN_CHARACTER))
            doc_new.extend(doc[0:len_doc])
            if self.end_character:
                doc_new.append(END_CHARACTER)
            if self.pad:
                remaining = self.length - len(doc_new)
                doc_new.extend(list(map(self._word_transform, [ZERO_CHARACTER] * remaining)))
            return doc_new
        else:
            return doc

    def _word_transform(self, word):
        return self.word2int_[word]

    def transform(self, docs):
        docs = list(map(self._doc_transform, docs))
        if self.length and self.pad:
            # if both length and pad are set, then all documents
            # have the same length, so we can build a numpy array
            # out of docs
            docs = np.array(docs)
        if self.onehot:
            docs = onehot(docs, D=self.nb_words_)
        return docs

    def inverse_transform(self, X):
        if self.onehot:
            X = X.argmax(axis=-1)
        docs = []
        for s in X:
            docs.append([self.int2word_[w] for w in s])
        if self._tokens_are_chars:
            docs = list(map(doc_to_str, docs))
        return docs


def doc_to_str(doc):
    # find the end character and truncate
    try:
        idx = doc.index(END_CHARACTER)
        doc = doc[0:idx]
    except ValueError:
        # if end character does not exist, do not do anything
        # and take the whole thing as a result.
        pass
    # after removing the end character, remove the zero and begin character
    doc = [d for d in doc if d not in (BEGIN_CHARACTER, ZERO_CHARACTER, END_CHARACTER)]
    return ''.join(doc)


class Scaler:
    """
    Scale by some value.
    nothing is fitted, the `value` is used as is.
    """
    def __init__(self, value):
        self.value = value

    def partial_fit(self, X):
        pass

    def transform(self, X):
        return X / self.value

    def inverse_transform(self, X):
        return X * self.value


transformers = {
    'Standardize': Standardize,
    'ColorDiscretizer': ColorDiscretizer,
    'FileLoader': FileLoader,
    'DocumentVectorizer': DocumentVectorizer,
    'Scaler': Scaler,
}


def make_transformers_pipeline(transformer_list, transformers=transformers):
    """
    helpers create a list of instances of Transformer.

    Parameters
    ----------

    transformer_list : list of dict
        each dict has two keys, `name` and `params`.
        `name` is the name of the Transformer.
        `params` are the parameters used for the __init__ of the Transformer.
        available transformers :
            - 'Standardize'
            - 'ColorDiscretizer'

    Returns
    -------

    list of Transformer

    """
    return [transformers[t['name']](**t['params']) for t in transformer_list]


def fit_transformers(transformer_list, iter_generator):
    """
    fit a list of Transformers

    Parameters
    ----------

    transformer_list: list of Transformer

    iter_generator : callable
        function that returns an iterator (fresh one)

    WARNING:
        make sure that the iterators generated by the call
        are deterministic so that we don't end up each time
        with a different sample
    """
    for i, t in enumerate(transformer_list):
        tprev = transformer_list[0:i]
        for X in iter_generator():
            for tp in tprev:
                X = tp.transform(X)
            t.partial_fit(X)


def transform(iterator, transformer_list):
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
        d = transform_one(d, transformer_list)
        yield d


def transform_one(d, transformer_list):
    """
    apply a list of transformers to `d`
    """
    for t in transformer_list:
        d = t.transform(d)
    return d


def inverse_transform_one(d, transformer_list):
    """
    apply inverse transform of a list of transformers to `d`.
    because it is inverse transform, transformers is processed
    backwards.
    """
    for t in transformer_list[::-1]:
        d = t.inverse_transform(d)
    return d
