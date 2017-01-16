import numpy as np
import os
import logging
import pickle

from skimage.io import imsave

from keras.models import load_model

from ..data import floatX
from ..data import minibatcher

from ..interface import train as train_basic

from ..common import check_family_or_exception
from ..common import custom_objects
from ..common import get_layers

from ..utils import mkdir_path
from ..utils import get_axis

from ..viz import reshape_to_images
from ..viz import grid_of_images_default
from ..viz import horiz_merge

from ..callbacks import DoEachEpoch
from ..transformers import inverse_transform_one

from ..model_builders import builders as model_builders_basic
from .model_builders import builders as model_builders_autoencoder

from ..interface import default_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_builders = model_builders_basic.copy()
model_builders.update(model_builders_autoencoder)

default_config = default_config._replace(model_builders=model_builders)


def train(params, config=default_config, custom_callbacks=[], logger=logger):
    check_family_or_exception(params['family'], 'autoencoder')
    # Callbacks
    callbacks = params['report']['callbacks']
    report_callbacks = []
    if callbacks:
        if 'image_reconstruction' in callbacks:
            cb = DoEachEpoch(_report_image_reconstruction)
            cb.outdir = params['report']['outdir']
            report_callbacks.append(cb)
        if 'image_features' in callbacks:
            cb = DoEachEpoch(_report_image_features)
            cb.outdir = params['report']['outdir']
            report_callbacks.append(cb)
    # Call training function
    return train_basic(
        params,
        custom_callbacks=custom_callbacks + report_callbacks,
        config=config)


def load(folder, custom_objects=custom_objects):
    model = load_model(os.path.join(folder, 'model.h5'), custom_objects=custom_objects)
    with open(os.path.join(folder, 'transformers.pkl'), 'rb') as fd:
        transformers = pickle.load(fd)
    model.transformers = transformers
    return model


def generate(params):
    method = params['method']
    model_params = params['model']
    folder = model_params['folder']
    model = load(folder)
    return _run_method(method, model)


def _run_method(method, model):
    name = method['name']
    params = method['params']
    save_folder = method['save_folder']
    func = get_method(name)
    X = func(params, model)
    mkdir_path(save_folder)
    filename = os.path.join(save_folder, 'generated.npz')
    np.savez_compressed(filename, full=X, generated=X[:, -1])


def _iterative_refinement(params, model):
    """
    take params of iterative refinement (see below), a keras model
    and returns a numpy array of shape :
        (nb_samples, nb_iter,) + model.input_shape[1:]
    """
    # get params
    batch_size = params['batch_size']
    N = params['nb_samples']
    nb_iter = params['nb_iter']

    binarize = params['binarize']
    binarize_name = binarize['name']
    binarize_params = binarize['params']

    noise = params['noise']
    noise_name = noise['name']
    noise_params = noise['params']

    stop_if_unchanged = params['stop_if_unchanged']
    seed = params['seed']
    # I could have just used model.input_shape directly without asking
    # for this (optional) parameter, however some models have None
    # in their input_shape, for instance in RNN models the number
    # of timesteps could be not specified. In that case, it's the user
    # that should provide the number of timesteps.
    # setting model_input_shape avoids the error "TypeError: an integer is required"
    # when there is a None in the input_shape of the model.
    model_input_shape = params.get('model_input_shape', model.input_shape[1:])

    # Initialize the reconstructions
    transformers = model.transformers
    # shape is shape of inputs before
    # applying the transformers (if there are transformers)
    shape = transformers[0].input_shape_ if len(transformers) else model_input_shape
    shape = tuple(shape)
    # if there are transformers, we will need the input dtype
    # it can be provided by the first transformer.
    # it is needed because we have to initialize the input
    # here, so we also need to know the type of the data.
    # by default, the type is float32.
    if len(transformers) and hasattr(transformers[0], 'input_dtype_'):
        dtype = transformers[0].input_dtype_
    else:
        dtype = 'float32'
    X = np.empty((N, nb_iter + 1,) + shape, dtype=dtype)
    rng = np.random.RandomState(seed)

    s = rng.uniform(size=(N,) + model_input_shape)
    X[:, 0] = inverse_transform_one(s, transformers)
    # Build apply function
    reconstruct = minibatcher(model.predict, batch_size=batch_size)

    # reconstruction loop
    previous_score = None
    for i in (range(1, nb_iter + 1)):
        logger.info('Iteration {}'.format(i))
        s = _apply_noise(noise_name, noise_params, s, rng=rng)
        s_orig = s
        s = reconstruct(s)
        s = _apply_binarization(binarize_name, binarize_params, s, rng=rng)
        X[:, i] = inverse_transform_one(s, transformers)
        score = float(np.abs(s - s_orig).mean())
        logger.info('Mean absolute error : {:.5f}'.format(score))
        if (previous_score and score == previous_score and stop_if_unchanged):
            logger.info('Stopping at iteration {}/{} because score did not change'.format(i, nb_iter))
            X = X[:, 0:i + 1]
            break
        previous_score = score
    return X


def _apply_noise(name, params, X, rng=np.random):
    if name == 'masking':
        # with proba noise_pr, set X value to zero, e.g
        # with proba 1 - noise_pr, leave X value as it is
        # e.g [0.7, 0.1, 0.3, 0.9] becomes [0.7, 0, 0, 0.9]
        noise_pr = params['proba']
        X = (rng.uniform(size=X.shape) <= (1 - noise_pr)) * X
        X = floatX(X)
        return X
    elif name == 'gaussian':
        std = params['std']
        return X + np.random.normal(loc=0, scale=std, size=X.shape)
    elif name == 'choice':
        # applies to some axis and assumes one hot representation
        # on that selected axis.
        # with proba noise_pr, switch the category at random, e.g [1 0 0 0] becomes [0 1 0 0]
        # with proba 1 - noise_pr, leave the category as it is, e.g [1 0 0 0] stays [1 0 0 0]
        axis = get_axis(params['axis'])
        noise_pr = params['proba']
        mask = rng.uniform(size=X.shape)
        mask = (mask == mask.max(axis=axis, keepdims=True))
        shape = list(X.shape)
        shape[axis] = 1
        u = rng.uniform(size=shape) <= (1 - noise_pr)
        X = X * u + mask * (1 - u)
        X = floatX(X)
        return X
    elif name == 'none':
        return X
    else:
        raise ValueError('Unknown noise method : {}'.format(name))


def _apply_binarization(name, params, X, rng=np.random):
    if name == 'sample_bernoulli':
        # replace by one with proba X, 0 with proba (1 - X)
        X = rng.binomial(n=1, p=X, size=X.shape)
        return X
    elif name == 'binary_threshold':
        # replace by one if greater than threshold, else by zero

        # "moving" threshold, which will be selected to
        # guarantee a ratio of ones in X after thresholding
        is_moving = params['is_moving']
        if is_moving:
            one_ratio = params['one_ratio']
            vals = X.flatten()
            vals = vals[np.argsort(vals)]
            value = vals[-int(one_ratio * len(vals)) - 1]
        # otherwise, use a given fixed threshold
        else:
            value = params['value']
        X = X > value
        X = floatX(X)
        return X
    elif name == 'onehot':
        # the max value along an axis gets replaced by 1, the others
        # to zero
        axis = get_axis(params['axis'])
        return (X == X.max(axis=axis, keepdims=True))
    elif name == 'multinomial':
        # sample from multinomial distribution by using an axis
        # as categories.
        axis = get_axis(params['axis'])
        # put the selected axis at the end, that is,
        # if axis==2 and shape of X is (nb_examples, nb_features, nb_timesteps)
        # then it would be transformed to (nb_examples, nb_timesteps, nb_features)
        shape = list(range(len(X.shape)))
        shape[axis], shape[-1] = shape[-1], shape[axis]
        shape = tuple(shape)
        X = X.transpose(shape)
        # flatten it to (nb_examples * nb_timestems, nb_features)
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))
        # sample from each example and onehotit
        for x in X:
            cat = rng.choice(np.arange(x.shape[0]), p=x)
            x[:] = 0
            x[cat] = 1
        X = X.reshape(orig_shape)
        X = X.transpose(shape)
        return X
    elif name == 'none':
        return X
    else:
        raise ValueError('Unknown binarization method  : {}'.format(name))


def get_method(name):
    return {'iterative_refinement': _iterative_refinement}[name]


def _report_image_reconstruction(cb):
    model = cb.model
    data_iterators = cb.data_iterators
    outdir = cb.outdir
    epoch = cb.epoch
    transformers = cb.transformers
    data = next(data_iterators['train'](batch_size=128))
    if hasattr(model, 'get_input_col'):
        X = model.get_input_col(data)
    else:
        X = data['X']
    X_rec = model.predict(X)
    X_rec = inverse_transform_one(X_rec, transformers)
    X = inverse_transform_one(data['X'], transformers)
    img = _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images_default)
    folder = os.path.join(outdir, 'recons')
    mkdir_path(folder)
    filename = os.path.join(folder, '{:05d}.png'.format(epoch))
    imsave(filename, img)


def _report_image_features(cb):
    model = cb.model
    epoch = cb.epoch
    outdir = cb.outdir

    # this can happen with input is discretized
    # into a number of colors. in that case the
    # color channel can be any integer. ignore that
    # case.
    if model.input_shape[1] not in (1, 3):
        return

    for layer in get_layers(model):
        if hasattr(layer, 'W'):
            W = layer.W.get_value()
            try:
                img = reshape_to_images(W, input_shape=model.input_shape[1:])
            except ValueError:
                continue
            img = grid_of_images_default(img, normalize=True)
            folder = os.path.join(outdir, 'features_{}'.format(layer.name))
            mkdir_path(folder)
            filename = os.path.join(folder, '{:05d}.png'.format(epoch))
            imsave(filename, img)
        else:
            pass


def _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images_default):
    X = grid_of_images(X)
    X_rec = grid_of_images(X_rec)
    img = horiz_merge(X, X_rec)
    return img
