import numpy as np
import os
import logging
import pickle

from skimage.io import imsave

from keras.models import load_model

from ..data import floatX
from ..interface import train as train_basic
from ..common import object_to_dict
from ..common import mkdir_path
from ..common import minibatcher
from ..common import check_family_or_exception
from ..common import custom_objects
from ..viz import reshape_to_images
from ..viz import grid_of_images_default
from ..viz import horiz_merge
from ..callbacks import DoEachEpoch
from ..transformers import inverse_transform_one
from .. import model_builders
from . import model_builders as model_builders_autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_builders = object_to_dict(model_builders)
model_builders_autoencoder = object_to_dict(model_builders_autoencoder)
model_builders.update(model_builders_autoencoder)


def train(params):
    check_family_or_exception(params['family'], 'autoencoder')
    # Callbacks
    report_callbacks = []
    domain_specific = params['report'].get('domain_specific')
    if domain_specific:
        if 'image_reconstruction' in domain_specific:
            cb = DoEachEpoch(_report_image_reconstruction)
            cb.outdir = params['report']['outdir']
            report_callbacks.append(cb)
        if 'image_features' in domain_specific:
            cb = DoEachEpoch(_report_image_features)
            cb.outdir = params['report']['outdir']
            report_callbacks.append(cb)
    # Call training functions
    return train_basic(
        params,
        builders=model_builders,
        inputs='X', outputs='X',
        logger=logger,
        callbacks=report_callbacks)

def load(folder):
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
    return func(params, model, save_folder)

def _iterative_refinement(params, model, folder):
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

    # Initialize the reconstructions
    transformers = model.transformers
    # shape is shape of inputs before
    # applying the transformers (if there are transformers)
    shape = transformers[0].input_shape_ if len(transformers) else model.input_shape[1:]
    X = np.empty((N, nb_iter + 1,) + shape)
    X = floatX(X)

    rng = np.random.RandomState(seed)

    s = rng.uniform(size=(N,) + model.input_shape[1:])
    X[:, 0] = inverse_transform_one(s, transformers)

    # Build apply function
    reconstruct = minibatcher(model.predict, batch_size=batch_size)

    # reconstruction loop
    previous_score = None
    for i in (range(1, nb_iter + 1)):
        logger.info('Iteration {}'.format(i))
        s = _apply_noise(noise_name, noise_params, s, rng=rng)
        s = reconstruct(s)
        s = _apply_binarization(binarize_name, binarize_params, s, rng=rng)
        X[:, i] = inverse_transform_one(s, transformers)
        score = float(np.abs(X[:, i] - X[:, i - 1]).mean())
        logger.info('Mean absolute error : {:.5f}'.format(score))
        if previous_score and score == previous_score and stop_if_unchanged:
            logger.info('Stopping at iteration {}/{} because score did not change'.format(i, nb_iter))
            X = X[:, 0:i+1]
            break
        previous_score = score
    mkdir_path(folder)
    filename = os.path.join(folder, 'generated.npz')
    np.savez_compressed(filename, full=X, generated=X[:, -1])

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
        axis = params['axis']
        noise_pr = params['proba']
        mask = rng.uniform(size=X.shape)
        mask = (mask == mask.max(axis=axis, keepdims=True))
        shape = list(X.shape)
        shape[axis] = 1
        u = rng.uniform(size=shape) <= (1 - noise_pr)
        X = X * u  + mask * (1 - u)
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
        axis = params['axis']
        return (X == X.max(axis=axis, keepdims=True))
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

    for layer in model.layers:
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
