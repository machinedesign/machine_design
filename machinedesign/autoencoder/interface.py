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
from ..common import WrongModelFamilyException
from ..common import custom_objects
from ..viz import reshape_to_images
from ..viz import grid_of_images_default
from ..viz import horiz_merge
from ..callbacks import DoEachEpoch
from ..transformers import inverse_transform_one
from . import model_builders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_builders = object_to_dict(model_builders)

def train(params):
    family = params['family']
    if family != 'autoencoder':
        raise WrongModelFamilyException("expected family to be 'autoencoder', got {}".format(family))

    # Callbacks
    report_callbacks = []
    domain_specific = params['report'].get('domain_specific')
    if domain_specific:
        if 'image_reconstruction' in domain_specific:
            report_callbacks.append(DoEachEpoch(_report_image_reconstruction))
        if 'image_features' in domain_specific:
            report_callbacks.append(DoEachEpoch(_report_image_features))

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

    # Initialize the reconstructions
    shape = model.input_shape[1:]
    X = np.empty((N, nb_iter + 1,) + shape)
    X = floatX(X)
    X[:, 0] = np.random.uniform(size=(N,) + shape)

    # Build apply function
    reconstruct = minibatcher(model.predict, batch_size=batch_size)

    # reconstruction loop
    for i in (range(1, nb_iter + 1)):
        print('Iteration {}'.format(i))
        sprev = X[:, i - 1]
        s = sprev
        s = _apply_noise(noise_name, noise_params, s)
        s = reconstruct(s)
        s = _apply_binarization(binarize_name, binarize_params, s)
        X[:, i] = s
        score = float(np.abs(X[:, i] - X[:, i - 1]).mean())
        print('Mean absolute error : {:.3f}'.format(score))
        if score == 0:
            print('Stopping at iteration {}/{} because score is 0'.format(i, nb_iter))
            X = X[:, 0:i+1]
            break
    mkdir_path(folder)
    filename = os.path.join(folder, 'generated.npz')
    np.savez_compressed(filename, full=X, generated=X[:, -1])

def _apply_noise(name, params, X):
    if name == 'masking':
        noise_pr = params['proba']
        X = (np.random.uniform(size=X.shape) <= (1 - noise_pr)) * X
        X = floatX(X)
        return X
    elif name == 'none':
        return X
    else:
        raise ValueError('Unknown noise method : {}'.format(name))

def _apply_binarization(name, params, X):
    if name == 'sample_bernoulli':
        X = np.random.binomial(n=1, p=X, size=X.shape)
        return X
    elif name == 'binary_threshold':
        is_moving = params['is_moving']
        if is_moving:
            one_ratio = params['one_ratio']
            vals = X.flatten()
            vals = vals[np.argsort(vals)]
            value = vals[-int(one_ratio * len(vals)) - 1]
        else:
            value = params['value']
        X = X > value
        X = floatX(X)
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
    params = cb.params
    epoch = cb.epoch
    transformers = cb.transformers

    data = next(data_iterators['train'](batch_size=128))
    X = data['X']
    X_rec = model.predict(X)
    X_rec = inverse_transform_one(X_rec, transformers)
    X = inverse_transform_one(X, transformers)
    img = _get_input_reconstruction_grid(X, X_rec, grid_of_images=grid_of_images_default)
    folder = os.path.join(params['report']['outdir'], 'recons')
    mkdir_path(folder)
    filename = os.path.join(folder, '{:05d}.png'.format(epoch))
    imsave(filename, img)

def _report_image_features(cb):
    model = cb.model
    epoch = cb.epoch
    params = cb.params
    for layer in model.layers:
        if hasattr(layer, 'W'):
            W = layer.W.get_value()
            if model.input_shape[1:] not in (1, 3):
                continue
            try:
                img = reshape_to_images(W, input_shape=model.input_shape[1:])
            except ValueError:
                continue
            img = grid_of_images_default(img, normalize=True)
            folder = os.path.join(params['report']['outdir'], 'features_{}'.format(layer.name))
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
