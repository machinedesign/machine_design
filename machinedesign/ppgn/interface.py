"""
Implementation of the generation procedure of Plug and Play generative networks.
See paper at <https://arxiv.org/pdf/1612.00005v1.pdf>.
"""
import numpy as np
import os
import logging

from keras.models import load_model
import keras.backend as K

from ..common import custom_objects
from ..common import mkdir_path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train(params):
    # no need for training ppgn
    pass


def generate(params, custom_objects=custom_objects):
    nb_samples = params['nb_samples']
    nb_iter = params['nb_iter']
    folder = params['folder']
    autoencoder_filename = params['autoencoder']
    classifier_filename = params['classifier']
    genetor_filename = params['generator']
    epsilon1 = params['epsilon1']
    epsilon2 = params['epsilon2']
    epsilon3 = params['epsilon3']

    class_indices = params['class_indices']
    seed = params['seed']

    rng = np.random.RandomState(seed)

    autoencoder = load_model(autoencoder_filename, custom_objects=custom_objects)
    generator = load_model(genetor_filename, custom_objects=custom_objects)
    classifier = load_model(classifier_filename, custom_objects=custom_objects)

    # follow paper notation (see eq. 11 of paper)
    C = classifier
    G = generator
    H = autoencoder

    e1 = epsilon1
    e2 = epsilon2
    e3 = epsilon3

    # check compositionality constraints
    msg = "'autoencoder' expects input_shape == output_shape, input_shape is {} while output_shape is {}"
    msg = msg.format(H.input_shape, H.output_shape)
    assert H.input_shape == H.output_shape, msg

    msg = "'generator' input_shape ({}) must be 'autoencoder' output_shape ({}), they are different"
    msg = msg.format(G.input_shape, autoencoder.output_shape)
    assert G.input_shape == autoencoder.output_shape, msg

    msg = "'generator' output_shape ({}) must be 'classifier' input_shape ({}), they are different"
    msg = msg.format(G.output_shape, autoencoder.input_shape)
    assert G.output_shape == classifier.input_shape, msg

    h = K.placeholder(autoencoder.input_shape)

    # get the two needed terms (see eq. 11 of paper)

    # diff between reconstruction and current feature space, shape : e.g (nb_examples, hid)
    h_rec = H(h) - h
    x = G(h)  # generated objects from feature space, shape : e.g (nb_examples, c, h, w)
    y = K.log(C(x))  # predicted probabilities of the classifier, shape : e.g (nb_examples, nb_classes)
    y_prob = C(x)
    # feature space gradients, shape : e.g (nb_examples, hid)
    dc_dh = K.gradients(y[:, class_indices].sum(), h)

    delta_h = e1 * h_rec + e2 * dc_dh

    delta_h_func = K.function([h, K.learning_phase()], delta_h)
    # just a wrapper of the above to make it more readable
    get_delta_h = lambda hid: delta_h_func([hid, 0])

    generate_func = K.function([h, K.learning_phase()], x)
    # just a wrapper of the above to make it more readable
    get_generated = lambda hid: generate_func([hid, 0])

    classify_func = K.function([x, K.learning_phase()], y_prob)
    # just a wrapper of the above to make it more readable
    predict_probas = lambda inp: classify_func([inp, 0])

    hid = rng.normal(0, 1, size=(nb_iter + 1, nb_samples) + H.output_shape[1:])
    generated = np.zeros((nb_iter + 1, nb_samples) + G.output_shape[1:])
    probas = np.zeros((nb_iter + 1, nb_samples) + C.output_shape[1:])
    for i in range(1, nb_iter + 1):
        logger.info('Iteration {}'.format(i))
        # update feature space and generate
        noise = (rng.normal(0, e3, size=hid[i - 1].shape) if e3 > 0 else 0)
        dh = (get_delta_h(hid[i - 1]) + noise)
        alpha = 1. / (np.abs(dh).mean())
        hid[i] = hid[i - 1] + alpha * dh
        # important to keep this in the "domain", how to do that in general ? we
        # need a dataset or the user can specify it
        hid[i] = np.clip(hid[i], 0, 30)
        generated[i] = get_generated(hid[i])
        probas[i] = predict_probas(generated[i])
        # report classifier probas
        logger.debug(probas[i][:, class_indices])
    mkdir_path(folder)
    filename = os.path.join(folder, 'generated.npz')
    np.savez_compressed(filename, generated=generated, classifier_probas=probas)
