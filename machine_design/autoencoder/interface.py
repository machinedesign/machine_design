from ..interface import train as train_basic
import model_builders

def train(params):
    return train_basic(params, builders=model_builders)

def load(filename):
    pass

def generate(params):
    pass
