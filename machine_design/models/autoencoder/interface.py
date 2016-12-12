import model as model_builders

def train(params):
    model_name = params['model']['name']
    model_params = params['model']['params']
    model_builder = getattr(model_builders, model_name)
    model = model_builder(model_params)

def load(filename):
    pass


def generate(params):
    pass
