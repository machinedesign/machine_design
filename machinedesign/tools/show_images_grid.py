import click
import numpy as np
from skimage.io import imsave

from machinedesign.viz import grid_of_images_default

@click.command()
@click.option('--filename', required=True)
@click.option('--dest', required=True)
def show(filename, dest):
    data = np.load(filename)
    X = data['generated']
    img = grid_of_images_default(X)
    imsave(dest, img)

if __name__ == '__main__':
    show()
