"""
this module contains a common set of general purpose utility functions
"""
import os
import csv
import time


def object_to_dict(obj):
    """return the attributes of an object"""
    return obj.__dict__


def mkdir_path(path):
    """
    Create folder in `path` silently: if it exists, ignore, if not
    create all necessary folders reaching `path`
    """
    if not os.access(path, os.F_OK):
        os.makedirs(path)


def write_csv(iterable, filename):
    """
    write a list of dicts into a csv file
    (like pandas.to_csv(...) but I didnt want to add that dependency
     just for that)

    Parameters
    ----------

    iterable : iterable of dict
        this will constitute the rows of the csv file.
        the header will be the keys of the dicts.
    filename : str
        filename where to write the content
    """
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=iterable[0].keys())
        writer.writeheader()
        writer.writerows(iterable)


axes = {
    'time': 1,
    'channels': 1,
    'height': 2,
    'width': 3,
    'time_features': 2,
    'features': 1
}


def get_axis(axis):
    """ get axis index from a human readable name"""
    return axes.get(axis, axis)


def print_time(func):

    def func_(*args, **kwargs):
        print('Applying {}...'.format(func.__name__))
        t0 = time.time()
        val = func(*args, **kwargs)
        delta_t = time.time() - t0
        print('{:.3f}s elapsed'.format(delta_t))
        return val
    return func_
