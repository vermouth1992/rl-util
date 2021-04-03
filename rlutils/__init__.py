import os

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'VERSION.txt')) as f:
    __version__ = f.read()
