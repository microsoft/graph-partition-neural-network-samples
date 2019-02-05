# A compatability wrapper for the pickle module. cPickle was removed
# in Python 3.

try:
  import cPickle as pickle
except ModuleNotFoundError:
  import pickle


def load(*args, **kwargs):
  return pickle.load(*args, **kwargs)


def dump(*args, **kwargs):
  return pickle.dump(*args, **kwargs)
