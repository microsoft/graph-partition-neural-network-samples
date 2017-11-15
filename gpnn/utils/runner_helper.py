import os
import numpy as np

def mkdir(folder):
  if not os.path.isdir(folder):
    os.mkdir(folder)

def convert_to_np_array_or_not(data):
  if not isinstance(data, np.ndarray):
    data = np.array(data)

  return data


class EarlyStopper(object):
  """ 
    Check whether the early stop condition (always 
    observing decrease in a window of time steps) is met.

    Usage:
    my_stopper = EarlyStopper([0, 0], 1)
    is_stop = my_stopper.tick([-1,-1]) # returns True
  """

  def __init__(self, init_val, win_size=10, is_decrease=True):
    if not isinstance(init_val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    self._win_size = win_size
    self._num_val = len(init_val)
    self._val = [[False] * win_size for _ in xrange(self._num_val)]
    self._last_val = init_val[:]
    self._comp_func = (lambda x, y: x < y) if is_decrease else (
        lambda x, y: x >= y)

  def tick(self, val):
    if not isinstance(val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    assert len(val) == self._num_val

    for ii in xrange(self._num_val):
      self._val[ii].pop(0)

      if self._comp_func(val[ii], self._last_val[ii]):
        self._val[ii].append(True)
      else:
        self._val[ii].append(False)

      self._last_val[ii] = val[ii]

    is_stop = all([all(xx) for xx in self._val])

    return is_stop
