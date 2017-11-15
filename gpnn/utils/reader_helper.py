import os
import glob
import scipy
import numpy as np

from gpnn.utils.logger import get_logger

logger = get_logger()


def read_idx_file(file_name):
  idx = []
  with open(file_name) as f:
    for line in f:
      idx += [int(line)]

  return idx


def get_filenames_from_dir(dirname, pattern):
  """ Return a list of files in directory dirname with some pattern (e.g., "Node_*.txt") """
  return sorted(glob.glob(os.path.join(dirname, pattern)))


def read_txt_file(file_name):
  """ Read the data inside a file to a dictionary """

  data_dict = {}
  key_list = []

  with open(file_name, "r") as file:
    for count, line in enumerate(file):
      # the first line should contain the key of each column
      if count == 0:
        key_list = line.split()
        num_key = len(key_list)
        for key in key_list:
          data_dict[key] = []
      else:
        token_list = line.split()

        if len(token_list) < num_key:
          # sanity check
          logger.warning("There are missing tokens")
          # handle corner cases (missing token) in node file
          data_dict[key_list[0]].append(token_list[0])
          data_dict[key_list[1]].append("missing_token")
          data_dict[key_list[2]].append(token_list[1])
          # print("file_name = {}".format(file_name))
          # raw_input("wait")
        elif len(token_list) > num_key:
          logger.warning("There is something wrong!")
          logger.warning("file name = {}".format(file_name))
          raw_input("wait")
        else:
          for idx, token in enumerate(token_list):
            data_dict[key_list[idx]].append(token)

  return data_dict


def gen_split_idx(num_all, num_train, num_val, num_test, seed=1234):
  """Generate train/val/test split indices 

  Args:
    num_all: total number of data
    num_train: number of training data
    num_val: number of validation data
    num_test: number of testing data
    seed: seed of random number generator

  Returns:
    train_idx: index of training data
    val_idx: index of validation data
    test_idx index of testing data
  """

  assert num_train + num_val + num_test <= num_all
  prng = np.random.RandomState(seed)
  perm_idx = prng.permutation(num_all)

  train_idx = perm_idx[:num_train]
  val_idx = perm_idx[num_train:num_train + num_val]
  test_idx = perm_idx[num_train + num_val:num_train + num_val + num_test]

  return train_idx, val_idx, test_idx


def read_list_from_file(filename):
  data_list = []

  with open(filename, "r") as ff:
    for line in ff:
      data_list += [line.rstrip()]

  return data_list


def read_csv_file(file_name):
  with open(file_name, "r") as ff:
    count = 0

    for line in ff:
      line_str = line.rstrip().split(",")

      if count == 0:
        num_col = len(line_str)
        results = [[] for _ in xrange(num_col)]

      for ii, xx in enumerate(line_str):
        results[ii] += [int(xx)]

      count += 1

  return results
