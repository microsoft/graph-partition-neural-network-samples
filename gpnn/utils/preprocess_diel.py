# This file is largely based on the one released in:
# http://cs.cmu.edu/~zhiliny/data/diel_data.tar.gz.

import os
import argparse
import numpy as np
from scipy import sparse
from collections import defaultdict as dd
from gpnn.utils import gpnn_pickle as pickle
from gpnn.utils.logger import get_logger

logger = get_logger()


CATS = ["disease", "drug", "ingredient", "symptom"]
FEATURE_SAVED = False


def save_sparse_csr(filename, array):
  np.savez(
      filename,
      data=array.data,
      indices=array.indices,
      indptr=array.indptr,
      shape=array.shape)


def load_sparse_csr(filename):
  loader = np.load(filename)
  return sparse.csr_matrix(
      (loader["data"], loader["indices"], loader["indptr"]),
      shape=loader["shape"],
      dtype=np.float32)


def read_features(filename):
  logger.info("reading features")
  features, f_num = {}, 0
  for line in open(filename):
    inputs = line.strip().split()
    features[inputs[0]] = []
    for t in inputs[1:]:
      tt = int(t)
      f_num = max(f_num, tt + 1)
      features[inputs[0]].append(tt)
  return features, f_num


def read_cites(filename):
  logger.info("reading cites")
  cites, s_graph = [], dd(list)
  for i, line in enumerate(open(filename)):
    if i % 100000 == 0:
      logger.info("reading cites {}".format(i))
    inputs = line.strip().split()
    cites.append((inputs[1], inputs[2]))
    s_graph[inputs[2]].append(inputs[1])
    s_graph[inputs[1]].append(inputs[2])
  return cites, s_graph


def read_sim_dict(filename):
  logger.info("reading sim_dict")
  sim_dict = dd(list)
  for i, line in enumerate(open(filename)):
    inputs = line.strip().split()
    sim_dict[inputs[0]].append(inputs[1])
  return sim_dict


def read_train_labels(filename):
  ret = []
  for line in open(filename):
    inputs = line.strip().split()
    ret.append(inputs[2])
  return ret


def read_test_labels(filename):
  ret = []
  for line in open(filename):
    ret.append(line.strip().replace(" ", "_"))
  return ret


def add_index(index, cnt, key):
  if key in index: return cnt
  index[key] = cnt
  return cnt + 1


def construct_graph(train_id, test_id, cites):
  id2index, cnt = {}, 0
  for id in train_id:
    cnt = add_index(id2index, cnt, id)
  for id in test_id:
    cnt = add_index(id2index, cnt, id)
  graph = dd(list)
  for id1, id2 in cites:
    cnt = add_index(id2index, cnt, id1)
    cnt = add_index(id2index, cnt, id2)
    i, j = id2index[id1], id2index[id2]
    graph[i].append(j)
    graph[j].append(i)
  return graph, id2index


def construct_x_y(ents, in_labels, features, f_num):
  row, col = [], []
  for i, ent in enumerate(ents):
    for f_ind in features[ent]:
      row.append(i)
      col.append(f_ind)
  data = np.ones(len(row), dtype=np.float32)
  x = sparse.coo_matrix(
      (data, (row, col)), shape=(len(ents), f_num), dtype=np.float32).tocsr()
  y = np.zeros((len(ents), len(CATS)), dtype=np.int32)
  for i, ent in enumerate(ents):
    for j, cat in enumerate(CATS):
      if ent in in_labels[cat]:
        y[i, j] = 1
  return x, y


def read_test_cov(filename):
  logger.info("reading test cov")
  test_cov = {}
  for cat in CATS:
    test_cov[cat] = []
  for line in open(filename):
    inputs = line.strip().split()
    test_cov[CATS[int(inputs[1]) - 1]].append(inputs[0])
  return test_cov


def run(folder, run_num, features, f_num, cites, s_graph, sim_dict):
  train_list, in_labels = set(), dd(set)
  for cat in CATS:
    logger.info("processing {}".format(cat))
    train_item = read_train_labels(
        "{}/{}/{}_devel_50p_proppr_seed_forTrainList".format(
            folder, run_num, cat))
    for item in train_item:
      for l in s_graph[item]:
        train_list.add(l)
        in_labels[cat].add(l)

  test_list = set()
  for l, _ in cites:
    if l not in train_list:
      test_list.add(l)
  train_list = list(train_list)
  test_list = list(test_list)

  if not FEATURE_SAVED:
    logger.info("constructing training")
    x, y = construct_x_y(train_list, in_labels, features, f_num)
    logger.info("constructing test")
    tx, ty = construct_x_y(test_list, in_labels, features, f_num)

    logger.info("saving")
    save_sparse_csr(
        os.path.join(folder, "{}".format(run_num), "{}.x".format(run_num)), x)
    save_sparse_csr(
        os.path.join(folder, "{}".format(run_num), "{}.tx".format(run_num)), tx)
    np.save(
        os.path.join(folder, "{}".format(run_num), "{}.y".format(run_num)), y)
    np.save(
        os.path.join(folder, "{}".format(run_num), "{}.ty".format(run_num)), ty)
  else:
    logger.info("loading")
    x = load_sparse_csr("{}.x.npz".format(run_num))
    tx = load_sparse_csr("{}.tx.npz".format(run_num))
    y = np.load("{}.y.npy".format(run_num))
    ty = np.load("{}.ty.npy".format(run_num))

  logger.info(x.shape, y.shape)
  logger.info(tx.shape, ty.shape)

  logger.info("constructing graph")
  graph, id2index = construct_graph(train_list, test_list, cites)

  test_cov = read_test_cov(
      os.path.join(folder, "{}/coverage_eva_multiAdded".format(run_num)))

  pickle.dump([graph, id2index],
              open(
                  os.path.join(folder, "{}".format(run_num),
                               "{}_graph.p".format(run_num)), "wb"))
  pickle.dump(train_list,
              open(
                  os.path.join(folder, "{}".format(run_num),
                               "{}_train_list.p".format(run_num)), "wb"))
  pickle.dump(test_list,
              open(
                  os.path.join(folder, "{}".format(run_num),
                               "{}_test_list.p".format(run_num)), "wb"))
  pickle.dump(test_cov,
              open(
                  os.path.join(folder, "{}".format(run_num),
                               "{}_test_cov.p".format(run_num)), "wb"))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Preprocess DIEL data")
  parser.add_argument(
      "-d",
      "--dir",
      type=str,
      default="./data/diel",
      required=True,
      help="data folder")
  args = parser.parse_args()

  if not FEATURE_SAVED:
    features, f_num = read_features(args.dir + "/list_features.txt")
  else:
    features, f_num = None, None
  cites, s_graph = read_cites(args.dir + "/hasItem.cfacts")
  sim_dict = read_sim_dict(args.dir + "/sim.dict")

  for i in range(10):
    logger.info("=" * 80)
    logger.info("Preprocess {:02d}-th split".format(i))
    run(args.dir, i, features, f_num, cites, s_graph, sim_dict)
