import os
import numpy as np
from scipy import sparse
from scipy.sparse import vstack

from gpnn.factory import ReaderFactory
from gpnn.data.gpnn_data import GPNNData
from gpnn.utils import gpnn_pickle as pickle
from gpnn.utils.logger import get_logger
from gpnn.utils.preprocess_diel import load_sparse_csr, read_cites, read_sim_dict
from gpnn.utils.reader_helper import read_idx_file

logger = get_logger()
register = ReaderFactory.register
EPS = np.finfo(np.float32).eps


@register("GPNNReader")
class GPNNReader(object):

  def __init__(self, param):
    self._dataset_name = param["dataset_name"]
    self._num_edgetype = param["num_edgetype"]
    self._num_nodes_val = param["num_valid"]
    self._use_rand_split = param["use_rand_split"]
    self._data_folder = param["data_folder"]
    self._seed = param["seed"]
    self._idx_type = np.int32
    self._num_cluster = param["num_cluster"]
    self._decomp_method = param["decomp_method"]
    self._param = param

    if self._dataset_name == "nell":
      self._feat_dim_raw = param["feat_dim_raw"]
      self._label_rate = param["label_rate"]

    if self._dataset_name == "diel":
      self._split_id = param["split_id"]
      self._split_folder = os.path.join(param["data_folder"],
                                        "{}".format(self._split_id))

    self._suffixes = [
        "x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"
    ]
    self._var_names = [
        "train_feat", "train_label", "test_feat", "test_label", "all_feat",
        "all_label", "graph", "test_id"
    ]

    assert (self.num_edgetype == 1)

  def _preprocess_features(self, features, norm_method=None):
    """ Normalize feature matrix """

    if norm_method == "L1":
      # L1 norm
      features /= (features.sum(1, keepdims=True) + EPS)

    elif norm_method == "L2":
      # L2 norm
      features /= (np.sqrt(np.square(features).sum(1, keepdims=True)) + EPS)

    elif norm_method == "std":
      # Standardize
      std = np.std(features, axis=0, keepdims=True)
      features -= np.mean(features, 0, keepdims=True)
      features /= (std + EPS)
    else:
      # nothing
      pass

    return features

  def _get_data_dict(self):
    data_dict = {}

    if self.dataset_name == "diel":
      # load diel dataset
      data_dict["category"] = ['disease', 'drug', 'ingredient', 'symptom']

      train_feat = load_sparse_csr(
          os.path.join(self._split_folder, "{}.x.npz".format(self._split_id)))
      test_feat = load_sparse_csr(
          os.path.join(self._split_folder, "{}.tx.npz".format(self._split_id)))
      train_label = np.load(
          os.path.join(self._split_folder, "{}.y.npy".format(self._split_id)))
      test_label = np.load(
          os.path.join(self._split_folder, "{}.ty.npy".format(self._split_id)))

      self._node_feat = vstack([train_feat, test_feat], format="csr")
      self._node_gt_label = np.argmax(
          np.vstack([train_label, test_label]), axis=1)
      data_dict["graph"], data_dict["id2index"] = pickle.load(
          open(
              os.path.join(self._split_folder, "{}_graph.p".format(
                  self._split_id))))
      data_dict["train_list"] = pickle.load(
          open(
              os.path.join(self._split_folder, "{}_train_list.p".format(
                  self._split_id))))
      data_dict["test_list"] = pickle.load(
          open(
              os.path.join(self._split_folder, "{}_test_list.p".format(
                  self._split_id))))
      data_dict["test_cov"] = pickle.load(
          open(
              os.path.join(self._split_folder, "{}_test_cov.p".format(
                  self._split_id))))

      data_dict["cites"], data_dict["s_graph"] = read_cites(
          self._data_folder + '/hasItem.cfacts')
      data_dict["sim_dict"] = read_sim_dict(self._data_folder + '/sim.dict')

      self._train_feat_idx = [
          data_dict["id2index"][xx] for xx in data_dict["train_list"]
      ]
      data_dict["train_idx"] = np.array(self._train_feat_idx, dtype=np.int32)

      # split data
      num_nodes_train = len(data_dict["train_list"])
      num_nodes_val = int(np.floor(
          num_nodes_train / 5.0))  # keep train/val ratio to be 4:1

      prng = np.random.RandomState(self.seed)
      perm_idx = prng.permutation(num_nodes_train)
      split_val_list = [
          data_dict["train_list"][xx] for xx in perm_idx[:num_nodes_val]
      ]
      split_train_idx = [
          data_dict["train_list"][xx] for xx in perm_idx[num_nodes_val:]
      ]

      self._node_idx = sorted(data_dict["graph"].keys())
      self._train_idx = [data_dict["id2index"][xx] for xx in split_train_idx]
      self._val_idx = [data_dict["id2index"][xx] for xx in split_val_list]
      self._train_idx += self._val_idx
      self._test_idx = [
          data_dict["id2index"][xx] for xx in data_dict["test_list"]
      ]
      self._feat_idx = self._train_feat_idx + self._test_idx
    else:
      # load other datasets
      if self.dataset_name == "nell":
        base_name = ".".join(["ind", self.dataset_name, str(self.label_rate)])
      else:
        base_name = ".".join(["ind", self.dataset_name])

      for nn, ss in zip(self._var_names, self._suffixes):
        if nn == "test_id":
          data_dict[nn] = read_idx_file(
              os.path.join(self._data_folder, ".".join([base_name, ss])))
        else:
          data_dict[nn] = pickle.load(
              open(os.path.join(self._data_folder, ".".join([base_name, ss]))))

      self._node_id = sorted(data_dict["graph"].keys())
      self._node_idx = range(len(self._node_id))
      self._num_feat = data_dict["all_feat"].shape[0]
      assert (np.all(np.array(self._node_id) == np.array(self._node_idx)))

    return data_dict

  def _concat_data(self, data_dict):
    self._feat_all = self._preprocess_features(
        np.concatenate(
            [data_dict["all_feat"].toarray(), data_dict["test_feat"].toarray()],
            axis=0),
        norm_method=None)
    self._label_all = np.argmax(
        np.concatenate(
            [data_dict["all_label"], data_dict["test_label"]], axis=0),
        axis=1)
    self._idx_all = np.concatenate(
        [np.arange(self.num_feat),
         np.array(data_dict["test_id"])], axis=0)

  def _get_split(self):
    if self.use_rand_split:
      prng = np.random.RandomState(self.seed)
      perm_idx = prng.permutation(len(self._idx_all))
      split_train_idx = perm_idx[:self.num_nodes_train]
      split_val_idx = perm_idx[self.num_nodes_train:
                               self.num_nodes_train + self.num_nodes_val]
      split_test_idx = perm_idx[
          self.num_nodes_train + self.num_nodes_val:
          self.num_nodes_train + self.num_nodes_val + self.num_nodes_test]
    else:
      # using the inductive split as the Planetoid paper
      split_train_idx = np.arange(self.num_nodes_train)
      split_val_idx = np.arange(self.num_nodes_train,
                                self.num_nodes_train + self.num_nodes_val)
      split_test_idx = np.arange(self.num_feat,
                                 self.num_feat + self.num_nodes_test)

    self._train_idx = self._idx_all[split_train_idx]
    self._val_idx = self._idx_all[split_val_idx]
    self._test_idx = self._idx_all[split_test_idx]

  def _pack_data(self, data_dict):
    data = GPNNData(
        data_dict["graph"],
        train_idx=self.train_idx,
        num_edgetype=self.num_edgetype,
        num_cluster=self._num_cluster,
        decomp_method=self._decomp_method,
        seed=self.seed)

    data.get_graph_partition()
    data.get_prop_index(data.cluster_graphs, data.cut_graph)
    self._param["num_node_cut"] = data.num_node_cut
    self._param["cluster_size"] = data.cluster_size
    logger.info("cluster_size = {}".format(self._param["cluster_size"]))

    data.train_idx = np.array([data._pos_map[xx] for xx in self.train_idx])
    data.val_idx = np.array([data._pos_map[xx] for xx in self.val_idx])
    data.test_idx = np.array([data._pos_map[xx] for xx in self.test_idx])

    if self.dataset_name == "diel":
      # pack up diel data
      data.sim_dict = data_dict["sim_dict"]
      data.test_cov = data_dict["test_cov"]
      data.test_list = data_dict["test_list"]
      data.s_graph = data_dict["s_graph"]
      data.category = data_dict["category"]

      feat_idx = np.concatenate([
          np.array([data._pos_map[xx]
                    for xx in self._train_feat_idx]), data.test_idx
      ])

      data.node_gt_label = np.zeros(self._num_nodes, dtype=np.int32)
      data.node_gt_label[feat_idx] = self.node_gt_label

      row_idx, col_idx, values = sparse.find(self.node_feat)
      row_idx = np.array([feat_idx[xx] for xx in row_idx])
      data.node_feat_indices = np.stack(
          [row_idx.astype(np.int64),
           col_idx.astype(np.int64)], axis=1)
      data.node_feat_values = values.astype(np.float32)
      data.node_feat_shape = np.array(
          [self._num_nodes, self._feat_dim], dtype=np.int64)
    else:
      # pack up other data
      if self.dataset_name == "nell":
        self._node_feat = np.zeros(
            [self._num_nodes, self._feat_dim_raw], dtype=np.float32)
      else:
        self._node_feat = np.zeros(
            [self._num_nodes, self._feat_dim], dtype=np.float32)

      self._node_feat[self._idx_all, :] = self._feat_all
      row_idx, col_idx, values = sparse.find(sparse.csr_matrix(self._node_feat))

      # construc label
      feat_idx = np.array([data._pos_map[xx] for xx in self._idx_all])
      data.node_gt_label = np.zeros(self._num_nodes, dtype=np.int32)
      data.node_gt_label[feat_idx] = self._label_all

      # concatenate 1-hot feature for nell dataset
      if self.dataset_name == "nell":
        idx_entity = self._idx_all
        idx_relation = list(set(self._node_idx) - set(self._idx_all.tolist()))
        self._one_hot_dim = len(idx_relation)

        row_idx = row_idx.tolist() + idx_relation
        row_idx = np.array([data._pos_map[xx] for xx in row_idx])

        col_idx = np.array(col_idx.tolist() + np.arange(
            self._feat_dim_raw,
            self._feat_dim_raw + self._one_hot_dim).tolist())
        values = np.array(values.tolist() + np.ones([self._one_hot_dim
                                                    ]).tolist())

        data.node_feat_shape = np.array(
            [self._num_nodes, self._feat_dim_raw + self._one_hot_dim],
            dtype=np.int64)
      else:
        row_idx = np.array([data._pos_map[xx] for xx in row_idx])
        data.node_feat_shape = np.array(
            [self._num_nodes, self._feat_dim], dtype=np.int64)

      data.node_feat_indices = np.stack(
          [row_idx.astype(np.int64),
           col_idx.astype(np.int64)], axis=1)
      data.node_feat_values = values.astype(np.float32)

    return data

  def read(self):
    data_dict = self._get_data_dict()

    # get statistics
    self._num_nodes = len(data_dict["graph"].keys())
    if self._dataset_name != "diel":
      self._num_nodes_train = data_dict["train_feat"].shape[0]
      self._num_nodes_test = data_dict["test_feat"].shape[0]
      self._feat_dim = data_dict["train_feat"].shape[1]
    else:
      self._num_nodes_train = len(self._train_idx)
      self._num_nodes_val = len(self._val_idx)
      self._num_nodes_test = len(self._test_idx)
      self._feat_dim = self.node_feat.shape[1]

    logger.info("Number of training nodes = {}".format(self.num_nodes_train))
    logger.info("Number of validation nodes = {}".format(self.num_nodes_val))
    logger.info("Number of testing nodes = {}".format(self.num_nodes_test))
    logger.info("Number of nodes in the graph = {}".format(self._num_nodes))
    logger.info("Dimension of feature = {}".format(self._feat_dim))

    # concat data & get split
    if self._dataset_name != "diel":
      self._concat_data(data_dict)
      self._get_split()

    # pack up data
    return self._pack_data(data_dict)

  @property
  def dataset_name(self):
    return self._dataset_name

  @property
  def label_rate(self):
    return self._label_rate

  @property
  def num_feat(self):
    return self._num_feat

  @property
  def use_rand_split(self):
    return self._use_rand_split

  @property
  def seed(self):
    return self._seed

  @property
  def num_nodes_train(self):
    return self._num_nodes_train

  @property
  def num_nodes_val(self):
    return self._num_nodes_val

  @property
  def num_nodes_test(self):
    return self._num_nodes_test

  @property
  def num_edgetype(self):
    return self._num_edgetype

  @property
  def node_idx(self):
    return self._node_idx

  @property
  def train_idx(self):
    return self._train_idx

  @property
  def val_idx(self):
    return self._val_idx

  @property
  def test_idx(self):
    return self._test_idx

  @property
  def node_feat(self):
    return self._node_feat

  @property
  def node_gt_label(self):
    return self._node_gt_label

  @property
  def node_feat_indices(self):
    return self._node_feat_indices

  @property
  def node_feat_shape(self):
    return self._node_feat_shape

  @property
  def node_feat_values(self):
    return self._node_feat_values
