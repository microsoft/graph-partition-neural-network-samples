import os
import numpy as np
from scipy import sparse

from gpnn.factory import ReaderFactory
from gpnn.data.gpnn_data import GPNNData
from gpnn.reader.gpnn_reader import GPNNReader
from gpnn.utils import gpnn_pickle as pickle
from gpnn.utils.logger import get_logger

logger = get_logger()
register = ReaderFactory.register


@register("GPNNReaderCustom")
class GPNNReaderCustom(GPNNReader):

  def __init__(self, param):
    super(GPNNReaderCustom, self).__init__(param)

    self._suffixes = ["feature", "label", "graph", "split"]
    self._var_names = ["all_feat", "all_label", "graph", "split"]

  def _get_data_dict(self):
    data_dict = {}

    for nn, ss in zip(self._var_names, self._suffixes):
      data_dict[nn] = pickle.load(
          open(
              os.path.join(self._data_folder, ".".join([self.dataset_name, ss
                                                       ]))))

    self._train_idx = np.where(data_dict["split"] == 0)[0]
    self._val_idx = np.where(data_dict["split"] == 1)[0]
    self._test_idx = np.where(data_dict["split"] == 2)[0]
    self._node_feat = data_dict["all_feat"]

    return data_dict

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

    row_idx, col_idx, values = sparse.find(sparse.csr_matrix(self._node_feat))

    # construc label
    feat_idx = np.array([data._pos_map[xx] for xx in xrange(self._num_nodes)])
    data.node_gt_label = np.zeros(self._num_nodes, dtype=np.int32)
    data.node_gt_label[feat_idx] = data_dict["all_label"]

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
    self._num_nodes_train = len(self._train_idx)
    self._num_nodes_val = len(self._val_idx)
    self._num_nodes_test = len(self._test_idx)
    self._feat_dim = self.node_feat.shape[1]

    logger.info("Number of training nodes = {}".format(self.num_nodes_train))
    logger.info("Number of validation nodes = {}".format(self.num_nodes_val))
    logger.info("Number of testing nodes = {}".format(self.num_nodes_test))
    logger.info("Number of nodes in the graph = {}".format(self._num_nodes))
    logger.info("Dimension of feature = {}".format(self._feat_dim))

    # pack up data
    return self._pack_data(data_dict)
