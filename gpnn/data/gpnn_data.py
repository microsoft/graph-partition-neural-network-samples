import json
import numpy as np
from collections import defaultdict

from gpnn.utils.spectral_graph_partition import spectral_clustering
from gpnn.utils.flood_fill_partition import multi_seed_flood_fill
from gpnn.utils.logger import get_logger

logger = get_logger()


class GPNNData(object):

  def __init__(self,
               graph,
               train_idx=None,
               num_edgetype=1,
               node_info=None,
               num_cluster=10,
               decomp_method="spectral_cluster",
               idx_type=np.int32,
               seed=1234):
    self._graph = graph
    self._node = sorted(graph.keys())
    self._num_edgetype = num_edgetype
    self._num_node = len(self._node)
    self._seed = seed

    self._node_info = node_info
    self._idx_type = idx_type
    self._num_cluster = num_cluster
    self._train_idx = train_idx
    self._decomp_method = decomp_method
    self._cluster_size = [0] * self._num_cluster
    self._num_node_cut = 0
    assert (self._num_cluster > 0)

  def get_graph_partition(self):
    # graph partition
    if self._num_cluster == 1:
      logger.info("No partition is need!")
      self._cluster_label = [0] * self._num_node
    else:
      if self._decomp_method == "spectral_cluster":
        logger.info("Start spectral_clustering")
        self._cluster_label = spectral_clustering(
            self._graph,
            self._num_cluster,
            is_multigraph=False,
            use_sparse=True,
            seed=self._seed)
        logger.info("Finish spectral_clustering")
      elif self._decomp_method == "multi_seed_flood_fill":
        logger.info("Start multi-seed flood fill")
        assert (self._train_idx is not None)
        labeled_node = np.zeros(self._num_node, dtype=np.int32)
        labeled_node[self._train_idx] = 1
        self._cluster_label = multi_seed_flood_fill(
            self._graph,
            self._num_cluster,
            node_label=labeled_node,
            is_multigraph=False,
            rnd_seed=self._seed)
        logger.info("Finish multi-seed flood fill")
      else:
        raise ValueError("Unsupported decomposition method!")

    # check whether there are empty clusters
    assert (len(set(self._cluster_label)) == self._num_cluster)

    # get sub-graphs
    self._cluster_graphs, self._cut_graph = self._get_subgraph(
        self._cluster_label, self._num_cluster)

  def _get_subgraph(self, cluster_label, num_cluster):
    cluster_graphs = [defaultdict(list) for _ in xrange(num_cluster)]
    cut_graph = defaultdict(list)

    for nn in self._node:
      label_nn = cluster_label[nn]

      # consider isolate node case
      if not nn in cluster_graphs[label_nn]:
        cluster_graphs[label_nn][nn] = []

      for mm in self._graph[nn]:
        label_mm = cluster_label[mm]

        if label_mm == label_nn:
          cluster_graphs[label_nn][nn] += [mm]
        else:
          cut_graph[nn] += [mm]

          if not mm in cut_graph:
            cut_graph[mm] = []

    return cluster_graphs, cut_graph

  def _get_pos_map(self, cluster_graphs):
    count = 0
    pos_map = {}  # map from original index to new index
    self._cluster_size = []

    for ii in xrange(self._num_cluster):
      nodes = sorted(cluster_graphs[ii].keys())

      for jj, nn in enumerate(nodes):
        pos_map[nn] = jj + count

      count += len(nodes)
      self._cluster_size += [len(nodes)]

    return pos_map

  def get_prop_index(self, cluster_graphs, cut_graph):
    """ Get index for mix sequential propagation """

    self._num_node_cut = len(cut_graph.keys())

    # get position map
    self._pos_map = self._get_pos_map(cluster_graphs)

    # get prop index of clusters
    self._get_cluster_idx(cluster_graphs)

    # get partition & stich index
    self._get_split_idx(cut_graph, self._pos_map)

    # get prop index of cut
    self._get_cut_idx(cut_graph)

  def _get_split_idx(self, cut, pos_map):
    """ Get index which splits nodes in cut """
    self._partition_idx = np.ones_like(self._node, dtype=np.int32)
    self._stitch_idx = [None] * 2

    for nn in self._node:
      if nn in cut:
        self._partition_idx[pos_map[nn]] = 0

    self._stitch_idx[0] = np.where(
        self._partition_idx == 0)[0].tolist()  # cut idx
    self._stitch_idx[1] = np.where(
        self._partition_idx == 1)[0].tolist()  # non-cut idx
    self._cut_idx_map = dict(
        zip(self._stitch_idx[0], range(len(self._stitch_idx[0]))))

  def _get_cluster_idx(self, cluster_graphs):
    # cluster graph index
    self._send_idx_cluster = [[[0] for _ in xrange(self._num_edgetype)]
                              for _ in xrange(self._num_cluster)
                             ]  # initial dummy send idx is 0
    self._receive_idx_cluster = [None for _ in xrange(self._num_cluster)]

    for ii in xrange(self._num_cluster):
      sub_graph = cluster_graphs[ii]
      nodes = sorted(sub_graph.keys())
      self._cluster_size[ii] = len(nodes)
      num_node = self._cluster_size[ii]
      cluster_idx_map = dict(zip(nodes, range(num_node)))
      tmp_receive_idx = [[num_node] for _ in xrange(self._num_edgetype)
                        ]  # initial dummy receive idx is num_node

      for nn in sub_graph:
        num_neighbors = len(sub_graph[nn])
        self._send_idx_cluster[ii][0] += [cluster_idx_map[nn]] * num_neighbors
        tmp_receive_idx[0] += [cluster_idx_map[mm] for mm in sub_graph[nn]]

      self._receive_idx_cluster[ii] = reduce(lambda x, y: x + y,
                                             tmp_receive_idx)

  def _get_cut_idx(self, cut_graph):
    # cut graph index
    self._send_idx_cut = [[0] for _ in xrange(self._num_edgetype)]

    #
    num_node = self._num_node_cut
    tmp_receive_idx = [[num_node] for _ in xrange(self._num_edgetype)
                      ]  # initial dummy receive idx is num_node

    for nn in cut_graph:
      num_neighbors = len(cut_graph[nn])
      self._send_idx_cut[0] += [self._cut_idx_map[self._pos_map[nn]]
                               ] * num_neighbors
      tmp_receive_idx[0] += [
          self._cut_idx_map[self._pos_map[mm]] for mm in cut_graph[nn]
      ]

    self._receive_idx_cut = reduce(lambda x, y: x + y, tmp_receive_idx)

  @property
  def num_node_cut(self):
    return self._num_node_cut

  @property
  def cluster_size(self):
    return self._cluster_size

  @property
  def cluster_graphs(self):
    return self._cluster_graphs

  @property
  def cut_graph(self):
    return self._cut_graph

  @property
  def cluster_label(self):
    return self._cluster_label

  @property
  def partition_idx(self):
    return self._partition_idx

  @property
  def stitch_idx(self):
    return self._stitch_idx

  @property
  def send_idx_cluster(self):
    return self._send_idx_cluster

  @property
  def receive_idx_cluster(self):
    return self._receive_idx_cluster

  @property
  def send_idx_cut(self):
    return self._send_idx_cut

  @property
  def receive_idx_cut(self):
    return self._receive_idx_cut
