import numpy as np
from gpnn.utils.logger import get_logger

logger = get_logger()


def compute_outgoing_degree(graph):
  """ compute outgoing degree of nodes """
  return np.array([len(graph[key]) for key in graph])


def get_seeds_semi_supervised_rand(successor, K, node_label=None, seed=1234):
  """
  Get K seeds of the flood fill

  Args:
    successor: dict, graph, i.e., children list
    K: number of seeds
    node_label: list, binary mask (whether the node has supervised label in the task)

  Returns:
    seeds: list [size K], index of nodes
  """
  # get out-going degree and rank (descending)
  degree = compute_outgoing_degree(successor)
  npr = np.random.RandomState(seed)

  if node_label is None:
    prob = np.float(degree) / np.sum(degree, dtype=np.float32)
    return npr.choice(len(degree), size=K, replace=False, p=prob).tolist()
  else:
    idx_labeled_node = np.nonzero(node_label)[0]
    assert (len(idx_labeled_node) >= K)
    prob = degree[idx_labeled_node].astype(np.float32) / np.sum(
        degree[idx_labeled_node], dtype=np.float32)

    return npr.choice(idx_labeled_node, size=K, replace=False, p=prob).tolist()


def multi_seed_flood_fill(successor,
                          K,
                          node_label=None,
                          is_multigraph=False,
                          rnd_seed=1234):
  """
  Implement a multi-seed flood fill algorithm

  Args:
    successor: dict, graph, i.e., children list
    K: int, number of seeds
    node_label: list, binary mask (whether the node has supervised label in the task)

  Returns:
    cluster_label: list, label of cluster
  """
  nodes = sorted(successor.keys())
  num_nodes = len(nodes)

  assert (num_nodes >= K)
  node_visited = dict((nn, False) for nn in nodes)
  npr = np.random.RandomState(rnd_seed)

  # get seeds
  seeds = get_seeds_semi_supervised_rand(
      successor, K, node_label=node_label, seed=rnd_seed)

  queue_list = [[seeds[kk]] for kk in xrange(K)]  # queue for each seed
  cluster_size = [1] * K
  cluster_label = {seeds[kk]: kk for kk in xrange(K)}

  for nn in seeds:
    node_visited[nn] = True

  # flood fill in a round robin fashion
  if is_multigraph:
    while any(queue_list):
      # At every time step, each cluster grows once
      rnd_idx_cluster = npr.permutation(K).tolist()
      for kk in rnd_idx_cluster:
        # check whether the k-cluster can still growing
        if queue_list[kk]:
          nn = queue_list[kk].pop(0)

          for child in successor[nn]:
            if not node_visited[child[0]]:
              cluster_size[kk] += 1
              cluster_label[child[0]] = kk
              node_visited[child[0]] = True
              queue_list[kk].append(child[0])
  else:
    while any(queue_list):
      # At every time step, each cluster grows once
      rnd_idx_cluster = npr.permutation(K).tolist()
      for kk in rnd_idx_cluster:
        # check whether the k-cluster can still growing
        if queue_list[kk]:
          nn = queue_list[kk].pop(0)

          for child in successor[nn]:
            if not node_visited[child]:
              cluster_size[kk] += 1
              cluster_label[child] = kk
              node_visited[child] = True
              queue_list[kk].append(child)

  # when there are dis-connected components left,
  # put them into the smallest cluster
  if len(cluster_label) < num_nodes:
    idx_min = np.argmin(cluster_size)
    return [
        cluster_label[nn] if nn in cluster_label else idx_min for nn in nodes
    ]
  else:
    return [cluster_label[nn] for nn in nodes]


def add_reverse_edges(successor):

  for nn in successor:
    for vv in successor[nn]:
      successor[vv] += [nn]

  for nn in successor:
    successor[nn] = list(set(successor[nn]))
