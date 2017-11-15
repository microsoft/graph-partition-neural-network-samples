import scipy
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, coo_matrix, spdiags, issparse

EPS = np.finfo(np.float32).eps


def check_symmetric(m, tol=1e-8):

  if issparse(m):
    if m.shape[0] != m.shape[1]:
      raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
      m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
      return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    return np.allclose(vl, vu, atol=tol)
  else:
    return np.allclose(m, m.T, atol=tol)


def construct_adj_mat(successor, node_map, is_multigraph=False):
  num_node = len(successor.keys())
  adj_mat = np.zeros([num_node, num_node])

  for ii in successor:
    uu = node_map[ii]
    for jj in successor[ii]:
      if is_multigraph:
        vv = node_map[jj[0]]
      else:
        vv = node_map[jj]

      adj_mat[uu, vv] = 1

  return adj_mat


def compute_laplacian(W):
  D = np.sum(W, axis=0)

  # unnormalized graph laplacian
  # L = np.diag(D) - W

  # normalized graph laplacian
  idx = D != 0

  diag_val = np.zeros_like(D)
  diag_val[idx] = 1.0

  D_inv = np.zeros_like(D)
  D_inv[idx] = 1.0 / D[idx]

  L = np.diag(diag_val) - D_inv.reshape([-1, 1]) * W

  return L


def construct_adj_mat_sparse(successor, node_map, is_multigraph=False):
  num_node = len(successor.keys())
  row = []
  col = []
  data = []

  for ii in successor:
    uu = node_map[ii]
    for jj in successor[ii]:
      if is_multigraph:
        vv = node_map[jj[0]]
      else:
        vv = node_map[jj]

      row += [uu]
      col += [vv]
      data += [1]

  adj_mat = csr_matrix(
      (np.array(data), (np.array(row), np.array(col))),
      shape=[num_node, num_node],
      dtype=np.float32)

  return adj_mat


def compute_laplacian_sparse(W):

  num_node = W.shape[0]
  D = np.sum(W, axis=0).A1

  # unnormalized graph laplacian
  # L = np.diag(D) - W

  # normalized graph laplacian
  idx = D != 0
  diag_vec = np.zeros_like(D)
  diag_vec[idx] = 1.0

  row = np.nonzero(D)[0]
  col = row
  diag_val = csr_matrix(
      (diag_vec, (row, col)), shape=[num_node, num_node], dtype=np.float32)

  D_inv = np.zeros_like(D)
  D_inv[idx] = 1.0 / D[idx]

  L = diag_val - spdiags(D_inv, 0, num_node, num_node) * W

  return L


def spectral_clustering(successor,
                        K,
                        is_multigraph=False,
                        use_sparse=False,
                        seed=1234):
  """
  Implement paper "Shi, J. and Malik, J., 2000. Normalized cuts and image 
  segmentation. IEEE Transactions on pattern analysis and machine intelligence, 
  22(8), pp.888-905."

  Args:
    successor: dict, children list
    K: int, number of clusters

  Returns:
    node_label: list

  N.B.: for simplicity, we will ignore multigraph and assume that reverse edges
        are already added, thus dealing with undirected graph  

        Use sparse matrix when data becomes large
  """
  nodes = successor.keys()
  num_nodes = len(nodes)
  assert (
      K < num_nodes - 1)  # due to the requirement of "scipy.sparse.linalg.eigs"
  node_map = dict(zip(nodes, range(num_nodes)))

  if use_sparse:
    W = construct_adj_mat_sparse(
        successor, node_map, is_multigraph=is_multigraph)
    L = compute_laplacian_sparse(W)
  else:
    W = construct_adj_mat(successor, node_map, is_multigraph=is_multigraph)
    L = compute_laplacian(W)

  if check_symmetric(W):
    eig, eig_vec = scipy.sparse.linalg.eigsh(
        L, k=K, which="SM", maxiter=num_nodes * 10000, tol=0, mode="normal")
  else:
    eig, eig_vec = scipy.sparse.linalg.eigs(
        L, k=K, which="SM", maxiter=num_nodes * 10000, tol=0)

  kmeans = KMeans(n_clusters=K, random_state=seed).fit(eig_vec.real)

  return [kmeans.labels_[node_map[ii]] for ii in nodes]
