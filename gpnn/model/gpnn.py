from __future__ import (absolute_import, division, print_function)

import numpy as np
import tensorflow as tf

from gpnn.factory import ModelFactory
from gpnn.model.base_model import BaseModel
from gpnn.model.model_helper import aggregate
from gpnn.model.nn_cells import MLP, GRU, weight_variable
from gpnn.utils.logger import get_logger

logger = get_logger()
register = ModelFactory.register


@register("GPNN")
class GPNN(BaseModel):
  """ Implementation of Graph Partition Neural Networks """

  def __init__(self, param):
    super(GPNN, self).__init__(param)
    self._seed = param["seed"]
    self._dataset_name = param["dataset_name"]
    self._num_pass = param["num_pass"]
    self._num_cluster = param["num_cluster"]
    self._prop_step_intra = param["prop_step_intra"]
    self._prop_step_inter = param["prop_step_inter"]
    self._num_nodes = param["num_nodes"]
    self._update_MLP_dim = param["update_MLP_dim"]
    self._update_MLP_act = param["update_MLP_act"]
    self._output_MLP_dim = param["output_MLP_dim"]
    self._output_MLP_act = param["output_MLP_act"]
    self._dropout = param["dropout_prob"]
    self._update_type = param["update_type"]
    self._param = param
    assert (self._dropout >= 0.0 and self._dropout <= 1.0)
    assert (len(self._output_MLP_dim) == len(self._output_MLP_act))

  def _create_placehoders(self):
    self._num_node_cut_ph = tf.placeholder(tf.int32, shape=[])
    self._cluster_size_ph = tf.placeholder(tf.int32, shape=[self._num_cluster])
    self._node_feat_idx_ph = tf.placeholder(tf.int64, shape=[None, 2])
    self._node_feat_val_ph = tf.placeholder(self._dtype, shape=[None])
    self._node_feat_shape_ph = tf.placeholder(tf.int64, shape=[2])
    self._gt_label_ph = tf.placeholder(tf.int32, shape=[self._num_nodes])
    self._node_mask_ph = tf.placeholder(tf.int32, shape=[self._num_nodes])
    self._train_idx_ph = tf.placeholder(tf.int32, shape=[None])
    self._val_idx_ph = tf.placeholder(tf.int32, shape=[None])
    self._test_idx_ph = tf.placeholder(tf.int32, shape=[None])
    self._dropout_rate = tf.placeholder_with_default(0.0, shape=[])

    # index which partition node (cluster order) into cut (0) & non-cut (1)
    self._partition_idx_ph = tf.placeholder(tf.int32, shape=[self._num_nodes])

    # index which stitch cut & non-cut into node (cluster order)
    self._stitch_idx_ph = [
        tf.placeholder(tf.int32, shape=[self._num_node_cut]),
        tf.placeholder(tf.int32, shape=[self._num_node_noncut])
    ]

    # incoming node of edege
    # local index within cluster
    self._send_idx_cluster_ph = [[
        tf.placeholder(tf.int32, shape=(None))
        for _ in xrange(self._num_edgetype)
    ] for _ in xrange(self._num_cluster)]

    # outgoing node of edge
    # local index within cluster
    self._receive_idx_cluster_ph = [
        tf.placeholder(tf.int32, shape=(None))
        for _ in xrange(self._num_cluster)
    ]

    # incoming node of edege
    # local index within cut
    self._send_idx_cut_ph = [
        tf.placeholder(tf.int32, shape=(None))
        for _ in xrange(self._num_edgetype)
    ]

    # outgoing node of edge
    # local index within cut
    self._receive_idx_cut_ph = tf.placeholder(tf.int32, shape=(None))

  def _create_variables(self):
    """ Create variables """
    self._train_idx = tf.get_variable(
        "train_idx",
        shape=[1],
        dtype=tf.int32,
        validate_shape=False,
        trainable=False)

    self._val_idx = tf.get_variable(
        "val_idx",
        shape=[1],
        dtype=tf.int32,
        validate_shape=False,
        trainable=False)

    self._test_idx = tf.get_variable(
        "test_idx",
        shape=[1],
        dtype=tf.int32,
        validate_shape=False,
        trainable=False)

    self._node_feat_idx = tf.get_variable(
        "node_feat_idx",
        shape=[1, 2],
        dtype=tf.int64,
        validate_shape=False,
        trainable=False)

    self._node_feat_val = tf.get_variable(
        "node_feat_val",
        shape=[1],
        dtype=self._dtype,
        validate_shape=False,
        trainable=False)

    self._node_feat_shape = tf.get_variable(
        "node_feat_shape",
        shape=[2],
        dtype=tf.int64,
        validate_shape=False,
        trainable=False)

    self._num_node_cut_var = tf.get_variable(
        "num_node_cut", shape=[], dtype=tf.int32, trainable=False)

    self._cluster_size_var = tf.get_variable(
        "cluster_size",
        shape=[self._num_cluster],
        dtype=tf.int32,
        validate_shape=True,
        trainable=False)

    self._partition_idx = tf.get_variable(
        "partition_idx",
        shape=[self._num_nodes],
        dtype=tf.int32,
        trainable=False)

    self._stitch_idx = [
        tf.get_variable(
            "stitch_idx_0",
            shape=[self._num_node_cut],
            dtype=tf.int32,
            validate_shape=False,
            trainable=False),
        tf.get_variable(
            "stitch_idx_1",
            shape=[self._num_node_noncut],
            dtype=tf.int32,
            validate_shape=False,
            trainable=False)
    ]

    self._send_idx_cluster = [[
        tf.get_variable(
            "send_idx_cluster_{}_{}".format(ii, jj),
            shape=[1],
            dtype=tf.int32,
            validate_shape=False,
            trainable=False) for jj in xrange(self._num_edgetype)
    ] for ii in xrange(self._num_cluster)]

    self._receive_idx_cluster = [
        tf.get_variable(
            "receive_idx_cluster_{}".format(ii),
            shape=[1],
            dtype=tf.int32,
            validate_shape=False,
            trainable=False) for ii in xrange(self._num_cluster)
    ]

    self._send_idx_cut = [
        tf.get_variable(
            "send_idx_cut_{}".format(ii),
            shape=[1],
            dtype=tf.int32,
            validate_shape=False,
            trainable=False) for ii in xrange(self._num_edgetype)
    ]

    self._receive_idx_cut = tf.get_variable(
        "receive_idx_cut",
        shape=[1],
        dtype=tf.int32,
        validate_shape=False,
        trainable=False)

    if self._dataset_name == "nell" or self._dataset_name == "diel":
      self._node_embedding = tf.get_variable(
          "node_embedding",
          shape=[self._num_nodes, self._hidden_dim],
          initializer=tf.random_uniform_initializer(
              minval=-1.0, maxval=1.0, seed=self._seed, dtype=self._dtype),
          dtype=self._dtype,
          trainable=True)

    self._node_mask = tf.get_variable(
        "node_mask", shape=[self._num_nodes], dtype=tf.int32, trainable=False)
    self._gt_label = tf.get_variable(
        "gt_label", shape=[self._num_nodes], dtype=tf.int32, trainable=False)
    self._ops["gt_label"] = self._gt_label

  def _get_assign_ops(self):
    """ load data """
    self._assign_ops = []
    self._assign_ops += [
        tf.assign(self._num_node_cut_var, self._num_node_cut_ph)
    ]
    self._assign_ops += [
        tf.assign(self._cluster_size_var, self._cluster_size_ph)
    ]

    self._assign_ops += [
        tf.assign(self._train_idx, self._train_idx_ph, validate_shape=False)
    ]
    self._assign_ops += [
        tf.assign(self._val_idx, self._val_idx_ph, validate_shape=False)
    ]
    self._assign_ops += [
        tf.assign(self._test_idx, self._test_idx_ph, validate_shape=False)
    ]

    self._assign_ops += [
        tf.assign(
            self._node_feat_idx, self._node_feat_idx_ph, validate_shape=False)
    ]
    self._assign_ops += [
        tf.assign(
            self._node_feat_val, self._node_feat_val_ph, validate_shape=False)
    ]
    self._assign_ops += [
        tf.assign(self._node_feat_shape, self._node_feat_shape_ph)
    ]

    self._assign_ops += [tf.assign(self._node_mask, self._node_mask_ph)]
    self._assign_ops += [tf.assign(self._gt_label, self._gt_label_ph)]

    self._assign_ops += [tf.assign(self._partition_idx, self._partition_idx_ph)]
    self._assign_ops += [tf.assign(self._stitch_idx[0], self._stitch_idx_ph[0])]
    self._assign_ops += [tf.assign(self._stitch_idx[1], self._stitch_idx_ph[1])]

    for ii in xrange(self._num_cluster):
      for jj in xrange(self._num_edgetype):
        self._assign_ops += [
            tf.assign(
                self._send_idx_cluster[ii][jj],
                self._send_idx_cluster_ph[ii][jj],
                validate_shape=False)
        ]

    for ii in xrange(self._num_cluster):
      self._assign_ops += [
          tf.assign(
              self._receive_idx_cluster[ii],
              self._receive_idx_cluster_ph[ii],
              validate_shape=False)
      ]

    for ii in xrange(self._num_edgetype):
      self._assign_ops += [
          tf.assign(
              self._send_idx_cut[ii],
              self._send_idx_cut_ph[ii],
              validate_shape=False)
      ]

    self._assign_ops += [
        tf.assign(
            self._receive_idx_cut,
            self._receive_idx_cut_ph,
            validate_shape=False)
    ]
    self._assign_ops = tf.group(*self._assign_ops)

  def _prepare(self):
    self._num_node_cut = self.param["num_node_cut"]
    self._cluster_size = self.param["cluster_size"]
    self._num_node_noncut = self._num_nodes - self._num_node_cut

    self._create_placehoders()
    self._create_variables()
    self._get_assign_ops()

    self._node_feat = tf.SparseTensor(
        indices=self._node_feat_idx,
        values=self._node_feat_val,
        dense_shape=self._node_feat_shape)

    # node states
    self._node_vec = [None] * (self._num_pass + 1)

    # build propagation MLP for each edge type
    if self._msg_type == "msg_embedding":
      pass
    elif self._msg_type == "msg_mlp":
      self._MLP_prop = [
          MLP([self._hidden_dim, self._hidden_dim],
              act_func=[None],
              add_bias=True,
              dtype=self._dtype,
              wd=self._weight_decay,
              scope="MLP_prop_{}".format(ii))
          for ii in xrange(self._num_edgetype)
      ]
    else:
      raise ValueError("Unsupported message type!")

    # build feat embedding MLP
    self._MLP_feat = MLP(
        [self._hidden_dim, self._feat_dim],
        act_func=[None],
        add_bias=True,
        dtype=self._dtype,
        wd=self._weight_decay,
        trainable=True,
        sparse_input=True,
        scope="MLP_feat")

    # build a update function shared by all nodes
    if self._update_type == "GRU":
      self._update_func = GRU(
          self._hidden_dim,
          self._hidden_dim,
          dtype=self._dtype,
          wd=self._weight_decay,
          trainable=True,
          scope="GRU_update")
    elif self._update_type == "MLP":
      self._update_func = MLP(
          self._update_MLP_dim + [self._hidden_dim, 2 * self._hidden_dim],
          act_func=self._update_MLP_act + ["tanh"],
          add_bias=True,
          dtype=self._dtype,
          wd=self._weight_decay,
          trainable=True,
          scope="MLP_update")
    else:
      raise ValueError("Unsupported message type!")

    # build output MLP
    if self._dataset_name == "nell" or self._dataset_name == "diel":
      self._MLP_out = MLP(
          self._output_MLP_dim + [self._label_size, 2 * self._hidden_dim],
          act_func=self._output_MLP_act + [None],
          add_bias=True,
          dtype=self._dtype,
          wd=self._weight_decay,
          scope="MLP_out")
    else:
      self._MLP_out = MLP(
          self._output_MLP_dim +
          [self._label_size, self._hidden_dim + self._feat_dim],
          act_func=self._output_MLP_act + [None],
          add_bias=True,
          dtype=self._dtype,
          wd=self._weight_decay,
          scope="MLP_out")

  def _construct_feeddict(self, data, is_train=True):
    feed_dict = {}
    feed_dict[self._train_idx_ph] = data.train_idx
    feed_dict[self._val_idx_ph] = data.val_idx
    feed_dict[self._test_idx_ph] = data.test_idx

    feed_dict[self._node_feat_idx_ph] = data.node_feat_indices
    feed_dict[self._node_feat_val_ph] = data.node_feat_values
    feed_dict[self._node_feat_shape_ph] = data.node_feat_shape

    feed_dict[self._gt_label_ph] = data.node_gt_label
    feed_dict[self._num_node_cut_ph] = self._num_node_cut
    feed_dict[self._cluster_size_ph] = self._cluster_size

    if is_train:
      node_mask = np.zeros_like(data.node_gt_label, dtype=np.int32)
      node_mask[data.train_idx] = 1
      feed_dict[self._node_mask_ph] = node_mask
      feed_dict[self._dropout_rate] = self._dropout
    else:
      feed_dict[self._node_mask_ph] = np.ones_like(
          data.node_gt_label, dtype=np.int32)

    feed_dict[self._partition_idx_ph] = data.partition_idx
    feed_dict[self._stitch_idx_ph[0]] = data.stitch_idx[0]
    feed_dict[self._stitch_idx_ph[1]] = data.stitch_idx[1]

    for ii in xrange(self._num_cluster):
      for jj in xrange(self._num_edgetype):
        feed_dict[self._send_idx_cluster_ph[ii][jj]] = data.send_idx_cluster[
            ii][jj]

    for ii in xrange(self._num_cluster):
      feed_dict[self._receive_idx_cluster_ph[ii]] = data.receive_idx_cluster[ii]

    for ii in xrange(self._num_edgetype):
      feed_dict[self._send_idx_cut_ph[ii]] = data.send_idx_cut[ii]

    feed_dict[self._receive_idx_cut_ph] = data.receive_idx_cut

    return feed_dict

  def _inference(self):
    with tf.variable_scope("inference"):
      input_feat = self._MLP_feat(self._node_feat)[-1]

      if self._dataset_name == "nell" or self._dataset_name == "diel":
        self._node_feat = input_feat
        self._node_vec[-1] = self._node_embedding
      else:
        self._node_feat = tf.sparse_tensor_to_dense(
            self._node_feat, validate_indices=False)
        self._node_vec[-1] = input_feat

      for pp in xrange(self._num_pass):
        with tf.variable_scope("pass_{}".format(pp)):
          ### parallel synchoronous propagation within clusters
          node_vec_cluster = [[None] * (self._prop_step_intra + 1)
                              for _ in xrange(self._num_cluster)]
          node_vec_cluster_init = tf.split(
              self._node_vec[pp - 1], self._cluster_size_var, axis=0)

          for ii in xrange(self._num_cluster):
            with tf.variable_scope("cluster_{}".format(ii)):
              # node representation
              node_vec_cluster[ii][-1] = node_vec_cluster_init[ii]

              for tt in xrange(self._prop_step_intra):
                # pull messages
                node_vec_sum = [None] * self._num_edgetype

                for ee in xrange(self._num_edgetype):
                  node_active = tf.gather(node_vec_cluster[ii][tt - 1],
                                          self._send_idx_cluster[ii][ee])

                  if self._msg_type == "msg_embedding":
                    # compute msg using embedding alone
                    node_vec_sum[ee] = node_active
                  elif self._msg_type == "msg_mlp":
                    # compute msg using a MLP
                    node_vec_sum[ee] = self._MLP_prop[ee](node_active)[-1]

                # aggregate messages
                concat_msg = tf.concat(node_vec_sum, axis=0)
                message = aggregate(
                    concat_msg,
                    self._receive_idx_cluster[ii],
                    self._cluster_size_var[ii] + 1,
                    method=self._aggregate)

                # update hidden states
                if self._update_type == "MLP":
                  node_vec_cluster[ii][tt] = self._update_func(
                      tf.concat(
                          [message[:-1, :], node_vec_cluster[ii][tt - 1]],
                          axis=1))[-1]
                else:
                  node_vec_cluster[ii][tt] = self._update_func(
                      message[:-1, :], node_vec_cluster[ii][tt - 1])

          ### update node representation
          node_vec = tf.concat([xx[-2] for xx in node_vec_cluster], axis=0)

          is_cut_empty = tf.equal(
              tf.reduce_sum(self._partition_idx), self._num_nodes)

          ### synchoronous propagation within cut
          def prop_in_cut():
            with tf.variable_scope("cut"):
              node_vec_part = tf.dynamic_partition(node_vec,
                                                   self._partition_idx, 2)
              node_vec_cut = [None] * (self._prop_step_inter + 1)
              node_vec_cut[-1] = node_vec_part[0]

              for tt in xrange(self._prop_step_inter):
                # pull messages
                node_vec_sum = [None] * self._num_edgetype

                for ee in xrange(self._num_edgetype):
                  # partition
                  node_active = tf.gather(node_vec_cut[tt - 1],
                                          self._send_idx_cut[ee])

                  if self._msg_type == "msg_embedding":
                    # compute msg using embedding alone
                    node_vec_sum[ee] = node_active
                  elif self._msg_type == "msg_mlp":
                    # compute msg using a MLP
                    node_vec_sum[ee] = self._MLP_prop[ee](node_active)[-1]

                # aggregate messages
                concat_msg = tf.concat(node_vec_sum, axis=0)
                message = aggregate(
                    concat_msg,
                    self._receive_idx_cut,
                    self._num_node_cut_var + 1,
                    method=self._aggregate)

                # update hidden states via GRU
                if self._update_type == "MLP":
                  node_vec_cut[tt] = self._update_func(
                      tf.concat(
                          [message[:-1, :], node_vec_cut[tt - 1]], axis=1))[-1]
                else:
                  node_vec_cut[tt] = self._update_func(message[:-1, :],
                                                       node_vec_cut[tt - 1])

              return tf.dynamic_stitch(self._stitch_idx,
                                       [node_vec_cut[-2], node_vec_part[1]])

          def no_prop():
            return node_vec

          # Update final representation
          self._node_vec[pp] = tf.cond(is_cut_empty, no_prop, prop_in_cut)

          logger.info("Propagation pass = {}".format(pp))

      output_feat = tf.concat([self._node_vec[-2], self._node_feat], axis=1)
      self._logits = self._MLP_out(output_feat, self._dropout_rate)[-1]

      self._ops["pred_logits"] = tf.nn.softmax(self._logits)
      self._ops["val_pred_logits"] = tf.gather(self._ops["pred_logits"],
                                               self._val_idx)
      self._ops["test_pred_logits"] = tf.gather(self._ops["pred_logits"],
                                                self._test_idx)

  def _loss(self):
    logger.info("Use cross entropy loss")
    node_mask_float = tf.cast(self._node_mask, tf.float32)

    CE_loss = tf.reduce_sum(
        node_mask_float * tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._gt_label)) / tf.reduce_sum(
                node_mask_float)

    tf.summary.scalar("CE_loss", CE_loss)
    tf.add_to_collection("losses", CE_loss)
    self._ops["CE_loss"] = CE_loss
    self._total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    self._ops["weight_decay"] = self._total_loss - CE_loss

  @property
  def param(self):
    return self._param

  @property
  def assign_ops(self):
    return self._assign_ops
