from __future__ import (absolute_import, division, print_function)

import tensorflow as tf
from gpnn.utils.logger import get_logger

logger = get_logger()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_para=None,
                    wd=None,
                    seed=1234,
                    name=None,
                    trainable=True,
                    validate_shape=True):
  """ Initialize Weights 

    Input:
        shape: list of int, shape of the weights
        init_method: string, indicates initialization method
        init_para: a dictionary, 
        init_val: if it is not None, it should be a tensor
        wd: a float, weight decay
        name:
        trainable:

    Output:
        var: a TensorFlow Variable
  """

  if init_method is None:
    initializer = tf.zeros_initializer(shape, dtype=dtype)
  elif init_method == "normal":
    initializer = tf.random_normal_initializer(
        mean=init_para["mean"],
        stddev=init_para["stddev"],
        seed=seed,
        dtype=dtype)
  elif init_method == "truncated_normal":
    initializer = tf.truncated_normal_initializer(
        mean=init_para["mean"],
        stddev=init_para["stddev"],
        seed=seed,
        dtype=dtype)
  elif init_method == "uniform":
    initializer = tf.random_uniform_initializer(
        minval=init_para["minval"],
        maxval=init_para["maxval"],
        seed=seed,
        dtype=dtype)
  elif init_method == "constant":
    initializer = tf.constant_initializer(value=init_para["val"], dtype=dtype)
  elif init_method == "xavier":
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=True, seed=seed, dtype=dtype)
  else:
    raise ValueError("Unsupported initialization method!")

  # var = tf.Variable(initializer(shape), name=name, trainable=trainable)

  var = tf.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      validate_shape=validate_shape,
      trainable=trainable)

  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_decay")
    tf.add_to_collection("losses", weight_decay)

  return var


class GRU(object):
  """ Gated Recurrent Units (GRU)

    Input:
        input_dim: input dimension
        hidden_dim: hidden dimension
        wd: a float, weight decay         
        scope: tf scope of the model

    Output:
        a function which computes the output of GRU with one step 
  """

  def __init__(self,
               input_dim,
               hidden_dim,
               wd=None,
               dtype=tf.float32,
               init_std=None,
               trainable=True,
               seed=1234,
               scope="GRU"):
    if init_std:
      self._init_method = "truncated_normal"
    else:
      self._init_method = "xavier"

    logger.info("GRU cell: {}".format(scope))
    logger.info("Input dim: {}".format(input_dim))
    logger.info("Hidden dim: {}".format(hidden_dim))
    logger.info("Var init method: {}".format(self._init_method))

    # initialize variables
    with tf.variable_scope(scope):
      self._w_xi = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_xi",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._w_hi = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_hi",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._b_i = weight_variable(
          [hidden_dim],
          init_method="constant",
          init_para={"val": 1.0},
          wd=wd,
          name="b_i",
          trainable=trainable,
          dtype=dtype,
          seed=seed)

      self._w_xr = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_xr",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._w_hr = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_hr",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._b_r = weight_variable(
          [hidden_dim],
          init_method="constant",
          init_para={"val": 0.0},
          wd=wd,
          name="b_r",
          trainable=trainable,
          dtype=dtype,
          seed=seed)

      self._w_xu = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_xu",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._w_hu = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="w_hu",
          trainable=trainable,
          dtype=dtype,
          seed=seed)
      self._b_u = weight_variable(
          [hidden_dim],
          init_method="constant",
          init_para={"val": 0.0},
          wd=wd,
          name="b_u",
          trainable=trainable,
          dtype=dtype,
          seed=seed)

  def __call__(self, x, state):
    # update gate
    g_i = tf.sigmoid(
        tf.matmul(x, self._w_xi) + tf.matmul(state, self._w_hi) + self._b_i)

    # reset gate
    g_r = tf.sigmoid(
        tf.matmul(x, self._w_xr) + tf.matmul(state, self._w_hr) + self._b_r)

    # new memory
    # implementation 1
    u = tf.tanh(
        tf.matmul(x, self._w_xu) + tf.matmul(g_r * state, self._w_hu) +
        self._b_u)

    # implementation 2
    # u = tf.tanh(
    #     tf.matmul(x, self._w_xu) + g_r * tf.matmul(state, self._w_hu) +
    #     self._b_u)

    # hidden state
    new_state = state * g_i + u * (1 - g_i)

    return new_state


class MLP(object):
  """ Multi Layer Perceptron (MLP)
        Note: the number of layers is N

    Input:
        dims: a list of N+1 int, number of hidden units (last one is the input dimension)
        act_func: a list of N activation functions
        add_bias: a boolean, indicates whether adding bias or not
        wd: a float, weight decay 
        scope: tf scope of the model

    Output:
        a function which outputs a list of N tensors, each is the hidden activation of one layer 
  """

  def __init__(self,
               dims,
               act_func=None,
               add_bias=True,
               wd=None,
               dtype=tf.float32,
               init_std=None,
               trainable=True,
               sparse_input=False,
               seed=1234,
               scope="MLP"):

    if init_std:
      self._init_method = "truncated_normal"
    else:
      self._init_method = "xavier"

    self._scope = scope
    self._add_bias = add_bias
    self._num_layer = len(dims) - 1
    self._w = [None] * self._num_layer
    self._b = [None] * self._num_layer
    self._act_func = [None] * self._num_layer
    self._sparse_input = sparse_input
    self._seed = seed

    logger.info("MLP: {}".format(scope))
    logger.info("Input dim: {}".format(dims[-1]))
    logger.info("Hidden dim: {}".format(dims[:-1]))
    logger.info("Activation: {}".format(act_func))
    logger.info("Var init method: {}".format(self._init_method))

    # initialize variables
    with tf.variable_scope(scope):
      for ii in xrange(self._num_layer):
        with tf.variable_scope("layer_{}".format(ii)):
          dim_in = dims[ii - 1]
          dim_out = dims[ii]

          self._w[ii] = weight_variable(
              [dim_in, dim_out],
              init_method=self._init_method,
              init_para={"mean": 0.0,
                         "stddev": init_std},
              wd=wd,
              name="w",
              trainable=trainable,
              dtype=dtype,
              seed=seed)

          logger.info("MLP weight: {}".format([dim_in, dim_out]))

          if add_bias:
            self._b[ii] = weight_variable(
                [dim_out],
                init_method="constant",
                init_para={"val": 1.0e-2},
                wd=wd,
                name="b",
                trainable=trainable,
                dtype=dtype,
                seed=seed)
            logger.info("MLP bias: {}".format(dim_out))

          if act_func and act_func[ii] is not None:
            if act_func[ii] == "relu":
              self._act_func[ii] = tf.nn.relu
            elif act_func[ii] == "sigmoid":
              self._act_func[ii] = tf.sigmoid
            elif act_func[ii] == "tanh":
              self._act_func[ii] = tf.tanh
            else:
              raise ValueError("Unsupported activation method!")

  def __call__(self, x, dropout_rate=None):
    h = [None] * self._num_layer

    with tf.variable_scope(self._scope):
      for ii in xrange(self._num_layer):
        with tf.variable_scope("layer_{}".format(ii)):
          if ii == 0:
            input_vec = x
          else:
            input_vec = h[ii - 1]

          if self._sparse_input and ii == 0:
            h[ii] = tf.sparse_tensor_dense_matmul(input_vec, self._w[ii])
          else:
            h[ii] = tf.matmul(input_vec, self._w[ii])

          if dropout_rate is not None and ii <= self._num_layer - 1:
            h[ii] = tf.nn.dropout(
                h[ii], keep_prob=1.0 - dropout_rate, seed=self._seed)

          if self._add_bias:
            h[ii] += self._b[ii]

          if self._act_func[ii] is not None:
            h[ii] = self._act_func[ii](h[ii])

    return h


class LSTM(object):
  """ Long Short-term Memory (LSTM)

    Input:
        input_dim: input dimension
        hidden_dim: hidden dimension
        wd: a float, weight decay         
        scope: tf scope of the model

    Output:
        a function which computes the output of GRU with one step 
  """

  def __init__(self,
               input_dim,
               hidden_dim,
               wd=None,
               dtype=tf.float32,
               init_std=None,
               seed=1234,
               scope="LSTM"):

    if init_std:
      self._init_method = "truncated_normal"
    else:
      self._init_method = "xavier"

    logger.info("LSTM cell: {}".format(scope))
    logger.info("Input dim: {}".format(input_dim))
    logger.info("Hidden dim: {}".format(hidden_dim))
    logger.info("Var init method: {}".format(self._init_method))

    # initialize variables
    with tf.variable_scope(scope):
      # forget gate
      self._Wf = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Wf",
          dtype=dtype,
          seed=seed)

      self._Uf = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Uf",
          dtype=dtype,
          seed=seed)

      self._bf = weight_variable(
          [hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="bf",
          dtype=dtype,
          seed=seed)

      # input gate
      self._Wi = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Wi",
          dtype=dtype,
          seed=seed)

      self._Ui = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Ui",
          dtype=dtype,
          seed=seed)

      self._bi = weight_variable(
          [hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="bi",
          dtype=dtype,
          seed=seed)

      # output gate
      self._Wo = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Wo",
          dtype=dtype,
          seed=seed)

      self._Uo = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Uo",
          dtype=dtype,
          seed=seed)

      self._bo = weight_variable(
          [hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="bo",
          dtype=dtype,
          seed=seed)

      # output gate
      self._Wo = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Wo",
          dtype=dtype,
          seed=seed)

      self._Uo = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Uo",
          dtype=dtype,
          seed=seed)

      self._bo = weight_variable(
          [hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="bo",
          dtype=dtype,
          seed=seed)

      # cell
      self._Wc = weight_variable(
          [input_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Wc",
          dtype=dtype,
          seed=seed)

      self._Uc = weight_variable(
          [hidden_dim, hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="Uc",
          dtype=dtype,
          seed=seed)

      self._bc = weight_variable(
          [hidden_dim],
          init_method=self._init_method,
          init_para={"mean": 0.0,
                     "stddev": init_std},
          wd=wd,
          name="bc",
          dtype=dtype,
          seed=seed)

  def __call__(self, x, state, memory):
    # forget gate
    f = tf.sigmoid(
        tf.matmul(x, self._Wf) + tf.matmul(state, self._Uf) + self._bf)

    # input gate
    i = tf.sigmoid(
        tf.matmul(x, self._Wi) + tf.matmul(state, self._Ui) + self._bi)

    # output gate
    o = tf.sigmoid(
        tf.matmul(x, self._Wo) + tf.matmul(state, self._Uo) + self._bo)

    # new memory
    new_memory = f * memory + i * tf.tanh(
        tf.matmul(x, self._Wc) + tf.matmul(state, self._Uc) + self._bc)

    # hidden state
    new_state = o * tf.tanh(new_memory)

    return new_state, new_memory
