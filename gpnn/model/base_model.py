import tensorflow as tf


class BaseModel(object):
  """ abstract base class of a generic model """

  def __init__(self, param):
    for key in param:
      setattr(self, "_" + key, param[key])

    self._dtype = tf.float64 if self._dtype == "float64" else tf.float32
    self._ops = {}
    self._var_list = None

  def _prepare(self):
    # should inherit this function and create variables, placeholders, etc.
    raise NotImplementedError

  def _inference(self):
    # should inherit this function and construct inference part
    raise NotImplementedError

  def _loss(self):
    # should inherit this function and construct loss function
    raise NotImplementedError

  def _optimizer(self):
    # either use this function or inherit it and construct self._optimizer
    global_step = tf.Variable(0.0, trainable=False)
    learn_rate = tf.train.exponential_decay(
        self._base_learn_rate,
        global_step,
        self._learn_rate_decay_step,
        self._learn_rate_decay_rate,
        staircase=True)
    self._ops["learning_rate"] = learn_rate

    if self._optimizer_name == "SGD":
      optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    elif self._optimizer_name == "Momentum":
      optimizer = tf.train.MomentumOptimizer(
          learn_rate, momentum=self._momentum)
    elif self._optimizer_name == "Adam":
      optimizer = tf.train.AdamOptimizer(learn_rate, epsilon=1e-7)
    else:
      raise ValueError("Unsupported Optimizer!")

    if self._is_clip_grad:
      # clip-gradient
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(
          tf.gradients(self._total_loss, tvars), self._clip_val)
      self._ops["train_step"] = optimizer.apply_gradients(zip(grads, tvars))
    else:
      self._ops["train_step"] = optimizer.minimize(
          self._total_loss, global_step=global_step)

  def build(self, tf_graph):
    with tf.device(self._device_id):
      with tf_graph.as_default():
        # set random seed of the graph
        tf.set_random_seed(self._seed)

        # create variables etc.
        self._prepare()

        # build inference part of model
        self._inference()

        # build loss function
        self._loss()

        # build optimizer
        self._optimizer()

        # create saver
        if self._var_list == None:
          self._saver = tf.train.Saver(max_to_keep=0)
        else:
          self._saver = tf.train.Saver(var_list=self._var_list, max_to_keep=0)

        # create summary op
        self._ops["summary"] = tf.summary.merge_all()

        # creat init op
        self._ops["init"] = tf.global_variables_initializer()

  @property
  def ops(self):
    return self._ops

  @property
  def saver(self):
    return self._saver
