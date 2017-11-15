import os
import tensorflow as tf
from tensorflow.python.client import timeline

from gpnn.utils.logger import get_logger
from gpnn.factory import (ReaderFactory, ModelFactory)

logger = get_logger()


class BaseRunner(object):
  """ abstract base class of runner """

  def __init__(self, param):
    self._param = param
    self._data_reader = ReaderFactory.factory(param["reader_name"])(param)
    self._model = ModelFactory.factory(param["model_name"])(param)

    # set attribute
    attr_list = [
        "gpu_only", "save_dir", "is_resume_training", "resume_model_path",
        "test_model_path", "early_stop_window", "display_iter", "valid_iter",
        "snapshot_iter", "bat_size", "max_epoch", "is_profile"
    ]

    for key in attr_list:
      setattr(self, "_" + key, param[key])

  def _build_graph(self):
    """ build computational graph of tensorflow """
    tf_graph = tf.Graph()
    self._model.build(tf_graph)

    # initialize computational graph
    if self._is_profile:
      self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      self._run_metadata = tf.RunMetadata()

    self._tf_config = tf.ConfigProto(allow_soft_placement=(not self._gpu_only))
    self._session = tf.Session(graph=tf_graph, config=self._tf_config)
    self._session.run(self._model.ops["init"])

  def _save_model(self, save_name):
    """ Snapshot model """
    self._model.saver.save(self._session, save_name)

  def _get(self, feed_data, op_names):
    """ Get results of one mini-batch """
    ops = [self._model.ops[nn] for nn in op_names]

    # profile code
    if self._is_profile:
      op_results = self._session.run(
          ops,
          feed_dict=feed_data,
          options=self._run_options,
          run_metadata=self._run_metadata)
      trace = timeline.Timeline(self._run_metadata.step_stats)
      chrome_trace = trace.generate_chrome_trace_format()
      with open(os.path.join(self._save_dir, 'timeline.json'), 'w') as f:
        f.write(chrome_trace)
    else:
      op_results = self._session.run(ops, feed_dict=feed_data)

    results = {}
    for rr, name in zip(op_results, op_names):
      results[name] = rr

    return results

  def train(self):
    """ Train model """
    raise NotImplementedError

  def test(self):
    """ Test model """
    raise NotImplementedError
