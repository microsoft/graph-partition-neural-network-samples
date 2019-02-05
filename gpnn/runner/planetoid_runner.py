from __future__ import (absolute_import, division, print_function)

import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict

from gpnn.factory import RunnerFactory
from gpnn.runner.base_runner import BaseRunner
from gpnn.utils.logger import get_logger
from gpnn.utils.reader_helper import read_list_from_file
from gpnn.utils.runner_helper import EarlyStopper, mkdir, convert_to_np_array_or_not
from gpnn.utils.eval_helper import compute_accuracy, compute_acc_multiclass

logger = get_logger()
register = RunnerFactory.register


@register("PlanetoidRunner")
class PlanetoidRunner(BaseRunner):

  def __init__(self, param):
    super(PlanetoidRunner, self).__init__(param)

    self._run_id = param["run_id"]
    self._label_size = param["label_size"]
    self._dataset_name = param["dataset_name"]
    self._dropout = param["dropout_prob"]

    # since we only load data once, we put it in init
    self._data = self._data_reader.read()

  def _run_valid_ops(self, val_op_names, is_val=True):
    bat_results = self._get({self._model._dropout_rate: 0.0}, val_op_names)

    if is_val:
      pred_logits = bat_results["val_pred_logits"]
      gt_label = self._data.node_gt_label[self._data.val_idx]
    else:
      pred_logits = bat_results["test_pred_logits"]
      gt_label = self._data.node_gt_label[self._data.test_idx]

    results = self._eval_validation(pred_logits, gt_label)
    results["CE_loss"] = -np.mean(
        np.log(pred_logits[np.arange(len(gt_label)), gt_label]))

    if self._dataset_name == "diel" and not is_val:
      pred_logits = bat_results["test_pred_logits"]
      results["recall"] = self._eval_validation_diel(self._data, pred_logits)

    return results

  def _eval_validation_diel(self, data, pred_logits):
    tpy = pred_logits
    tpy_ind = np.argmax(tpy, axis=1)
    test_list = data.test_list
    s_graph = data.s_graph
    test_cov = data.test_cov
    sim_dict = data.sim_dict
    CATS = data.category

    st_dict = defaultdict(float)
    for i, l in enumerate(test_list):
      j = tpy_ind[i]
      cat = CATS[j]
      for item in s_graph[l]:
        cur = st_dict[(item, cat)]
        if tpy[i, j] > cur:
          st_dict[(item, cat)] = tpy[i, j]

    st_dict = sorted(st_dict.items(), key=lambda x: x[1], reverse=True)

    pred_labels = defaultdict(set)
    for k, _ in st_dict[:240000]:
      item, cat = k
      pred_labels[cat].add(item)

    pred_num = 0

    for cat in CATS:
      pred_num += len(pred_labels[cat])

    tot, cor = 0, 0
    for cat in CATS:
      for test_item in test_cov[cat]:
        tot += 1
        for item in sim_dict[test_item]:
          if item in pred_labels[cat]:
            cor += 1
            break

    recall = 1.0 * cor / tot

    return recall

  def _eval_validation(self, pred_logits, gt_label):
    return {"multi_class_acc": compute_accuracy(pred_logits, gt_label)}

  def _disp_stats(self, results, disp_tag="Validation"):
    logger.info("=" * 80)
    if disp_tag != "Testing":
      logger.info("{} CE Loss = {:e}".format(disp_tag, results["CE_loss"]))
      logger.info("{} mean accuracy = {}".format(disp_tag, results[
          "multi_class_acc"] * 100.0))
    else:
      if self._dataset_name == "diel":
        logger.info("{} recall @ 240000 = {}".format(disp_tag, results["recall"]
                                                     * 100.0))
      else:
        logger.info("{} accuracy = {}".format(disp_tag, results[
            "multi_class_acc"] * 100.0))

  def _update_best(self, results, best_results):
    best_results["CE_loss"] = results["CE_loss"]
    best_results["multi_class_acc"] = results["multi_class_acc"]

  def _update_summary(self, summary_writer, results, iter_count):
    summary = tf.Summary()
    summary.value.add(tag="acc", simple_value=results["multi_class_acc"])
    summary.value.add(tag="CE_loss", simple_value=results["CE_loss"])
    summary_writer.add_summary(summary, iter_count)

  def train(self):
    """ Train model """
    # build graph
    self._build_graph()
    logger.info("Building Model Done!")

    # set up summary
    train_stats_folder = os.path.join(self._save_dir, "train")
    val_stats_folder = os.path.join(self._save_dir, "val")
    mkdir(train_stats_folder)
    mkdir(val_stats_folder)
    train_writer = tf.summary.FileWriter(train_stats_folder,
                                         self._session.graph)
    val_writer = tf.summary.FileWriter(val_stats_folder, self._session.graph)

    # resume training or not
    if self._is_resume_training:
      self._model.saver.restore(self._session, self._resume_model_path)
    else:
      feed_data = self._model._construct_feeddict(self._data, is_train=True)
      self._session.run(self._model.assign_ops, feed_dict=feed_data)

    # list of ops to run in training/val
    train_op_names = ["summary", "CE_loss", "train_step", "weight_decay"]
    val_op_names = ["val_pred_logits"]

    # set up initial results
    best_val_results = {"CE_loss": 0.0, "multi_class_acc": 0.0}
    if self._dataset_name == "diel":
      best_val_results["recall"] = 0.0

    early_stopper = EarlyStopper(
        [0.0], win_size=self._early_stop_window, is_decrease=True)

    # train loop
    logger.info("Training Loop Start!")
    for iter_count in xrange(self._max_epoch):
      # validation
      if iter_count == 0 or iter_count % self._valid_iter == 0:
        val_results = self._run_valid_ops(val_op_names)

        # disp val stats
        self._disp_stats(val_results, disp_tag="Validation")

        # save val stats
        self._update_summary(val_writer, val_results, iter_count + 1)
        val_metric = val_results["multi_class_acc"]
        best_val_metric = best_val_results["multi_class_acc"]

        # snapshot the best model
        if val_metric > best_val_metric:
          self._update_best(val_results, best_val_results)
          self._save_model(
              os.path.join(self._save_dir, "model_snapshot_best.ckpt"))

        self._disp_stats(best_val_results, disp_tag="Best validation")

        # check early stop
        if early_stopper.tick([val_metric]) or val_metric == 1.0:
          self._save_model(
              os.path.join(self._save_dir, "model_snapshot_{:07d}.ckpt".format(
                  iter_count + 1)))
          break

      # train step
      train_results = self._get({
          self._model._dropout_rate: self._dropout
      }, train_op_names)

      # display training statistics
      if iter_count == 0 or (iter_count + 1) % self._display_iter == 0:
        train_writer.add_summary(train_results["summary"], iter_count + 1)
        logger.info(
            "Train Step = {:06d} || CE loss = {:e} || Weight Decay = {:e}".
            format(iter_count + 1, train_results["CE_loss"], train_results[
                "weight_decay"]))

      # save model
      if (iter_count + 1) % self._snapshot_iter == 0 or (
          iter_count + 1) == self._max_epoch:
        self._save_model(
            os.path.join(self._save_dir, "model_snapshot_{:07d}.ckpt".format(
                iter_count + 1)))

      iter_count += 1

    # do test after training
    self._model.saver.restore(self._session,
                              os.path.join(self._save_dir,
                                           "model_snapshot_best.ckpt"))
    test_op_names = ["test_pred_logits"]
    test_results = self._run_valid_ops(test_op_names, is_val=False)
    self._disp_stats(test_results, disp_tag="Testing")
    self._session.close()

    return test_results

  def test(self):
    """ Test model """
    # build graph
    self._build_graph()
    logger.info("Building Model Done!")

    self._model.saver.restore(self._session, self._test_model_path)
    logger.info("Loading Trained Model Done!")

    # list of ops to run in testing
    test_op_names = ["test_pred_logits"]

    # test loop
    logger.info("Testing Loop Start!")
    test_results = self._run_valid_ops(test_op_names, is_val=False)

    # disp test stats
    self._disp_stats(test_results, disp_tag="Testing")
    self._session.close()

    return test_results
