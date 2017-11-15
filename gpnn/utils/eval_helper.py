import numpy as np

EPS = np.finfo(np.float32).eps


def compute_accuracy(pred_logits, gt_logits):
  """ Compute Accuracy

    Input: 
        pred_logits: N X K numpy array (K: number of classes)
        gt_logits:  N X 1 numpy array

    Output:
        acc: scalar
  """

  num_sample = pred_logits.shape[0]
  pred_idx = np.argmax(pred_logits, axis=1)
  acc = np.sum(np.equal(pred_idx, gt_logits).astype(float)) / num_sample

  return acc


def compute_PR_binary(pred_logits, gt_logits, threshold=0.0):
  """ Compute Precision & Recall of Binary Classification

    Input: 
        pred_logits: N X 2 numpy array
        gt_logits:  N X 1 numpy array

    Output:
        precision: list, precision
        recall: list, recall
  """
  pred_logits += np.array([[threshold, 0.0]])
  pred_idx = np.argmax(pred_logits, axis=1)
  gt_logits = gt_logits.astype(np.int32)

  if pred_idx.ndim == 1:
    pred_idx = np.expand_dims(pred_idx, axis=1)

  if gt_logits.ndim == 1:
    gt_logits = np.expand_dims(gt_logits, axis=1)

  idx_pred_P = (pred_idx == 1).astype(float)
  idx_gt_P = (gt_logits == 1).astype(float)
  idx_pred_N = (pred_idx == 0).astype(float)
  idx_gt_N = (gt_logits == 0).astype(float)

  TP = np.sum(idx_pred_P * idx_gt_P)
  FP = np.sum(idx_pred_P * idx_gt_N)
  TN = np.sum(idx_pred_N * idx_gt_N)
  FN = np.sum(idx_pred_N * idx_gt_P)

  precision = TP / (TP + FP + EPS)
  recall = TP / (TP + FN + EPS)
  specificity = TN / (TN + FP)
  accuracy = (TP + TN) / (TP + TN + FP + FN)

  return precision, recall, specificity, accuracy


def compute_PR_multiclass(pred_logits, gt_logits):
  """ Compute Precision & Recall of Multi-Class (1 out of C) Classification

    Input: 
        pred_logits: N X C numpy array, must be 0 or 1
        gt_logits:  N X C numpy array, must be 0 or 1

    Output:
        precision: list, precision
        recall: list, recall
  """
  pred_logits = pred_logits.astype(np.int32)
  gt_logits = gt_logits.astype(np.int32)

  if pred_logits.ndim == 1:
    pred_logits = np.expand_dims(pred_logits, axis=1)

  if gt_logits.ndim == 1:
    gt_logits = np.expand_dims(gt_logits, axis=1)

  num_class = pred_logits.shape[1]
  precision = [0.0] * num_class
  recall = [0.0] * num_class
  TP_all = 0
  FP_all = 0
  FN_all = 0

  for ii in xrange(num_class):
    idx_pred_P = (pred_logits[:, ii] == 1).astype(float)
    idx_gt_P = (gt_logits[:, ii] == 1).astype(float)
    idx_pred_N = (pred_logits[:, ii] == 0).astype(float)
    idx_gt_N = (gt_logits[:, ii] == 0).astype(float)

    TP = np.sum(idx_pred_P * idx_gt_P)
    FP = np.sum(idx_pred_P * idx_gt_N)
    FN = np.sum(idx_pred_N * idx_gt_P)
    TP_all += TP
    FP_all += FP
    FN_all += FN

    precision[ii] = TP / (TP + FP + EPS)
    recall[ii] = TP / (TP + FN + EPS)

  precision_all = TP_all / (TP_all + FP_all + EPS)
  recall_all = TP_all / (TP_all + FN_all + EPS)

  micro_F1 = comput_micro_F1(precision_all, recall_all)
  macro_F1 = comput_macro_F1(precision, recall)

  return precision, recall, precision_all, recall_all, micro_F1, macro_F1


def compute_acc_multiclass(pred_logits, gt_logits):
  """ Compute Precision & Recall of Multi-Class (1 out of C) Classification

    Input: 
        pred_logits: N X C numpy array, must be 0 or 1
        gt_logits:  N X C numpy array, must be 0 or 1

    Output:
        precision: list, precision
        recall: list, recall
  """
  num_class = pred_logits.shape[1]
  pred_logits = pred_logits.astype(np.int32)
  gt_logits = gt_logits.astype(np.int32)

  if pred_logits.ndim == 1:
    pred_logits = np.expand_dims(pred_logits, axis=1)

  if gt_logits.ndim == 1:
    gt_logits = np.expand_dims(gt_logits, axis=1)

  acc = [0] * num_class
  num_sample = [0] * num_class

  for ii in xrange(num_class):
    idx_pred_P = (pred_logits[:, ii] == 1).astype(float)
    idx_gt_P = (gt_logits[:, ii] == 1).astype(float)
    num_sample[ii] = np.sum(idx_gt_P)
    acc[ii] = np.sum(idx_pred_P * idx_gt_P) / (num_sample[ii] + EPS)

  return acc, num_sample


def comput_micro_F1(precision, recall):
  """ compute Micro F1 score """
  return 2.0 * precision * recall / (precision + recall + EPS)


def comput_macro_F1(precision, recall):
  """ compute Macro F1 score """

  mean_precision = np.mean(precision)
  mean_recall = np.mean(recall)

  return comput_micro_F1(mean_precision, mean_recall)
