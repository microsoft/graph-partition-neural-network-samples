import tensorflow as tf


def aggregate(data, agg_idx, new_size, method="sum"):
  """ Aggregate data

  Args:
    data: tf tensor, see "unsorted_segment_x" in tf documents for more detail
    agg_idx: tf tensor of int, index for aggregation
    new_size: tf tensor of int, size of the data after aggregation
    method: aggregation method

  Returns:
    agg_data: tf tensor, aggregated data
  """

  if method == "sum":
    agg_data = tf.unsorted_segment_sum(data, agg_idx, new_size)
  elif method == "avg":
    agg_data = tf.unsorted_segment_sum(data, agg_idx, new_size)
    denom_const = tf.unsorted_segment_sum(tf.ones_like(data), agg_idx, new_size)
    agg_data = tf.div(agg_data, (denom_const + tf.constant(1.0e-10)))
  elif method == "max":
    agg_data = tf.unsorted_segment_max(data, agg_idx, new_size)
  elif method == "min":
    agg_data = tf.unsorted_segment_max(-data, agg_idx, new_size)
  else:
    raise ValueError("Unsupported aggregation method!")

  return agg_data
