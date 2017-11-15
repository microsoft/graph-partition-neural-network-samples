import argparse


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Running Experiments of Graph Partition Neural Networks")
  parser.add_argument(
      "-c",
      "--config_file",
      type=str,
      default="exp/gpnn_cora.json",
      required=True,
      help="Path of config file")
  parser.add_argument(
      "-l",
      "--log_level",
      type=str,
      default="INFO",
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument("-m", "--comment", help="Experiment comment")
  parser.add_argument("-t", "--test", help="Test model", action="store_true")
  args = parser.parse_args()

  return args
