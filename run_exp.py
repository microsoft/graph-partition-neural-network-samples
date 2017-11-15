from __future__ import (absolute_import, division, print_function)

import os
import sys
import time
import json
import traceback
import numpy as np

from arg_config import parse_arguments
from gpnn.utils.logger import setup_logging
from gpnn.factory import RunnerFactory
from gpnn.factory import ModelFactory
from gpnn.factory import ReaderFactory
from gpnn.utils.runner_helper import mkdir

np.set_printoptions(threshold=np.inf)


def get_param(config_file):
  """ Construct and snapshot hyper parameters """
  config = json.loads(open(config_file, "r").read())
  dataset_info = json.loads(open("config/dataset_info.json", "r").read())

  # create run id
  run_id = str(os.getpid())
  exp_name = "_".join([
      config["model_name"], config["dataset_name"],
      time.strftime("%Y-%b-%d-%H-%M-%S"), run_id
  ])

  # create hyper parameters
  param = {"run_id": run_id, "exp_name": exp_name}
  dataset_name_map = dict([(xx["dataset_name"], xx["dataset_id"])
                           for xx in dataset_info])
  param.update(config)
  param.update(dataset_info[dataset_name_map[config["dataset_name"]]])
  param["save_dir"] = os.path.join(param["exp_dir"], exp_name)

  # snapshot hyperparameters
  mkdir(param["exp_dir"])
  mkdir(param["save_dir"])

  param_filename = os.path.join(param["save_dir"], "params.json")
  json.dump(param, open(param_filename, "w"), indent=2)

  return param


def main():
  args = parse_arguments()
  param = get_param(args.config_file)
  np.random.seed(param["seed"])

  # log info
  log_file = os.path.join(param["save_dir"],
                          "log_exp_{}.txt".format(param["run_id"]))

  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(param["run_id"]))
  logger.info("Exp comment = {}".format(args.comment))

  # Run the experiment
  try:
    runner = RunnerFactory.factory(param["runner_name"])(param)

    if not args.test:
      runner.train()
    else:
      runner.test()

  except:
    type_info, value, tb = sys.exc_info()
    output_info = traceback.format_exc(tb)
    logger.error(output_info)

  sys.exit(0)


if __name__ == "__main__":
  main()
