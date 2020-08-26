# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation module for computing UDR.

Binary for computing the UDR and UDR-A2A scores specified in "Unsupervised
Model Selection for Variational Disentangled Representation Learning"
(https://arxiv.org/abs/1905.12614)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

from absl import flags
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.udr.metrics import udr  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import gin.tf

FLAGS = flags.FLAGS


@gin.configurable("evaluation", blacklist=["model_dirs", "output_dir"])
def evaluate(model_dirs,
             output_dir,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):
  """Loads a trained estimator and evaluates it according to beta-VAE metric."""
  # The name will be part of the gin config and can be used to tag results.
  del name

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper dataset if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the dataset
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dirs[0], "results", "gin", "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      print(gin_dict["dataset.name"])
      gin.bind_parameter("dataset.name",
                         gin_dict["dataset.name"].replace("'", ""))

  output_dir = os.path.join(output_dir)
  if tf.io.gfile.isdir(output_dir):
    tf.io.gfile.rmtree(output_dir)

  dataset = named_data.get_named_ground_truth_data()

  with contextlib.ExitStack() as stack:
    representation_functions = []
    eval_functions = [
        stack.enter_context(
            hub.eval_function_for_module(os.path.join(model_dir, "tfhub")))
        for model_dir in model_dirs
    ]
    for f in eval_functions:

      def _representation_function(x, f=f):

        def compute_gaussian_kl(z_mean, z_logvar):
          return np.mean(
              0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1),
              axis=0)

        encoding = f(dict(images=x), signature="gaussian_encoder", as_dict=True)

        return np.array(encoding["mean"]), compute_gaussian_kl(
            np.array(encoding["mean"]), np.array(encoding["logvar"]))

      representation_functions.append(_representation_function)

    results_dict = evaluation_fn(
        dataset,
        representation_functions,
        random_state=np.random.RandomState(random_seed))

  original_results_dir = os.path.join(model_dirs[0], "results")
  results_dir = os.path.join(output_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "evaluation", results_dict,
                                  original_results_dir)
