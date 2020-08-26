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

"""Evaluation protocol to compute metrics.

As opposed to evaluate.py, this function evaluates the representation on fixed
data. This is used to perform model selection and make sure that each model had
access to the same labeled data.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import semi_supervised_utils  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import gin.tf


def validate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
  """Validate a representation based on the provided gin configuration.

  This function will set the provided gin bindings, call the evaluate()
  function and clear the gin config. Please see the evaluate() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the representation is saved.
    output_dir: String with the path where the evaluation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  validate(model_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "validation", blacklist=["model_dir", "output_dir", "overwrite"])
def validate(model_dir,
             output_dir,
             overwrite=False,
             validation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             num_labelled_samples=gin.REQUIRED,
             name=""):
  """Loads a representation TFHub module and computes disentanglement metrics.

  Args:
    model_dir: String with path to directory where the representation function
      is saved.
    output_dir: String with the path where the results should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    validation_fn: Function used to validate the representation (see metrics/
      for examples).
    random_seed: Integer with random seed used for training.
    num_labelled_samples: How many labelled samples are available.
    name: Optional string with name of the metric (can be used to name metrics).
  """
  # We do not use the variable 'name'. Instead, it can be used to name scores
  # as it will be part of the saved gin config.
  del name

  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dir, "results", "gin",
                                   "postprocess.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
          "'", ""))
  dataset = named_data.get_named_ground_truth_data()
  observations, labels, _ = semi_supervised_utils.sample_supervised_data(
      random_seed, dataset, num_labelled_samples)
  # Path to TFHub module of previously trained representation.
  module_path = os.path.join(model_dir, "tfhub")
  with hub.eval_function_for_module(module_path) as f:

    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])

    # Computes scores of the representation based on the evaluation_fn.
    results_dict = validation_fn(observations, np.transpose(labels),
                                 _representation_function)

  # Save the results (and all previous results in the pipeline) on disk.
  original_results_dir = os.path.join(model_dir, "results")
  results_dir = os.path.join(output_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "validation", results_dict,
                                  original_results_dir)
