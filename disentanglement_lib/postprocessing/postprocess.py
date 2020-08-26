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

"""Postprocessing step that extracts representation from trained model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.postprocessing import methods  # pylint: disable=unused-import
from disentanglement_lib.utils import convolute_hub
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import gin.tf


def postprocess_with_gin(model_dir,
                         output_dir,
                         overwrite=False,
                         gin_config_files=None,
                         gin_bindings=None):
  """Postprocess a trained model based on the provided gin configuration.

  This function will set the provided gin bindings, call the postprocess()
  function and clear the gin config. Please see the postprocess() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the model is saved.
    output_dir: String with the path where the representation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  postprocess(model_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "postprocess", blacklist=["model_dir", "output_dir", "overwrite"])
def postprocess(model_dir,
                output_dir,
                overwrite=False,
                postprocess_fn=gin.REQUIRED,
                random_seed=gin.REQUIRED,
                name=""):
  """Loads a trained Gaussian encoder and extracts representation.

  Args:
    model_dir: String with path to directory where the model is saved.
    output_dir: String with the path where the representation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    postprocess_fn: Function used to extract the representation (see methods.py
      for examples).
    random_seed: Integer with random seed used for postprocessing (may be
      unused).
    name: Optional string with name of the representation (can be used to name
      representations).
  """
  # We do not use the variable 'name'. Instead, it can be used to name
  # representations as it will be part of the saved gin config.
  del name

  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Set up timer to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dir, "results", "gin", "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
          "'", ""))
  dataset = named_data.get_named_ground_truth_data()

  # Path to TFHub module of previously trained model.
  module_path = os.path.join(model_dir, "tfhub")
  with hub.eval_function_for_module(module_path) as f:

    def _gaussian_encoder(x):
      """Encodes images using trained model."""
      # Push images through the TFHub module.
      output = f(dict(images=x), signature="gaussian_encoder", as_dict=True)
      # Convert to numpy arrays and return.
      return {key: np.array(values) for key, values in output.items()}

    # Run the postprocessing function which returns a transformation function
    # that can be used to create the representation from the mean and log
    # variance of the Gaussian distribution given by the encoder. Also returns
    # path to a checkpoint if the transformation requires variables.
    transform_fn, transform_checkpoint_path = postprocess_fn(
        dataset, _gaussian_encoder, np.random.RandomState(random_seed),
        output_dir)

    # Takes the "gaussian_encoder" signature, extracts the representation and
    # then saves under the signature "representation".
    tfhub_module_dir = os.path.join(output_dir, "tfhub")
    convolute_hub.convolute_and_save(
        module_path, "gaussian_encoder", tfhub_module_dir, transform_fn,
        transform_checkpoint_path, "representation")

  # We first copy over all the prior results and configs.
  original_results_dir = os.path.join(model_dir, "results")
  results_dir = os.path.join(output_dir, "results")
  results_dict = dict(elapsed_time=time.time() - experiment_timer)
  results.update_result_directory(results_dir, "postprocess", results_dict,
                                  original_results_dir)
