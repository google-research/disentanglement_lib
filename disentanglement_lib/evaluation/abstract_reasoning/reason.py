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

"""Main training protocol used for computing abstract reasoning scores.

This is the main pipeline for the reasoning step in the paper:
Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
https://arxiv.org/abs/1905.12506
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from disentanglement_lib.evaluation.abstract_reasoning import models  # pylint: disable=unused-import
from disentanglement_lib.evaluation.abstract_reasoning import pgm_data
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf
from tensorflow.contrib import tpu as contrib_tpu


def reason_with_gin(input_dir,
                    output_dir,
                    overwrite=False,
                    gin_config_files=None,
                    gin_bindings=None):
  """Trains a model based on the provided gin configuration.

  This function will set the provided gin bindings, call the reason() function
  and clear the gin config. Please see reason() for required gin bindings.

  Args:
    input_dir: String with path to directory where the representation is saved.
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
  reason(input_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "abstract_reasoning", blacklist=["input_dir", "output_dir", "overwrite"])
def reason(
    input_dir,
    output_dir,
    overwrite=False,
    model=gin.REQUIRED,
    num_iterations=gin.REQUIRED,
    training_steps_per_iteration=gin.REQUIRED,
    eval_steps_per_iteration=gin.REQUIRED,
    random_seed=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    name="",
):
  """Trains the estimator and exports the snapshot and the gin config.

  The use of this function requires the gin binding 'dataset.name' to be
  specified if a model is trained from scratch as that determines the data set
  used for training.

  Args:
    input_dir: String with path to directory where the representation function
      is saved.
    output_dir: String with the path where the results should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    model: GaussianEncoderModel that should be trained and exported.
    num_iterations: Integer with number of training steps.
    training_steps_per_iteration: Integer with number of training steps per
      iteration.
    eval_steps_per_iteration: Integer with number of validationand test steps
      per iteration.
    random_seed: Integer with random seed used for training.
    batch_size: Integer with the batch size.
    name: Optional string with name of the model (can be used to name models).
  """
  # We do not use the variable 'name'. Instead, it can be used to name results
  # as it will be part of the saved gin config.
  del name

  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Create a numpy random state. We will sample the random seeds for training
  # and evaluation from this.
  random_state = np.random.RandomState(random_seed)

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    if input_dir is None:
      raise ValueError("Cannot automatically infer data set for methods with"
                       " no prior model directory.")
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(input_dir, "results", "gin",
                                   "postprocess.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      gin.bind_parameter("dataset.name",
                         gin_dict["dataset.name"].replace("'", ""))
  dataset = pgm_data.get_pgm_dataset()

  # Set the path to the TFHub embedding if we are training based on a
  # pre-trained embedding..
  if input_dir is not None:
    tfhub_dir = os.path.join(input_dir, "tfhub")
    with gin.unlock_config():
      gin.bind_parameter("HubEmbedding.hub_path", tfhub_dir)

  # We create a TPUEstimator based on the provided model. This is primarily so
  # that we could switch to TPU training in the future. For now, we train
  # locally on GPUs.
  run_config = contrib_tpu.RunConfig(
      tf_random_seed=random_seed,
      keep_checkpoint_max=1,
      tpu_config=contrib_tpu.TPUConfig(iterations_per_loop=500))
  tpu_estimator = contrib_tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model.model_fn,
      model_dir=os.path.join(output_dir, "tf_checkpoint"),
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      config=run_config)

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Create a dictionary to keep track of all relevant information.
  results_dict_of_dicts = {}
  validation_scores = []
  all_dicts = []

  for i in range(num_iterations):
    steps_so_far = i * training_steps_per_iteration
    tf.logging.info("Training to %d steps.", steps_so_far)
    # Train the model for the specified steps.
    tpu_estimator.train(
        input_fn=dataset.make_input_fn(random_state.randint(2**32)),
        steps=training_steps_per_iteration)
    # Compute validation scores used for model selection.
    validation_results = tpu_estimator.evaluate(
        input_fn=dataset.make_input_fn(
            random_state.randint(2**32), num_batches=eval_steps_per_iteration))
    validation_scores.append(validation_results["accuracy"])
    tf.logging.info("Validation results %s", validation_results)
    # Compute test scores for final results.
    test_results = tpu_estimator.evaluate(
        input_fn=dataset.make_input_fn(
            random_state.randint(2**32), num_batches=eval_steps_per_iteration),
        name="test")
    dict_at_iteration = results.namespaced_dict(
        val=validation_results, test=test_results)
    results_dict_of_dicts["step{}".format(steps_so_far)] = dict_at_iteration
    all_dicts.append(dict_at_iteration)

  # Select the best number of steps based on the validation scores and add it as
  # as a special key to the dictionary.
  best_index = np.argmax(validation_scores)
  results_dict_of_dicts["best"] = all_dicts[best_index]

  # Save the results. The result dir will contain all the results and config
  # files that we copied along, as we progress in the pipeline. The idea is that
  # these files will be available for analysis at the end.
  if input_dir is not None:
    original_results_dir = os.path.join(input_dir, "results")
  else:
    original_results_dir = None
  results_dict = results.namespaced_dict(**results_dict_of_dicts)
  results_dir = os.path.join(output_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "abstract_reasoning",
                                  results_dict, original_results_dir)
