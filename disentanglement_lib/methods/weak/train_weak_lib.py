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

"""Main training protocol used for weakly-supervised disentanglement models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.methods.weak import weak_vae  # pylint: disable=unused-import
from disentanglement_lib.utils import results
from disentanglement_lib.visualize import visualize_util
import numpy as np
import tensorflow as tf

import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimator


@gin.configurable("dynamics", blacklist=["z", "ground_truth_data",
                                         "random_state",
                                         "return_index"])
def simple_dynamics(z, ground_truth_data, random_state,
                    return_index=False, k=gin.REQUIRED):
  """Create the pairs."""
  if k == -1:
    k_observed = random_state.randint(1, ground_truth_data.num_factors)
  else:
    k_observed = k
  index_list = random_state.choice(
      z.shape[1], random_state.choice([1, k_observed]), replace=False)
  idx = -1
  for index in index_list:
    z[:, index] = np.random.choice(
        range(ground_truth_data.factors_num_values[index]))
    idx = index
  if return_index:
    return z, idx
  return z, k_observed


def train_with_gin(model_dir,
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None):
  """Trains a model based on the provided gin configuration.

  This function will set the provided gin bindings, call the train() function
  and clear the gin config. Please see the train() for required gin bindings.

  Args:
    model_dir: String with path to directory where model output should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  train(model_dir, overwrite)
  gin.clear_config()


@gin.configurable("model", blacklist=["model_dir", "overwrite"])
def train(model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          random_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          name=""):
  """Trains the estimator and exports the snapshot and the gin config.

  The use of this function requires the gin binding 'dataset.name' to be
  specified as that determines the data set used for training.

  Args:
    model_dir: String with path to directory where model output should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    model: GaussianEncoderModel that should be trained and exported.
    training_steps: Integer with number of training steps.
    random_seed: Integer with random seed used for training.
    batch_size: Integer with the batch size.
    name: Optional string with name of the model (can be used to name models).
  """
  # We do not use the variable 'name'. Instead, it can be used to name results
  # as it will be part of the saved gin config.
  del name

  # Delete the output directory if necessary.
  if tf.compat.v1.gfile.IsDirectory(model_dir):
    if overwrite:
      tf.compat.v1.gfile.DeleteRecursively(model_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Create a numpy random state. We will sample the random seeds for training
  # and evaluation from this.
  random_state = np.random.RandomState(random_seed)

  # Obtain the dataset.
  dataset = named_data.get_named_ground_truth_data()
  # We create a TPUEstimator based on the provided model. This is primarily so
  # that we could switch to TPU training in the future. For now, we train
  # locally on GPUs.
  run_config = tpu_config.RunConfig(
      tf_random_seed=random_seed,
      keep_checkpoint_max=1,
      tpu_config=tpu_config.TPUConfig(iterations_per_loop=500))
  tpu_estimator = TPUEstimator(
      use_tpu=False,
      model_fn=model.model_fn,
      model_dir=model_dir,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      config=run_config)

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()
  # Do the actual training.
  tpu_estimator.train(
      input_fn=_make_input_fn(dataset, random_state.randint(2**32)),
      steps=training_steps)
  # Save model as a TFHub module.
  output_shape = named_data.get_named_ground_truth_data().observation_shape
  module_export_path = os.path.join(model_dir, "tfhub")
  gaussian_encoder_model.export_as_tf_hub(model, output_shape,
                                          tpu_estimator.latest_checkpoint(),
                                          module_export_path)

  # Save the results. The result dir will contain all the results and config
  # files that we copied along, as we progress in the pipeline. The idea is that
  # these files will be available for analysis at the end.
  results_dict = tpu_estimator.evaluate(
      input_fn=_make_input_fn(
          dataset, random_state.randint(2**32), num_batches=1000
          ))
  results_dir = os.path.join(model_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "train", results_dict)
  visualize_dir = os.path.join(model_dir, "visualizations")
  visualize_weakly_supervised_dataset(
      dataset, os.path.join(visualize_dir, "weak"), num_frames=1)


def _make_input_fn(ground_truth_data, seed, num_batches=None):
  """Creates an input function for the experiments."""

  def load_dataset(params):
    """TPUEstimator compatible input fuction."""
    dataset = weak_dataset_from_ground_truth_data(ground_truth_data, seed)
    batch_size = params["batch_size"]
    # We need to drop the remainder as otherwise we lose the batch size in the
    # tensor shape. This has no effect as our data set is infinite.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if num_batches is not None:
      dataset = dataset.take(num_batches)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return load_dataset


def weak_dataset_from_ground_truth_data(
    ground_truth_data, random_seed):
  """Generate a tf.data.DataSet disentanglement on weakly-supervised data.

  In this setting we have pairs of frames either temporally close to each other
  or randomly chosen.

  Args:
    ground_truth_data: Dataset class.
    random_seed: Random seed.

  Returns:
    tf.data.Dataset, each point contains two frames stacked along the first dim
    and integer labels.
    For dSprites each point is of type np.array(128, 64, 1).
  """

  def _generator():
    """Generator fn for the dataset."""
    # We need to hard code the random seed so that the data set can be reset.
    random_state = np.random.RandomState(random_seed)
    while True:
      sampled_factors = ground_truth_data.sample_factors(1, random_state)
      sampled_observation = ground_truth_data.sample_observations_from_factors(
          sampled_factors, random_state)

      next_factors, index = simple_dynamics(sampled_factors,
                                            ground_truth_data,
                                            random_state,
                                            return_index=True)
      next_observation = ground_truth_data.sample_observations_from_factors(
          next_factors, random_state)

      label = index
      yield (np.concatenate((sampled_observation, next_observation),
                            axis=1)[0], [label])

  dataset_shape = np.copy(ground_truth_data.observation_shape)
  dataset_shape[0] = dataset_shape[0] * 2
  weakly_supervised_dataset = \
      tf.data.Dataset.from_generator(
          _generator,
          (tf.float32, tf.int32),
          output_shapes=(dataset_shape, 1))

  return weakly_supervised_dataset


def visualize_weakly_supervised_dataset(
    data, path, num_animations=10, num_frames=20, fps=10):
  """Visualizes the data set by saving images to output_path.

  For each latent factor, outputs 16 images where only that latent factor is
  varied while all others are kept constant.

  Args:
    data: String with name of dataset as defined in named_data.py.
    path: String with path in which to create the visualizations.
    num_animations: Integer with number of distinct animations to create.
    num_frames: Integer with number of frames in each animation.
    fps: Integer with frame rate for the animation.
  """
  random_state = np.random.RandomState(0)

  # Create output folder if necessary.
  if not tf.compat.v1.gfile.IsDirectory(path):
    tf.compat.v1.gfile.MakeDirs(path)

  # Create animations.
  images = []
  for i in range(num_animations):
    images.append([])
    factor = data.sample_factors(1, random_state)

    images[i].append(
        np.squeeze(
            data.sample_observations_from_factors(factor, random_state),
            axis=0))
    for _ in range(num_frames):
      factor, _ = simple_dynamics(factor, data, random_state)
      images[i].append(
          np.squeeze(
              data.sample_observations_from_factors(factor, random_state),
              axis=0))

  visualize_util.save_animation(
      np.array(images), os.path.join(path, "animation.gif"), fps)
