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

"""Visualization module for disentangled representations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
import os
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results
from disentanglement_lib.visualize import visualize_util
from disentanglement_lib.visualize.visualize_irs import vis_all_interventional_effects
import numpy as np
from scipy import stats
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import tensorflow_hub as hub
import gin.tf


def visualize(model_dir,
              output_dir,
              overwrite=False,
              num_animations=5,
              num_frames=20,
              fps=10,
              num_points_irs=10000):
  """Takes trained model from model_dir and visualizes it in output_dir.

  Args:
    model_dir: Path to directory where the trained model is saved.
    output_dir: Path to output directory.
    overwrite: Boolean indicating whether to overwrite output directory.
    num_animations: Integer with number of distinct animations to create.
    num_frames: Integer with number of frames in each animation.
    fps: Integer with frame rate for the animation.
    num_points_irs: Number of points to be used for the IRS plots.
  """
  # Fix the random seed for reproducibility.
  random_state = np.random.RandomState(0)

  # Create the output directory if necessary.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  # Obtain the dataset name from the gin config of the previous step.
  gin_config_file = os.path.join(model_dir, "results", "gin", "train.gin")
  gin_dict = results.gin_dict(gin_config_file)
  gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
      "'", ""))

  # Automatically infer the activation function from gin config.
  activation_str = gin_dict["reconstruction_loss.activation"]
  if activation_str == "'logits'":
    activation = sigmoid
  elif activation_str == "'tanh'":
    activation = tanh
  else:
    raise ValueError(
        "Activation function  could not be infered from gin config.")

  dataset = named_data.get_named_ground_truth_data()
  num_pics = 64
  module_path = os.path.join(model_dir, "tfhub")

  with hub.eval_function_for_module(module_path) as f:
    # Save reconstructions.
    real_pics = dataset.sample_observations(num_pics, random_state)
    raw_pics = f(
        dict(images=real_pics), signature="reconstructions",
        as_dict=True)["images"]
    pics = activation(raw_pics)
    paired_pics = np.concatenate((real_pics, pics), axis=2)
    paired_pics = [paired_pics[i, :, :, :] for i in range(paired_pics.shape[0])]
    results_dir = os.path.join(output_dir, "reconstructions")
    if not gfile.IsDirectory(results_dir):
      gfile.MakeDirs(results_dir)
    visualize_util.grid_save_images(
        paired_pics, os.path.join(results_dir, "reconstructions.jpg"))

    # Save samples.
    def _decoder(latent_vectors):
      return f(
          dict(latent_vectors=latent_vectors),
          signature="decoder",
          as_dict=True)["images"]

    num_latent = int(gin_dict["encoder.num_latent"])
    num_pics = 64
    random_codes = random_state.normal(0, 1, [num_pics, num_latent])
    pics = activation(_decoder(random_codes))
    results_dir = os.path.join(output_dir, "sampled")
    if not gfile.IsDirectory(results_dir):
      gfile.MakeDirs(results_dir)
    visualize_util.grid_save_images(pics,
                                    os.path.join(results_dir, "samples.jpg"))

    # Save latent traversals.
    result = f(
        dict(images=dataset.sample_observations(num_pics, random_state)),
        signature="gaussian_encoder",
        as_dict=True)
    means = result["mean"]
    logvars = result["logvar"]
    results_dir = os.path.join(output_dir, "traversals")
    if not gfile.IsDirectory(results_dir):
      gfile.MakeDirs(results_dir)
    for i in range(means.shape[1]):
      pics = activation(
          latent_traversal_1d_multi_dim(_decoder, means[i, :], None))
      file_name = os.path.join(results_dir, "traversals{}.jpg".format(i))
      visualize_util.grid_save_images([pics], file_name)

    # Save the latent traversal animations.
    results_dir = os.path.join(output_dir, "animated_traversals")
    if not gfile.IsDirectory(results_dir):
      gfile.MakeDirs(results_dir)

    # Cycle through quantiles of a standard Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
      images = []
      for j in range(base_code.shape[0]):
        code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
        code[:, j] = visualize_util.cycle_gaussian(base_code[j], num_frames)
        images.append(np.array(activation(_decoder(code))))
      filename = os.path.join(results_dir, "std_gaussian_cycle%d.gif" % i)
      visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle through quantiles of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
      images = []
      for j in range(base_code.shape[0]):
        code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
        loc = np.mean(means[:, j])
        total_variance = np.mean(np.exp(logvars[:, j])) + np.var(means[:, j])
        code[:, j] = visualize_util.cycle_gaussian(
            base_code[j], num_frames, loc=loc, scale=np.sqrt(total_variance))
        images.append(np.array(activation(_decoder(code))))
      filename = os.path.join(results_dir, "fitted_gaussian_cycle%d.gif" % i)
      visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle through [-2, 2] interval.
    for i, base_code in enumerate(means[:num_animations]):
      images = []
      for j in range(base_code.shape[0]):
        code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
        code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                   -2., 2.)
        images.append(np.array(activation(_decoder(code))))
      filename = os.path.join(results_dir, "fixed_interval_cycle%d.gif" % i)
      visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle linearly through +-2 std dev of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
      images = []
      for j in range(base_code.shape[0]):
        code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
        loc = np.mean(means[:, j])
        total_variance = np.mean(np.exp(logvars[:, j])) + np.var(means[:, j])
        scale = np.sqrt(total_variance)
        code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                   loc-2.*scale, loc+2.*scale)
        images.append(np.array(activation(_decoder(code))))
      filename = os.path.join(results_dir, "conf_interval_cycle%d.gif" % i)
      visualize_util.save_animation(np.array(images), filename, fps)

    # Cycle linearly through minmax of a fitted Gaussian.
    for i, base_code in enumerate(means[:num_animations]):
      images = []
      for j in range(base_code.shape[0]):
        code = np.repeat(np.expand_dims(base_code, 0), num_frames, axis=0)
        code[:, j] = visualize_util.cycle_interval(base_code[j], num_frames,
                                                   np.min(means[:, j]),
                                                   np.max(means[:, j]))
        images.append(np.array(activation(_decoder(code))))
      filename = os.path.join(results_dir, "minmax_interval_cycle%d.gif" % i)
      visualize_util.save_animation(np.array(images), filename, fps)

    # Interventional effects visualization.
    factors = dataset.sample_factors(num_points_irs, random_state)
    obs = dataset.sample_observations_from_factors(factors, random_state)
    latents = f(
        dict(images=obs), signature="gaussian_encoder", as_dict=True)["mean"]
    results_dir = os.path.join(output_dir, "interventional_effects")
    vis_all_interventional_effects(factors, latents, results_dir)

  # Finally, we clear the gin config that we have set.
  gin.clear_config()


def latent_traversal_1d_multi_dim(generator_fn,
                                  latent_vector,
                                  dimensions=None,
                                  values=None,
                                  transpose=False):
  """Creates latent traversals for a latent vector along multiple dimensions.

  Creates a 2d grid image where each grid image is generated by passing a
  modified version of latent_vector to the generator_fn. In each column, a
  fixed dimension of latent_vector is modified. In each row, the value in the
  modified dimension is replaced by a fixed value.

  Args:
    generator_fn: Function that computes (fixed size) images from latent
      representation. It should accept a single Numpy array argument of the same
      shape as latent_vector and return a Numpy array of images where the first
      dimension corresponds to the different vectors in latent_vectors.
    latent_vector: 1d Numpy array with the base latent vector to be used.
    dimensions: 1d Numpy array with the indices of the dimensions that should be
      modified. If an integer is passed, the dimensions 0, 1, ...,
      (dimensions - 1) are modified. If None is passed, all dimensions of
      latent_vector are modified.
    values: 1d Numpy array with the latent space values that should be used for
      modifications. If an integer is passed, a linear grid between -1 and 1
      with that many points is constructed. If None is passed, a default grid is
      used (whose specific design is not guaranteed).
    transpose: Boolean which indicates whether rows and columns of the 2d grid
      should be transposed.

  Returns:
    Numpy array with image.
  """
  if latent_vector.ndim != 1:
    raise ValueError("Latent vector needs to be 1-dimensional.")

  if dimensions is None:
    # Default case, use all available dimensions.
    dimensions = np.arange(latent_vector.shape[0])
  elif isinstance(dimensions, numbers.Integral):
    # Check that there are enough dimensions in latent_vector.
    if dimensions > latent_vector.shape[0]:
      raise ValueError("The number of dimensions of latent_vector is less than"
                       " the number of dimensions requested in the arguments.")
    if dimensions < 1:
      raise ValueError("The number of dimensions has to be at least 1.")
    dimensions = np.arange(dimensions)
  if dimensions.ndim != 1:
    raise ValueError("Dimensions vector needs to be 1-dimensional.")

  if values is None:
    # Default grid of values.
    values = np.linspace(-1., 1., num=11)
  elif isinstance(values, numbers.Integral):
    if values <= 1:
      raise ValueError("If an int is passed for values, it has to be >1.")
    values = np.linspace(-1., 1., num=values)
  if values.ndim != 1:
    raise ValueError("Values vector needs to be 1-dimensional.")

  # We iteratively generate the rows/columns for each dimension as different
  # Numpy arrays. We do not preallocate a single final Numpy array as this code
  # is not performance critical and as it reduces code complexity.
  num_values = len(values)
  row_or_columns = []
  for dimension in dimensions:
    # Creates num_values copy of the latent_vector along the first axis.
    latent_traversal_vectors = np.tile(latent_vector, [num_values, 1])
    # Intervenes in the latent space.
    latent_traversal_vectors[:, dimension] = values
    # Generate the batch of images
    images = generator_fn(latent_traversal_vectors)
    # Adds images as a row or column depending whether transpose is True.
    axis = (1 if transpose else 0)
    row_or_columns.append(np.concatenate(images, axis))
  axis = (0 if transpose else 1)
  return np.concatenate(row_or_columns, axis)


def sigmoid(x):
  return stats.logistic.cdf(x)


def tanh(x):
  return np.tanh(x) / 2. + .5
