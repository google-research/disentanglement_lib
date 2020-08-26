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

"""Different functions to extract representations from a Gaussian encoder.

Currently, only the mean of the Gaussian and a random sample from the Gaussian
are supported. However, the interface is set up such that data dependent and
potentially random transformations with learned variables are supported.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import gin.tf


@gin.configurable(
    "mean_representation",
    blacklist=["ground_truth_data", "gaussian_encoder", "random_state"])
def mean_representation(
    ground_truth_data,
    gaussian_encoder,
    random_state,
    save_path,
):
  """Extracts the mean representation from a Gaussian encoder.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    gaussian_encoder: Function that takes observations as input and outputs a
      dictionary with mean and log variances of the encodings in the keys "mean"
      and "logvar" respectively.
    random_state: Numpy random state used for randomness.
    save_path: String with path where results can be saved.

  Returns:
    transform_fn: Function that takes as keyword arguments the "mean" and
      "logvar" tensors and returns a tensor with the representation.
    None as no variables are saved.

  """
  del ground_truth_data, gaussian_encoder, random_state, save_path

  def transform_fn(mean, logvar):
    del logvar
    return mean

  return transform_fn, None


@gin.configurable(
    "sampled_representation",
    blacklist=["ground_truth_data", "gaussian_encoder", "random_state"])
def sampled_representation(ground_truth_data, gaussian_encoder, random_state,
                           save_path):
  """Extracts the random representation from a Gaussian encoder.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    gaussian_encoder: Function that takes observations as input and outputs a
      dictionary with mean and log variances of the encodings in the keys "mean"
      and "logvar" respectively.
    random_state: Numpy random state used for randomness.
    save_path: String with path where results can be saved.

  Returns:
    transform_fn: Function that takes as keyword arguments the "mean" and
      "logvar" tensors and returns a tensor with the representation.
    None as no variables are saved.
  """
  del ground_truth_data, gaussian_encoder, random_state, save_path

  def transform_fn(mean, logvar):
    return tf.add(mean,
                  tf.exp(logvar / 2) * tf.random_normal(tf.shape(mean), 0, 1))

  return transform_fn, None
