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

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
    "mig",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_mig(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                batch_size=16):
  """Computes the mutual information gap.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
  del artifact_dir
  logging.info("Generating training set.")
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  return _compute_mig(mus_train, ys_train)


def _compute_mig(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  discretized_mus = utils.make_discretizer(mus_train)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  assert m.shape[0] == mus_train.shape[0]
  assert m.shape[1] == ys_train.shape[0]
  # m is [num_latents, num_factors]
  entropy = utils.discrete_entropy(ys_train)
  sorted_m = np.sort(m, axis=0)[::-1]
  score_dict["discrete_mig"] = np.mean(
      np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
  return score_dict


@gin.configurable(
    "mig_validation",
    blacklist=["observations", "labels", "representation_function"])
def compute_mig_on_fixed_data(observations, labels, representation_function,
                              batch_size=100):
  """Computes the MIG scores on the fixed set of observations and labels.

  Args:
    observations: Observations on which to compute the score. Observations have
      shape (num_observations, 64, 64, num_channels).
    labels: Observed factors of variations.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Batch size used to compute the representation.

  Returns:
    MIG computed on the provided observations and labels.
  """
  mus = utils.obtain_representation(observations, representation_function,
                                    batch_size)
  assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
  assert mus.shape[1] == observations.shape[0], "Wrong representation shape."
  return _compute_mig(mus, labels)
