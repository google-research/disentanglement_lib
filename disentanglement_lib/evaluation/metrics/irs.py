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

"""Interventional Robustness Score.

Based on the paper https://arxiv.org/abs/1811.00007.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
    "irs",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_irs(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                diff_quantile=0.99,
                num_train=gin.REQUIRED,
                batch_size=gin.REQUIRED):
  """Computes the Interventional Robustness Score.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with IRS and number of active dimensions.
  """
  del artifact_dir
  logging.info("Generating training set.")
  mus, ys = utils.generate_batch_factor_code(ground_truth_data,
                                             representation_function, num_train,
                                             random_state, batch_size)
  assert mus.shape[1] == num_train

  ys_discrete = utils.make_discretizer(ys)
  active_mus = _drop_constant_dims(mus)

  if not active_mus.any():
    irs_score = 0.0
  else:
    irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T,
                                               diff_quantile)["avg_score"]

  score_dict = {}
  score_dict["IRS"] = irs_score
  score_dict["num_active_dims"] = np.sum(active_mus)
  return score_dict


def _drop_constant_dims(ys):
  """Returns a view of the matrix `ys` with dropped constant rows."""
  ys = np.asarray(ys)
  if ys.ndim != 2:
    raise ValueError("Expecting a matrix.")

  variances = ys.var(axis=1)
  active_mask = variances > 0.
  return ys[active_mask, :]


def scalable_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
  """Computes IRS scores of a dataset.

  Assumes no noise in X and crossed generative factors (i.e. one sample per
  combination of gen_factors). Assumes each g_i is an equally probable
  realization of g_i and all g_i are independent.

  Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

  Returns:
    Dictionary with IRS scores.
  """
  num_gen = gen_factors.shape[1]
  num_lat = latents.shape[1]

  # Compute normalizer.
  max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
  cum_deviations = np.zeros([num_lat, num_gen])
  for i in range(num_gen):
    unique_factors = np.unique(gen_factors[:, i], axis=0)
    assert unique_factors.ndim == 1
    num_distinct_factors = unique_factors.shape[0]
    for k in range(num_distinct_factors):
      # Compute E[Z | g_i].
      match = gen_factors[:, i] == unique_factors[k]
      e_loc = np.mean(latents[match, :], axis=0)

      # Difference of each value within that group of constant g_i to its mean.
      diffs = np.abs(latents[match, :] - e_loc)
      max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
      cum_deviations[:, i] += max_diffs
    cum_deviations[:, i] /= num_distinct_factors
  # Normalize value of each latent dimension with its maximal deviation.
  normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
  irs_matrix = 1.0 - normalized_deviations
  disentanglement_scores = irs_matrix.max(axis=1)
  if np.sum(max_deviations) > 0.0:
    avg_score = np.average(disentanglement_scores, weights=max_deviations)
  else:
    avg_score = np.mean(disentanglement_scores)

  parents = irs_matrix.argmax(axis=1)
  score_dict = {}
  score_dict["disentanglement_scores"] = disentanglement_scores
  score_dict["avg_score"] = avg_score
  score_dict["parents"] = parents
  score_dict["IRS_matrix"] = irs_matrix
  score_dict["max_deviations"] = max_deviations
  return score_dict
