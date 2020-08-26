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

"""Implementation of a unified disentanglement score.

This score generalizes several existing disentanglement scores (DCI, MIG,
Modularity and SAP). First, we compute a matrix relating factors of variations
and latent codes. Then, we aggregate this matrix into a score.
DCI, MIG, Modularity and SAP can be obtained as a special case combining their
matrix with the corresponding aggregation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.evaluation.metrics import modularity_explicitness
from disentanglement_lib.evaluation.metrics import sap_score
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.utils import results
from disentanglement_lib.visualize import dendrogram
from disentanglement_lib.visualize import visualize_scores
import numpy as np

import gin.tf


@gin.configurable(
    "unified_scores",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_unified_scores(ground_truth_data, representation_function,
                           random_state,
                           artifact_dir=None,
                           num_train=gin.REQUIRED,
                           num_test=gin.REQUIRED,
                           matrix_fns=gin.REQUIRED,
                           batch_size=16):
  """Computes the unified disentanglement scores.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    matrix_fns: List of functions to relate factors of variations and codes.
    batch_size: Batch size for sampling.

  Returns:
    Unified scores.
  """
  logging.info("Generating training set.")
  # mus_train are of shape [num_codes, num_train], while ys_train are of shape
  # [num_factors, num_train].
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train
  mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test,
      random_state, batch_size)

  return unified_scores(mus_train, ys_train, mus_test, ys_test, matrix_fns,
                        artifact_dir, ground_truth_data.factor_names)


@gin.configurable(
    "unified_score_validation",
    blacklist=["observations", "labels", "representation_function"])
def compute_unified_score_on_fixed_data(
    observations, labels, representation_function,
    train_percentage=gin.REQUIRED, matrix_fns=gin.REQUIRED, batch_size=100):
  """Computes the unified scores on the fixed set of observations and labels.

  Args:
    observations: Observations on which to compute the score. Observations have
      shape (num_observations, 64, 64, num_channels).
    labels: Observed factors of variations.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    train_percentage: Percentage of observations used for training.
    matrix_fns: List of functions to relate factors of variations and codes.
    batch_size: Batch size used to compute the representation.

  Returns:
    Unified scores.
  """
  mus = utils.obtain_representation(observations, representation_function,
                                    batch_size)
  assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
  assert mus.shape[1] == observations.shape[0], "Wrong representation shape."

  mus_train, mus_test = utils.split_train_test(
      mus,
      train_percentage)
  ys_train, ys_test = utils.split_train_test(
      labels,
      train_percentage)
  return unified_scores(mus_train, ys_train, mus_test, ys_test, matrix_fns)


def unified_scores(mus_train, ys_train, mus_test, ys_test, matrix_fns,
                   artifact_dir=None, factor_names=None):
  """Computes unified scores."""

  scores = {}
  kws = {}
  for matrix_fn in matrix_fns:
    # Matrix should have shape [num_codes, num_factors].
    matrix = matrix_fn(mus_train, ys_train, mus_test, ys_test)
    matrix_name = matrix_fn.__name__
    if artifact_dir is not None:
      visualize_scores.heat_square(matrix.copy(), artifact_dir, matrix_name,
                                   "Latent codes", "Factors of Variation",
                                   factor_names=factor_names)
      visualize_scores.plot_recovery_vs_independent(matrix.copy().T,
                                                    artifact_dir,
                                                    matrix_name+"_pr")
      merge_points = dendrogram.dendrogram_plot(matrix.copy().T, os.path.join(
          artifact_dir, matrix_name+"_dendrogram"), factor_names)
      kws[matrix_name] = merge_points
    results_dict = pr_curves_values(matrix)
    if matrix_name in kws:
      kws[matrix_name].update(results_dict)
    else:
      kws[matrix_name] = results_dict
    for aggregation_fn in AGGREGATION_FN:
      results_dict = aggregation_fn(matrix, ys_train)
      kws[matrix_name].update(results_dict)
  scores = results.namespaced_dict(scores, **kws)
  return scores


def pr_curves_values(matrix):
  """Computes area of precision curve and max recall."""
  scores = {}
  thresholds = np.sort(matrix.flatten())[::-1]
  precisions = [visualize_scores.precision(matrix, x) for x in thresholds]
  area = 0.
  for i in range(len(precisions)-1):
    area += (
        precisions[i] + precisions[i+1])*(thresholds[i] - thresholds[i+1])*0.5
  scores["area_precision"] = area
  scores["max_precision"] = max(precisions)
  return scores


@gin.configurable(
    "importance_gbt_matrix",
    blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def importance_gbt_matrix(mus_train, ys_train, mus_test, ys_test):
  """Computes the importance matrix of the DCI Disentanglement score.

  The importance matrix is based on the importance of each code to predict a
  factor of variation with GBT.

  Args:
    mus_train: Batch of learned representations to be used for training.
    ys_train: Observed factors of variation corresponding to the representations
      in mus_train.
    mus_test: Batch of learned representations to be used for testing.
    ys_test: Observed factors of variation corresponding to the representations
    in mus_test.

  Returns:
    Importance matrix as computed for the DCI Disentanglement score.
  """
  matrix_importance_gbt, _, _ = dci.compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
  return matrix_importance_gbt


@gin.configurable(
    "mutual_information_matrix",
    blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def mutual_information_matrix(mus_train, ys_train, mus_test, ys_test):
  """Computes the mutual information matrix between codes and factors.

  The mutual information matrix is used to compute the MIG and Modularity
  scores.

  Args:
    mus_train: Batch of learned representations to be used for training.
    ys_train: Observed factors of variation corresponding to the representations
      in mus_train.
    mus_test: Unused.
    ys_test: Unused.

  Returns:
    Mutual information matrix as computed for the MIG and Modularity scores.
  """
  del mus_test, ys_test
  discretized_mus = utils.make_discretizer(mus_train)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  return m


@gin.configurable(
    "accuracy_svm_matrix",
    blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def accuracy_svm_matrix(mus_train, ys_train, mus_test, ys_test):
  """Prediction accuracy of a SVM predicting a factor from a single code.

  The matrix of accuracies is used to compute the SAP score.

  Args:
    mus_train: Batch of learned representations to be used for training.
    ys_train: Observed factors of variation corresponding to the representations
      in mus_train.
    mus_test: Batch of learned representations to be used for testing.
    ys_test: Observed factors of variation corresponding to the representations
    in mus_test.

  Returns:
    Accuracy matrix as computed for the SAP score.
  """
  return sap_score.compute_score_matrix(
      mus_train, ys_train, mus_test, ys_test, continuous_factors=False)


def aggregation_dci(matrix, ys):
  """Aggregation function of the DCI Disentanglement."""
  del ys
  score = {}
  score["dci_disentanglement"] = dci.disentanglement(matrix)
  score["dci_completeness"] = dci.completeness(matrix)
  score["dci"] = dci.disentanglement(matrix)
  disentanglement_per_code = dci.disentanglement_per_code(matrix)
  completeness_per_factor = dci.completeness_per_factor(matrix)
  assert len(disentanglement_per_code) == matrix.shape[0], "Wrong length."
  assert len(completeness_per_factor) == matrix.shape[1], "Wrong length."
  for i in range(len(disentanglement_per_code)):
    score["dci_disentanglement.code_{}".format(i)] = disentanglement_per_code[i]
  for i in range(len(completeness_per_factor)):
    score["dci_completeness.code_{}".format(i)] = completeness_per_factor[i]
  return score


def aggregation_mig(m, ys_train):
  """Aggregation function of the MIG."""
  score = {}
  entropy = utils.discrete_entropy(ys_train)
  sorted_m = np.sort(m, axis=0)[::-1]
  mig_per_factor = np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
  score["mig"] = np.mean(mig_per_factor)
  assert len(mig_per_factor) == m.shape[1], "Wrong length."
  for i in range(len(mig_per_factor)):
    score["mig.factor_{}".format(i)] = mig_per_factor[i]
  return score


def aggregation_sap(matrix, ys):
  """Aggregation function of the SAP score."""
  del ys
  score = {}
  score["sap"] = sap_score.compute_avg_diff_top_two(matrix)
  sap_per_factor = sap_compute_diff_top_two(matrix)
  assert len(sap_per_factor) == matrix.shape[1], "Wrong length."
  for i in range(len(sap_per_factor)):
    score["sap.factor_{}".format(i)] = sap_per_factor[i]
  return score


def sap_compute_diff_top_two(matrix):
  sorted_matrix = np.sort(matrix, axis=0)
  return sorted_matrix[-1, :] - sorted_matrix[-2, :]


def aggregation_modularity(matrix, ys):
  """Aggregation function of the modularity score."""
  del ys
  score = {}
  score["modularity"] = modularity_explicitness.modularity(matrix)
  modularity_per_code = compute_modularity_per_code(matrix)
  assert len(modularity_per_code) == matrix.shape[0], "Wrong length."
  for i in range(len(modularity_per_code)):
    score["modularity.code_{}".format(i)] = modularity_per_code[i]
  return score


def compute_modularity_per_code(mutual_information):
  """Computes the modularity from mutual information."""
  # Mutual information has shape [num_codes, num_factors].
  squared_mi = np.square(mutual_information)
  max_squared_mi = np.max(squared_mi, axis=1)
  numerator = np.sum(squared_mi, axis=1) - max_squared_mi
  denominator = max_squared_mi * (squared_mi.shape[1] -1.)
  delta = numerator / denominator
  modularity_score = 1. - delta
  index = (max_squared_mi == 0.)
  modularity_score[index] = 0.
  return modularity_score

AGGREGATION_FN = [aggregation_dci, aggregation_mig, aggregation_sap,
                  aggregation_modularity]
