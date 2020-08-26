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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn import ensemble
import gin.tf


@gin.configurable(
    "dci",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_dci(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=16):
  """Computes the DCI scores according to Sec 2.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with average disentanglement score, completeness and
      informativeness (train and test).
  """
  del artifact_dir
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
  scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
  return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores


@gin.configurable(
    "dci_validation",
    blacklist=["observations", "labels", "representation_function"])
def compute_dci_on_fixed_data(observations, labels, representation_function,
                              train_percentage=gin.REQUIRED, batch_size=100):
  """Computes the DCI scores on the fixed set of observations and labels.

  Args:
    observations: Observations on which to compute the score. Observations have
      shape (num_observations, 64, 64, num_channels).
    labels: Observed factors of variations.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    train_percentage: Percentage of observations used for training.
    batch_size: Batch size used to compute the representation.

  Returns:
    DCI score.
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
  return _compute_dci(mus_train, ys_train, mus_test, ys_test)


def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = ensemble.GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
