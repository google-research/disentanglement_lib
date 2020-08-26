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

"""Downstream out-of-distribution classification task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
    "strong_downstream_task",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_strong_downstream_task(ground_truth_data,
                                   representation_function,
                                   random_state,
                                   artifact_dir=None,
                                   num_train=gin.REQUIRED,
                                   num_test=gin.REQUIRED,
                                   n_experiment=gin.REQUIRED):
  """Computes loss of downstream task.

  This task is about strong generalization under covariate shifts. We first
  perform an intervention fixing a value for a factor in the whole training set.
  Then, we train a GBT classifier, and at test time, we consider all other
  values for that factor. We repeat the experiment n_experiment times, to ensure
  robustness.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    n_experiment: Number of repetitions of the experiment.

  Returns:
    Dictionary with scores.
  """
  del artifact_dir
  scores = {}
  for train_size in num_train:
    # sample factors
    factors_train = ground_truth_data.sample_factors(train_size, random_state)
    factors_test = ground_truth_data.sample_factors(num_test, random_state)
    # obtain_observations without intervention
    x_train = ground_truth_data.sample_observations_from_factors(
        factors_train, random_state)
    x_test = ground_truth_data.sample_observations_from_factors(
        factors_test, random_state)
    mus_train = representation_function(x_train)
    mus_test = representation_function(x_test)
    # train predictor on data without interbention
    predictor_model = utils.make_predictor_fn()
    y_train = np.transpose(factors_train)
    y_test = np.transpose(factors_test)
    train_err, test_err = _compute_loss(
        mus_train, y_train, mus_test,
        y_test, predictor_model)

    # train predictor on data with interventions
    train_err_int, test_err_int = _compute_loss_intervene(
        factors_train, factors_test, predictor_model, ground_truth_data,
        representation_function, random_state, n_experiment)

    size_string = str(train_size)
    scores[size_string +
           ":mean_train_accuracy"] = np.mean(train_err)
    scores[size_string +
           ":mean_test_accuracy"] = np.mean(test_err)
    scores[size_string +
           ":mean_strong_train_accuracy"] = np.mean(train_err_int)
    scores[size_string +
           ":mean_strong_test_accuracy"] = np.mean(test_err_int)
    scores[size_string + ":strong_generalization_gap"] = 1. - (
        scores[size_string + ":mean_strong_test_accuracy"] /
        scores[size_string + ":mean_test_accuracy"])
  return scores


def _compute_loss(x_train, y_train, x_test, y_test, predictor_fn):
  """Compute average accuracy for train and test set."""
  num_factors = y_train.shape[0]
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = predictor_fn()
    model.fit(x_train, y_train[i, :])
    train_loss.append(np.mean(model.predict(x_train) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
  return train_loss, test_loss


def _compute_loss_intervene(factors_train, factors_test, predictor_fn,
                            ground_truth_data, representation_function,
                            random_state, n_experiment=10):
  """Compute average accuracy for train and test set."""
  num_factors = factors_train.shape[1]
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    for _ in range(n_experiment):
      factors_train_int, factors_test_int, _, _ = intervene(
          factors_train.copy(), factors_test.copy(), i, num_factors,
          ground_truth_data)

      obs_train_int = ground_truth_data.sample_observations_from_factors(
          factors_train_int, random_state)
      obs_test_int = ground_truth_data.sample_observations_from_factors(
          factors_test_int, random_state)

      x_train_int = representation_function(obs_train_int)
      x_test_int = representation_function(obs_test_int)
      # train predictor on data without intervention
      y_train_int = np.transpose(factors_train_int)
      y_test_int = np.transpose(factors_test_int)
      model = predictor_fn()
      model.fit(x_train_int, y_train_int[i, :])
      train_loss.append(
          np.mean(model.predict(x_train_int) == y_train_int[i, :]))
      test_loss.append(np.mean(model.predict(x_test_int) == y_test_int[i, :]))
  return train_loss, test_loss


def intervene(y_train, y_test, target_y, num_factors, ground_truth_data):
  """Make random intervention on training data and remove it from test."""
  # sample coordinate to intervene on
  factor_list_to_interv = [j for j in range(num_factors) if j != target_y]
  # pick a factor to intervene on
  interv_factor = np.random.choice(factor_list_to_interv)
  # get the factor range
  all_val_factor = list(range(
      ground_truth_data.factors_num_values[interv_factor]))
  factor_interv_train = np.random.choice(all_val_factor)
  all_val_factor.remove(factor_interv_train)
  y_train[:, interv_factor] = factor_interv_train

  factor_interv_test = np.random.choice(all_val_factor,
                                        size=y_test.shape[0])
  y_test[:, interv_factor] = factor_interv_test
  return y_train, y_test, interv_factor, factor_interv_train
