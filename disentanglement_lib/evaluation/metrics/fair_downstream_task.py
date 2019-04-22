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

"""Downstream classification task.

The code selects the features that are the most correlated in the latent space
and ignores them before testing the classifier on the new fair space.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation.metrics import fairness_utils
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
  "fair_downstream_task",
  blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_downstream_task(ground_truth_data,
                            representation_function,
                            random_state,
                            num_train=gin.REQUIRED,
                            num_test=gin.REQUIRED,
                            batch_size=16):
  """Computes loss of downstream task.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  scores = {}
  for train_size in num_train:
    mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, train_size, random_state,
      batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test, random_state,
      batch_size)
    for sensitive_factor_index in range(ground_truth_data.num_factors):
      fair_mus_train, fair_mus_test =\
        fairness_utils.compute_fair_representation(mus_train, ys_train,
                                                   mus_test, ys_test,
                                                   sensitive_factor_index)
      predictor_model = utils.make_predictor_fn()

      train_err, test_err = _compute_loss(
        np.transpose(fair_mus_train), ys_train, np.transpose(fair_mus_test),
        ys_test, predictor_model)
      size_string = str(train_size)
      scores[size_string +
             ":mean_train_accuracy_removed_{}".format(
               sensitive_factor_index)] = np.mean(train_err)
      scores[size_string +
             ":mean_test_accuracy_removed_{}".format(
               sensitive_factor_index)] = np.mean(test_err)
      scores[size_string +
             ":min_train_accuracy_removed_{}".format(
               sensitive_factor_index)] = np.min(train_err)
      scores[size_string + ":min_test_accuracy_removed_{}".format(
               sensitive_factor_index)] = np.min(test_err)
      for i in range(len(train_err)):
        scores[size_string +
               ":train_accuracy_factor_{}_removed_{}".format(
                 i, sensitive_factor_index)] = train_err[i]
        scores[size_string + ":test_accuracy_factor_{}_removed_{}".format(
                 i, sensitive_factor_index)] = test_err[i]

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
