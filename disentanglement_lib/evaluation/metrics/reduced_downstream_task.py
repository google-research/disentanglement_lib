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

"""Reduced downstream classification task.

Test downstream performance after removing the k most predictive features for
each factor of variation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation.metrics import dci
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
  "reduced_downstream_task",
  blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_reduced_downstream_task(ground_truth_data,
                                    representation_function,
                                    random_state,
                                    num_factors_to_remove=gin.REQUIRED,
                                    num_train=gin.REQUIRED,
                                    num_test=gin.REQUIRED,
                                    batch_size=16):
  """Computes loss of a reduced downstream task.

  Measure the information leakage in each latent component after removing the
  k ("factors_to_remove") most informative features for the prediction task.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_factors_to_remove: number of factors to remove from the latent
    representation.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  scores = {}
  # Loop on different sizes of the training 'batch', as specified with gin
  for train_size in num_train:
    mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, train_size, random_state,
      batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test, random_state,
      batch_size)
    # Compute the reduced representation and test it for each factor of
    # variation
    for factor_of_interest in range(ground_truth_data.num_factors):
      # Copy the training data and eliminate the k most informative factors
      reduced_mus_train = mus_train.copy()
      reduced_mus_test = mus_test.copy()
      for _ in range(num_factors_to_remove):
        reduced_mus_train, reduced_mus_test =\
          compute_reduced_representation(reduced_mus_train, ys_train,
                                         reduced_mus_test, ys_test,
                                         factor_of_interest)
      predictor_model = utils.make_predictor_fn()
      train_acc, test_acc = compute_predictive_accuracy(
        np.transpose(reduced_mus_train), ys_train,
        np.transpose(reduced_mus_test),
        ys_test, predictor_model)
      # Save the scores (accuracies) in the score dictionary
      size_string = str(train_size)
      scores[size_string +
             ":mean_train_accuracy_factor_of_interest_{}".format(
               factor_of_interest)] = np.mean(train_acc)
      scores[size_string +
             ":mean_test_accuracy_factor_of_interest_{}".format(
               factor_of_interest)] = np.mean(test_acc)
      for i in range(len(train_acc)):
        scores[size_string +
               ":train_accuracy_{}_factor_of_interest_{}".format(
                 i, factor_of_interest)] = train_acc[i]
        scores[size_string + ":test_accuracy_{}_factor_of_interest_{}".format(
                 i, factor_of_interest)] = test_acc[i]
  return scores


@gin.configurable("reduced_representation")
def compute_reduced_representation(mus_train, ys_train, mus_test, ys_test,
                                   factor_of_interest,
                                   correlation_measure=gin.REQUIRED):
  """Compute a reduced representation of the data.

  The most informative factor with respect to the labels is deleted.

  Args:
    mus_train: latent means of the training batch.
    ys_train: labels of the training batch.
    mus_test: latent means of the test batch.
    ys_test: labels of the test batch.
    factor_of_interest: index of the factor of interest.
    correlation_measure: measure of correlation.
  """
  importance_matrix = correlation_measure(mus_train, ys_train, mus_test,
                                          ys_test)
  factor_of_interest_importance = importance_matrix[:, factor_of_interest]
  factor_to_remove_index = np.argmax(factor_of_interest_importance)
  # Remove the factor of variation above from the representation
  reduced_representation_train = np.delete(mus_train.copy(),
                                           factor_to_remove_index,
                                           axis=0)
  reduced_representation_test = np.delete(mus_test.copy(),
                                          factor_to_remove_index,
                                          axis=0)
  return reduced_representation_train, reduced_representation_test


@gin.configurable("factorwise_dci",
                  blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def compute_factorwise_dci(mus_train, ys_train, mus_test, ys_test):
  """Compute the DCI importance matrix of the attributes.

  Args:
    mus_train: latent means of the training batch.
    ys_train: labels of the training batch.
    mus_test: latent means of the test batch.
    ys_test: labels of the test batch.
  """
  importance_matrix, _, _ = dci.compute_importance_gbt(
    mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  return importance_matrix


def compute_predictive_accuracy(x_train, y_train, x_test, y_test, predictor_fn):
  """Compute average predictive accuracy for train and test set.

  Args:
    x_train: data x of the training batch.
    y_train: labels y of the training batch.
    x_test: data x of the test batch.
    y_test: labels y of the test batch.
    predictor_fn: function that is used to fit and predict the labels.
  """
  num_factors = y_train.shape[0]
  train_acc = []
  test_acc = []
  # Loop on the generative factors to predict
  for i in range(num_factors):
    model = predictor_fn()
    model.fit(x_train, y_train[i, :])
    train_acc.append(np.mean(model.predict(x_train) == y_train[i, :]))
    test_acc.append(np.mean(model.predict(x_test) == y_test[i, :]))
  return train_acc, test_acc
