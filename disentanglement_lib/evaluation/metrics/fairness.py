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

"""Fairness metric task.

In this task, we compute the unfairness of a classifier trained on a given
representation. This procedure is described in Section 4 of the paper "On the
Fairness of Disentangled Representations", Francesco Locatello, Gabriele Abbati,
Tom Rainforth, Stefan Bauer, Bernhard Schoelkopf, Olivier Bachem, NeurIPS 2019,
https://arxiv.org/abs/1905.13662.
The unfairness of a classifier is computed as the total variation distance
between the distribution of predictions of the target varible for different
interventions on the sensitive variable. This approach is described in Section
4 of the paper.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
from six.moves import range
import gin.tf


@gin.configurable(
    "fairness",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_fairness(ground_truth_data,
                     representation_function,
                     random_state,
                     artifact_dir=None,
                     num_train=gin.REQUIRED,
                     num_test_points_per_class=gin.REQUIRED,
                     batch_size=16):
  """Computes unfairness scores.

  We first compute either the mean or maximum total variation for a given
  sensitive and target variable. Then, we either average or take the maximum
  with respect to target and sensitive variable. For convenience, we compute and
  save all combinations. The score used in Section 4 of the paper is here called
  mean_fairness:mean_pred:mean_sens.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test_points_per_class: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  del artifact_dir
  factor_counts = ground_truth_data.factors_num_values
  num_factors = len(factor_counts)

  scores = {}
  # Training a predictive model.
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train, random_state,
      batch_size)
  predictor_model_fn = utils.make_predictor_fn()

  # For each factor train a single predictive model.
  mean_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  max_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  for i in range(num_factors):
    model = predictor_model_fn()
    model.fit(np.transpose(mus_train), ys_train[i, :])

    for j in range(num_factors):
      if i == j:
        continue
      # Sample a random set of factors once.
      original_factors = ground_truth_data.sample_factors(
          num_test_points_per_class, random_state)
      counts = np.zeros((factor_counts[i], factor_counts[j]), dtype=np.int64)
      for c in range(factor_counts[j]):
        # Intervene on the sensitive attribute.
        intervened_factors = np.copy(original_factors)
        intervened_factors[:, j] = c
        # Obtain the batched observations.
        observations = ground_truth_data.sample_observations_from_factors(
            intervened_factors, random_state)
        representations = utils.obtain_representation(observations,
                                                      representation_function,
                                                      batch_size)
        # Get the predictions.
        predictions = model.predict(np.transpose(representations))
        # Update the counts.
        counts[:, c] = np.bincount(predictions, minlength=factor_counts[i])
      mean_fairness[i, j], max_fairness[i, j] = inter_group_fairness(counts)

  # Report the scores.
  scores.update(compute_scores_dict(mean_fairness, "mean_fairness"))
  scores.update(compute_scores_dict(max_fairness, "max_fairness"))
  return scores


def compute_scores_dict(metric, prefix):
  """Computes scores for combinations of predictive and sensitive factors.

  Either average or take the maximum with respect to target and sensitive
  variable for all combinations of predictive and sensitive factors.

  Args:
    metric: Matrix of shape [num_factors, num_factors] with fairness scores.
    prefix: Prefix for the matrix in the returned dictionary.

  Returns:
    Dictionary containing all combinations of predictive and sensitive factors.
  """
  result = {}
  # Report min and max scores for each predictive and sensitive factor.
  for i in range(metric.shape[0]):
    for j in range(metric.shape[1]):
      if i != j:
        result["{}:pred{}:sens{}".format(prefix, i, j)] = metric[i, j]

  # Compute mean and max values across rows.
  rows_means = []
  rows_maxs = []
  for i in range(metric.shape[0]):
    relevant_scores = [metric[i, j] for j in range(metric.shape[1]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:pred{}:mean_sens".format(prefix, i)] = mean_score
    result["{}:pred{}:max_sens".format(prefix, i)] = max_score
    rows_means.append(mean_score)
    rows_maxs.append(max_score)

  # Compute mean and max values across rows.
  column_means = []
  column_maxs = []
  for j in range(metric.shape[1]):
    relevant_scores = [metric[i, j] for i in range(metric.shape[0]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:sens{}:mean_pred".format(prefix, j)] = mean_score
    result["{}:sens{}:max_pred".format(prefix, j)] = max_score
    column_means.append(mean_score)
    column_maxs.append(max_score)

  # Compute all combinations of scores.
  result["{}:mean_sens:mean_pred".format(prefix)] = np.mean(column_means)
  result["{}:mean_sens:max_pred".format(prefix)] = np.mean(column_maxs)
  result["{}:max_sens:mean_pred".format(prefix)] = np.amax(column_means)
  result["{}:max_sens:max_pred".format(prefix)] = np.amax(column_maxs)
  result["{}:mean_pred:mean_sens".format(prefix)] = np.mean(rows_means)
  result["{}:mean_pred:max_sens".format(prefix)] = np.mean(rows_maxs)
  result["{}:max_pred:mean_sens".format(prefix)] = np.amax(rows_means)
  result["{}:max_pred:max_sens".format(prefix)] = np.amax(rows_maxs)

  return result


def inter_group_fairness(counts):
  """Computes the inter group fairness for predictions based on the TV distance.

  Args:
   counts: Numpy array with counts of predictions where rows correspond to
     predicted classes and columns to sensitive classes.

  Returns:
    Mean and maximum total variation distance of a sensitive class to the
      global average.
  """
  # Compute the distribution of predictions across all sensitive classes.
  overall_distribution = np.sum(counts, axis=1, dtype=np.float32)
  overall_distribution /= overall_distribution.sum()

  # Compute the distribution for each sensitive class.
  normalized_counts = np.array(counts, dtype=np.float32)
  counts_per_class = np.sum(counts, axis=0)
  normalized_counts /= np.expand_dims(counts_per_class, 0)

  # Compute the differences and sum up for each sensitive class.
  differences = normalized_counts - np.expand_dims(overall_distribution, 1)
  total_variation_distances = np.sum(np.abs(differences), 0) / 2.

  mean = (total_variation_distances * counts_per_class)
  mean /= counts_per_class.sum()

  return np.sum(mean), np.amax(total_variation_distances)
