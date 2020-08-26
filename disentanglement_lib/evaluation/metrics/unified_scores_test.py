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

"""Tests for unified_scores.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import unified_scores
import numpy as np

import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return target


class UnifiedScoreTest(absltest.TestCase):

  def test_metric_mig(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = unified_scores.compute_unified_scores(
        ground_truth_data, representation_function, random_state, None, 10000,
        100, matrix_fns=[unified_scores.mutual_information_matrix])
    self.assertBetween(
        scores["mutual_information_matrix.mig"], 0.9, 1.0)
    self.assertBetween(
        scores["mutual_information_matrix.modularity"], 0.9, 1.0)

  def test_metric_dci(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = unified_scores.compute_unified_scores(
        ground_truth_data, representation_function, random_state, None, 10000,
        100, matrix_fns=[unified_scores.importance_gbt_matrix])
    self.assertBetween(
        scores["importance_gbt_matrix.dci_disentanglement"], 0.9, 1.0)

  def test_duplicated_latent_space_mig(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = unified_scores.compute_unified_scores(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000, matrix_fns=[unified_scores.mutual_information_matrix])
    self.assertBetween(
        scores["mutual_information_matrix.mig"], 0.0, 0.2)
    self.assertBetween(
        scores["mutual_information_matrix.modularity"], 0.9, 1.0)

  def test_duplicated_latent_space_dci(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = unified_scores.compute_unified_scores(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000, matrix_fns=[unified_scores.importance_gbt_matrix])
    self.assertBetween(
        scores["importance_gbt_matrix.dci_disentanglement"], 0.9, 1.0)

  def test_duplicated_latent_space_sap(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = unified_scores.compute_unified_scores(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000, matrix_fns=[unified_scores.accuracy_svm_matrix])
    self.assertBetween(scores["accuracy_svm_matrix.sap"], 0.0, 0.2)


if __name__ == "__main__":
  absltest.main()
