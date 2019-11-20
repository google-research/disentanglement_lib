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

"""Tests for udr.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.udr.metrics import udr
import numpy as np


class UdrTest(absltest.TestCase):

  def test_metric_spearman(self):
    ground_truth_data = dummy_data.DummyData()
    random_state = np.random.RandomState(0)
    num_factors = ground_truth_data.num_factors
    batch_size = 10
    num_data_points = 1000

    permutation = np.random.permutation(num_factors)
    sign_inverse = np.random.choice(num_factors, int(num_factors / 2))

    def rep_fn1(data):
      return (np.reshape(data, (batch_size, -1))[:, :num_factors],
              np.ones(num_factors))

    # Should be invariant to permutation and sign inverse.
    def rep_fn2(data):
      raw_representation = np.reshape(data, (batch_size, -1))[:, :num_factors]
      perm_rep = raw_representation[:, permutation]
      perm_rep[:, sign_inverse] = -1.0 * perm_rep[:, sign_inverse]
      return perm_rep, np.ones(num_factors)

    scores = udr.compute_udr_sklearn(
        ground_truth_data, [rep_fn1, rep_fn2],
        random_state,
        batch_size,
        num_data_points,
        correlation_matrix="spearman")
    self.assertBetween(scores["model_scores"][0], 0.8, 1.0)
    self.assertBetween(scores["model_scores"][1], 0.8, 1.0)

  def test_metric_lasso(self):
    ground_truth_data = dummy_data.DummyData()
    random_state = np.random.RandomState(0)
    num_factors = ground_truth_data.num_factors
    batch_size = 10
    num_data_points = 1000

    permutation = np.random.permutation(num_factors)
    sign_inverse = np.random.choice(num_factors, int(num_factors / 2))

    def rep_fn1(data):
      return (np.reshape(data, (batch_size, -1))[:, :num_factors],
              np.ones(num_factors))

    # Should be invariant to permutation and sign inverse.
    def rep_fn2(data):
      raw_representation = np.reshape(data, (batch_size, -1))[:, :num_factors]
      perm_rep = raw_representation[:, permutation]
      perm_rep[:, sign_inverse] = -1.0 * perm_rep[:, sign_inverse]
      return perm_rep, np.ones(num_factors)

    scores = udr.compute_udr_sklearn(
        ground_truth_data, [rep_fn1, rep_fn2],
        random_state,
        batch_size,
        num_data_points,
        correlation_matrix="lasso")
    self.assertBetween(scores["model_scores"][0], 0.8, 1.0)
    self.assertBetween(scores["model_scores"][1], 0.8, 1.0)

  def test_metric_kl(self):
    ground_truth_data = dummy_data.DummyData()
    random_state = np.random.RandomState(0)
    num_factors = ground_truth_data.num_factors
    batch_size = 10
    num_data_points = 1000

    # Representation without KL Mask where only first latent is valid.
    def rep_fn(data):
      rep = np.concatenate([
          np.reshape(data, (batch_size, -1))[:, :1],
          np.random.random_sample((batch_size, num_factors - 1))
      ],
                           axis=1)
      kl_mask = np.zeros(num_factors)
      kl_mask[0] = 1.0
      return rep, kl_mask

    scores = udr.compute_udr_sklearn(
        ground_truth_data, [rep_fn, rep_fn],
        random_state,
        batch_size,
        num_data_points,
        filter_low_kl=False)
    self.assertBetween(scores["model_scores"][0], 0.0, 0.2)
    self.assertBetween(scores["model_scores"][1], 0.0, 0.2)

    scores = udr.compute_udr_sklearn(
        ground_truth_data, [rep_fn, rep_fn],
        random_state,
        batch_size,
        num_data_points,
        filter_low_kl=True)
    self.assertBetween(scores["model_scores"][0], 0.8, 1.0)
    self.assertBetween(scores["model_scores"][1], 0.8, 1.0)

  def test_relative_strength_disentanglement(self):
    corr_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    self.assertEqual(udr.relative_strength_disentanglement(corr_matrix), 1.0)

    corr_matrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    self.assertBetween(
        udr.relative_strength_disentanglement(corr_matrix), 0.6, 0.7)

    corr_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    self.assertBetween(
        udr.relative_strength_disentanglement(corr_matrix), 0.3, 0.4)

    corr_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    self.assertEqual(udr.relative_strength_disentanglement(corr_matrix), 0.0)

  def test_spearman_correlation(self):
    random_state = np.random.RandomState(0)
    vec1 = random_state.random_sample((1000, 3))
    vec2 = np.copy(vec1)
    expected_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    self.assertTrue(
        np.allclose(
            udr.spearman_correlation_conv(vec1, vec2),
            expected_matrix,
            atol=0.1))

    vec1 = random_state.random_sample((1000, 3))
    vec2 = np.copy(vec1)
    vec2[:, 1] = vec2[:, 0]

    expected_matrix = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
    self.assertTrue(
        np.allclose(
            udr.spearman_correlation_conv(vec1, vec2),
            expected_matrix,
            atol=0.1))

  def test_lasso_correlation(self):
    random_state = np.random.RandomState(0)
    vec1 = random_state.random_sample((1000, 3)) * 10.0
    vec2 = np.copy(vec1)
    expected_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    self.assertTrue(
        np.allclose(
            udr.lasso_correlation_matrix(vec1, vec2, random_state=random_state),
            expected_matrix,
            atol=0.2))

    vec1 = random_state.random_sample((1000, 3)) * 10.0
    vec2 = np.copy(vec1)
    vec2[:, 1] = vec2[:, 0]

    expected_matrix = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
    self.assertTrue(
        np.allclose(
            udr.lasso_correlation_matrix(vec1, vec2, random_state=random_state),
            expected_matrix,
            atol=0.2))


if __name__ == "__main__":
  absltest.main()
