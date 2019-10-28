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

"""Tests for dci_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import dci
import numpy as np
from six.moves import range


class DisentanglementTest(absltest.TestCase):

  def test_diagonal(self):
    importance_matrix = np.diag(5.*np.ones(5))
    result = dci.disentanglement(importance_matrix)
    np.testing.assert_allclose(result, 1.0)

  def test_diagonal_empty_codes(self):
    importance_matrix = np.array([[1., 0.,], [0., 1.], [0., 0.]])
    result = dci.disentanglement(importance_matrix)
    np.testing.assert_allclose(result, 1.0)

  def test_zero(self):
    importance_matrix = np.zeros(shape=[10, 10], dtype=np.float64)
    result = dci.disentanglement(importance_matrix)
    np.testing.assert_allclose(result, .0)

  def test_redundant_codes(self):
    importance_matrix = np.diag(5.*np.ones(5))
    importance_matrix = np.vstack([importance_matrix, importance_matrix])
    result = dci.disentanglement(importance_matrix)
    np.testing.assert_allclose(result, 1.)

  def test_missed_factors(self):
    importance_matrix = np.diag(5.*np.ones(5))
    result = dci.disentanglement(importance_matrix[:2, :])
    np.testing.assert_allclose(result, 1.0)

  def test_one_code_two_factors(self):
    importance_matrix = np.diag(5.*np.ones(5))
    importance_matrix = np.hstack([importance_matrix, importance_matrix])
    result = dci.disentanglement(importance_matrix)
    np.testing.assert_allclose(result, 1. - np.log(2)/np.log(10))


class CompletenessTest(absltest.TestCase):

  def test_diagonal(self):
    importance_matrix = np.diag(5.*np.ones(5))
    result = dci.completeness(importance_matrix)
    np.testing.assert_allclose(result, 1.0)

  def test_diagonal_empty_codes(self):
    importance_matrix = np.array([[1., 0.,], [0., 1.], [0., 0.]])
    result = dci.completeness(importance_matrix)
    np.testing.assert_allclose(result, 1.0)

  def test_zero(self):
    importance_matrix = np.zeros(shape=[10, 10], dtype=np.float64)
    result = dci.completeness(importance_matrix)
    np.testing.assert_allclose(result, .0, atol=1e-7)

  def test_redundant_codes(self):
    importance_matrix = np.diag(5.*np.ones(5))
    importance_matrix = np.vstack([importance_matrix, importance_matrix])
    result = dci.completeness(importance_matrix)
    np.testing.assert_allclose(result, 1. - np.log(2)/np.log(10))

  def test_missed_factors(self):
    importance_matrix = np.diag(5.*np.ones(5))
    result = dci.completeness(importance_matrix[:2, :])
    np.testing.assert_allclose(result, 1.0)

  def test_one_code_two_factors(self):
    importance_matrix = np.diag(5.*np.ones(5))
    importance_matrix = np.hstack([importance_matrix, importance_matrix])
    result = dci.completeness(importance_matrix)
    np.testing.assert_allclose(result, 1.)


class DCITest(absltest.TestCase):

  def test_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.array(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = dci.compute_dci(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000)
    self.assertBetween(scores["disentanglement"], 0.9, 1.0)
    self.assertBetween(scores["completeness"], 0.9, 1.0)

  def test_bad_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    random_state_rep = np.random.RandomState(0)
    # The representation which randomly permutes the factors, should have equal
    # non-zero importance which should give a low modularity score.
    def representation_function(x):
      code = np.array(x, dtype=np.float64)
      for i in range(code.shape[0]):
        code[i, :] = random_state_rep.permutation(code[i, :])
      return code
    random_state = np.random.RandomState(0)
    scores = dci.compute_dci(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000)
    self.assertBetween(scores["disentanglement"], 0.0, 0.2)
    self.assertBetween(scores["completeness"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = dci.compute_dci(
        ground_truth_data, representation_function, random_state, None, 1000,
        1000)
    self.assertBetween(scores["disentanglement"], 0.9, 1.0)
    target = 1. - np.log(2)/np.log(10)
    self.assertBetween(scores["completeness"], target-.1, target+.1)

if __name__ == "__main__":
  absltest.main()
