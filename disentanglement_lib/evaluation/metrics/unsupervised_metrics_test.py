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

"""Tests for unsupervised_metrics.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.evaluation.metrics import unsupervised_metrics
import numpy as np
import scipy


class UnsupervisedMetricsTest(absltest.TestCase):

  def test_gaussian_total_correlation_zero(self):
    score = unsupervised_metrics.gaussian_total_correlation(
        np.diag(np.ones(5, dtype=np.float64)))
    self.assertBetween(score, -0.01, 0.01)

  def test_gaussian_total_correlation_same(self):
    """Check that the results of the both functions are the same."""
    cov = np.array([[1, 0.9], [0.9, 1.0]], dtype=np.float32)
    means = np.array([0.0, 0.0], dtype=np.float32)
    cov_central = np.diag(np.diag(cov))
    shouldbe = unsupervised_metrics.kl_gaussians_numerically_unstable(
        means, cov, means, cov_central, 2)
    score = unsupervised_metrics.gaussian_total_correlation(cov)
    self.assertBetween(score, shouldbe - 0.01, shouldbe + 0.01)

  def test_gaussian_wasserstein_correlation_zero(self):
    score = unsupervised_metrics.gaussian_wasserstein_correlation(
        np.diag(np.ones(5, dtype=np.float64)))
    self.assertBetween(score, -0.01, 0.01)

  def test_gaussian_wasserstein_correlation_same(self):
    cov = np.array([[1, 0.9], [0.9, 1.0]], dtype=np.float32)
    score = unsupervised_metrics.gaussian_wasserstein_correlation(cov)
    cov_only_diagonal = np.diag(np.diag(cov))
    sqrtm = scipy.linalg.sqrtm(np.matmul(cov, cov_only_diagonal))
    shouldbe = np.trace(cov + cov_only_diagonal - 2 * sqrtm)
    self.assertBetween(score, shouldbe - 0.01, shouldbe + 0.01)


if __name__ == "__main__":
  absltest.main()
