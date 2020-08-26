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

"""Tests for vae.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
from disentanglement_lib.methods.unsupervised import vae
import numpy as np
import tensorflow.compat.v1 as tf
import gin.tf.external_configurables  # pylint: disable=unused-import


def _make_symmetric_psd(matrix):
  return 0.5 * (matrix + matrix.T) + np.diag(np.ones(10)) * 10.


class VaeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((np.zeros([10, 10]), np.zeros([10, 10]), 0., 0.01),
                            (np.ones([10, 10]), np.zeros([10, 10]), 5., 5.01),
                            (np.ones([10, 10]), np.ones([10, 10]), 8.58, 8.6))
  def test_compute_gaussian_kl(self, mean, logvar, target_low, target_high):
    mean_tf = tf.convert_to_tensor(mean, dtype=np.float32)
    logvar_tf = tf.convert_to_tensor(logvar, dtype=np.float32)
    with self.test_session() as sess:
      test_value = sess.run(vae.compute_gaussian_kl(mean_tf, logvar_tf))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((0, 0., 0.01), (10, 10., 10.01),
                            (100, 100., 100.01), (101, 100., 100.01))
  def test_anneal(self, step, target_low, target_high):
    c_max = 100.
    iteration_threshold = 100
    with self.test_session() as sess:
      test_value = sess.run(vae.anneal(c_max, step, iteration_threshold))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters(
      (True, 0., 1.), (True, 0., 4.), (True, 1., 1.),
      (False, np.zeros(10), np.ones([10, 10])),
      (False, np.zeros(10), _make_symmetric_psd(np.random.random((10, 10)))))
  def test_compute_covariance_z_mean(self, diagonal, mean, cov):
    if diagonal:
      samples = tf.random.normal(
          shape=(100000, 10), mean=mean, stddev=tf.math.sqrt(cov))
      cov = np.diag(np.ones([10])) * cov
    else:
      samples = tf.constant(
          np.random.multivariate_normal(mean, cov, size=(1000000)))
    with self.test_session() as sess:
      test_value = sess.run(vae.compute_covariance_z_mean(samples))
      self.assertBetween(np.sum((test_value - cov)**2), 0., 0.1)

  @parameterized.parameters(
      (np.ones([10, 10]), 90., 90.1), (np.zeros([10, 10]), 10., 10.1),
      (np.diag(np.ones(10)), 0., 0.1), (2. * np.diag(np.ones(10)), 10., 10.1))
  def test_regularize_diag_off_diag_dip(self, matrix, target_low, target_high):
    matrix_tf = tf.convert_to_tensor(matrix, dtype=np.float32)
    with self.test_session() as sess:
      test_value = sess.run(vae.regularize_diag_off_diag_dip(matrix_tf, 1, 1))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((0., -1.4190, -1.4188), (1., -0.92, -0.91))
  def test_gaussian_log_density(self, z_mean, target_low, target_high):
    matrix = tf.ones(1)
    with self.test_session() as sess:
      test_value = sess.run(vae.gaussian_log_density(matrix, z_mean, 0.))[0]
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters(
      (1, 0., 0.1), (10, -82.9, -82.89))  # -82.893 = (10 - 1) * ln(10000)
  def test_total_correlation(self, num_dim, target_low, target_high):
    # Since there is no dataset, the constant should be (num_latent - 1)*log(N)
    z = tf.random.normal(shape=(10000, num_dim))
    z_mean = tf.zeros(shape=(10000, num_dim))
    z_logvar = tf.zeros(shape=(10000, num_dim))
    with self.test_session() as sess:
      test_value = sess.run(vae.total_correlation(z, z_mean, z_logvar))
      self.assertBetween(test_value, target_low, target_high)


if __name__ == "__main__":
  tf.test.main()
