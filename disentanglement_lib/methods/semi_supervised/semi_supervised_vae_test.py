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

"""Tests for the semi supervised training protocol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from disentanglement_lib.methods.semi_supervised import semi_supervised_vae  # pylint: disable=unused-import
import numpy as np
import tensorflow.compat.v1 as tf


class S2VaeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((0, 100., 100.01), (10, 100., 100.01),
                            (100, 100., 100.01), (101, 100., 100.01))
  def test_fixed_annealer(self, step, target_low, target_high):
    c_max = 100.
    iteration_threshold = 100
    test_value = semi_supervised_vae.fixed_annealer(c_max, step,
                                                    iteration_threshold)
    self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((0, 0., 0.01), (10, 10., 10.01),
                            (100, 100., 100.01), (101, 100., 100.01))
  def test_anneal_annealer(self, step, target_low, target_high):
    c_max = 100.
    iteration_threshold = 100
    with self.test_session() as sess:
      test_value = sess.run(
          semi_supervised_vae.annealed_annealer(c_max, step,
                                                iteration_threshold))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((0, 0., 0.01), (10, 0., 0.01),
                            (100, 0., 0.01), (101, 100., 100.01))
  def test_fine_tune_annealer(self, step, target_low, target_high):
    c_max = 100.
    iteration_threshold = 100
    with self.test_session() as sess:
      test_value = sess.run(
          semi_supervised_vae.fine_tune_annealer(c_max, step,
                                                 iteration_threshold))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters(
      (np.zeros([64, 10]), np.zeros([64, 5]), 221.806, 221.808),
      (np.zeros([64, 10]), np.zeros([64, 10]), 221.806*2, 221.808*2),
      (np.ones([64, 10]), np.zeros([64, 10]), 840.47, 840.49))
  # Values for this test are computed with averaging x - x * z +
  # log(1 + exp(-x)).
  def test_xent(self, rep_np, labels_np, target_low, target_high):
    representation = tf.convert_to_tensor(rep_np, dtype=np.float32)
    labels = tf.convert_to_tensor(labels_np, dtype=np.float32)
    with self.test_session() as sess:
      test_value = sess.run(
          semi_supervised_vae.supervised_regularizer_xent(
              representation, labels, factor_sizes=[1] * labels_np.shape[1]))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((np.zeros([64, 10]), np.zeros([64, 5]), 0., 0.1),
                            (np.zeros([64, 10]), np.zeros([64, 10]), 0., 0.1),
                            (np.array([[1., 0.], [0., 1.]]),
                             np.array([[0., 1.], [1., 0.]]), 0.125, 0.126))
  def test_cov_det(self, rep_np, labels_np, target_low, target_high):
    representation = tf.convert_to_tensor(rep_np, dtype=np.float32)
    labels = tf.convert_to_tensor(labels_np, dtype=np.float32)
    with self.test_session() as sess:
      test_value = sess.run(
          semi_supervised_vae.supervised_regularizer_cov(representation,
                                                         labels))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters((np.random.normal(
      loc=np.zeros([1, 10]), scale=np.ones([1, 10]), size=[100000, 10]), 0.,
                             0.01))
  def test_cov_rand(self, samples, target_low, target_high):
    samples = tf.convert_to_tensor(samples, dtype=np.float32)
    with self.test_session() as sess:
      test_value = sess.run(
          semi_supervised_vae.supervised_regularizer_cov(samples,
                                                         samples))
      self.assertBetween(test_value, target_low, target_high)

  @parameterized.parameters(
      (0.2, 0.6, 0.7))
  # The real mutual information for this test case is about 0.6589.
  # Mine is noisy.
  def test_mine(self, var, target_low, target_high):
    def _gen_x(batch_size):
      return np.sign(np.random.normal(0., 1., [batch_size, 1]))
    def _gen_y(x):
      return x + np.random.normal(0., np.sqrt(var), x.shape)

    x_ph = tf.placeholder(tf.float32, [None, 1])
    y_ph = tf.placeholder(tf.float32, [None, 1])
    loss, mine_op = semi_supervised_vae.mine(x_ph, y_ph)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(200):
        x_sample = _gen_x(10000)
        y_sample = _gen_y(x_sample)
        mi, _ = sess.run([loss, mine_op],
                         feed_dict={x_ph: x_sample, y_ph: y_sample})
      self.assertBetween(mi, target_low, target_high)


if __name__ == "__main__":
  tf.test.main()
